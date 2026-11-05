#!/usr/bin/env python3
"""Early-season dispersion check: why v11 trails the close in 2026-27 games 1-9.

  python research/early_dispersion_check.py      # writes diagnostics/early_dispersion/RESULTS_season.txt

Owner's request (2026-11-05): sanity-check the model and the early-season
picks, then explain why the model rates stable, strong teams (SAS, NYK)
below the market. Hindsight description only; nothing is fitted here and
nothing touches the ledgers, pages or model/*.json.

Inputs (each kept apart, never pooled):
  * native 2026-27 rows (data/nba_ledger.csv): lock P at the 5 h lock, v11
    family, the eight voided stale-log locks excluded;
  * rebuilt 2026-27 v11 rows (data/nba_rebuilt.csv, hindsight);
  * reconstructed rows (data/nba_reconstructed.csv, hindsight) for earlier
    seasons' games 1-9 (ESPN BET close) and 2025-26 games 10+ (DraftKings).

Readings:
  1. model - close log loss by sample and stage;
  2. roster continuity (share of last season's team minutes by players who
     have played for the team this season, strictly before the game for
     earlier seasons) against model - close log loss, in thirds;
  3. the per-team gap (model - close, signed toward the team) against the
     team's strength in the market;
  4. dispersion: SD of model and close logits, close ~ model slope and the
     outcome calibration slope, this season beside earlier early seasons;
  5. the rebuilt v11 margin decomposed into its terms under the shipped
     report-route coefficients (model/logit_avail.json), with last-season-
     only and this-season-only composite deltas; regressions of the close
     spread, the model margin and the actual margin on those parts.
± is 1 SE unless labelled.
"""
from __future__ import annotations

import json
import os
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import nba_composite as nc                        # noqa: E402
import player_availability as pav                 # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "early_dispersion")
DATA = os.path.join(ROOT, "data")
VOID = {401909897, 401909909, 401909910, 401909911, 401909916, 401909919,
        401909920, 401909921}
SEASON = 2027


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def lg(p):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return np.log(p / (1 - p))


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    se = np.sqrt(np.diag(e @ e / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    return " ".join(f"{v:+.3f} ± {s:.3f}" for v, s in zip(b, se))


def logit_slope(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None])
        w += np.linalg.solve(H, X.T @ (y - p))
    return w[1], float(np.sqrt(np.linalg.inv(H)[1, 1]))


def stage(df):
    mn = np.minimum(df["gp_home"], df["gp_away"])
    return np.where(mn == 0, "game 0", np.where(mn < 10, "games 1-9", "games 10+"))


def load():
    nat = pd.read_csv(os.path.join(DATA, "nba_ledger.csv"))
    nat = nat[nat["home_won"].notna() & nat["close_q_home"].notna()]
    reb = pd.read_csv(os.path.join(DATA, "nba_rebuilt.csv"))
    reb = reb[reb["home_won"].notna() & reb["close_q_home"].notna() & reb["p_home"].notna()]
    rec = pd.read_csv(os.path.join(DATA, "nba_reconstructed.csv"))
    rec = rec[rec["home_won"].notna() & rec["close_q_home"].notna() & rec["p_home"].notna()]
    for d in (nat, reb, rec):
        d["stage"] = stage(d)
    return nat, reb, rec


def gap_line(lab, p, q, y):
    d = ll(p, y) - ll(q, y)
    return (f"  {lab:44s} n {len(d):4d}  model - close {d.mean():+.4f} ± "
            f"{d.std() / np.sqrt(len(d)):.4f}  mean P {np.mean(p):.3f}  close q "
            f"{np.mean(q):.3f}  home won {np.mean(y):.3f}")


def section_gap(nat, reb, rec):
    L = ["== 1. Model - close log loss (positive = model worse; ± 1 SE)"]
    v = nat[nat["lock_model_tag"].str.endswith("_v11", na=False) & ~nat["game_id"].isin(VOID)]
    L.append(gap_line("native 2026-27, first snapshot (all tags)", nat["first_p_home"],
                      nat["close_q_home"], nat["home_won"]))
    L.append(gap_line("native 2026-27, v11 5 h lock, voids out", v["lock_p_home"],
                      v["close_q_home"], v["home_won"]))
    L.append(gap_line("rebuilt 2026-27 v11 (hindsight)", reb["p_home"], reb["close_q_home"],
                      reb["home_won"]))
    for (s, b, st), g in rec.groupby(["season", "close_book", "stage"]):
        if st == "game 0":
            continue
        L.append(gap_line(f"reconstructed {s - 1}-{str(s)[2:]} {b} {st}", g["p_home"],
                          g["close_q_home"], g["home_won"]))
    return L


def continuity_fn(prev, cur):
    pm = prev.groupby(["team", "player_id"])["minutes"].sum()
    tot = prev.groupby("team")["minutes"].sum()
    cur = cur[cur["minutes"] > 0]

    def f(team, date):
        if team not in tot.index:
            return np.nan
        now = set(cur[(cur["team"] == team) & (cur["date"] < date)]["player_id"])
        x = pm.loc[team]
        return float(x[x.index.isin(now)].sum() / tot[team])
    return f


def section_continuity(reb, rec):
    L = ["", "== 2. Roster continuity (lower of the two teams, strictly before the game) "
         "vs model - close log loss, games 1-9, in thirds (± 1 SE)"]
    box = {s: pd.read_csv(os.path.join(DATA, f"nba_box_{s}.csv")) for s in (2024, 2025, 2026, 2027)}
    for d, lab in ((reb, "rebuilt 2026-27 (DK)"),
                   (rec[rec["stage"] == "games 1-9"], "reconstructed 2024-25 + 2025-26 (ESPN BET)")):
        rows = []
        for s, g in d.groupby("season"):
            f = continuity_fn(box[s - 1], box[s])
            g = g.copy()
            g["cmin"] = [min(f(h, t), f(a, t)) for h, a, t in
                         zip(g["home"], g["away"], g["slate_date"])]
            rows.append(g)
        g = pd.concat(rows)
        g = g[g["stage"] != "game 0"]
        g["d"] = ll(g["p_home"], g["home_won"]) - ll(g["close_q_home"], g["home_won"])
        g["third"] = pd.qcut(g["cmin"], 3, labels=["low", "mid", "high"])
        L.append(f"  {lab}")
        for t, x in g.groupby("third", observed=True):
            L.append(f"    {t:5s} n {len(x):3d}  continuity {x['cmin'].min():.2f}-"
                     f"{x['cmin'].max():.2f}  model - close {x['d'].mean():+.4f} ± "
                     f"{x['d'].std() / np.sqrt(len(x)):.4f}")
    return L


def section_dispersion(reb, rec):
    L = ["", "== 4. Dispersion: SD of logits, close ~ model slope, outcome calibration "
         "slope (± 1 SE; 1 = calibrated, > 1 = model underconfident)"]
    sets = []
    for s in (2025, 2026):
        sets.append((f"{s - 1}-{str(s)[2:]} games 1-9 (ESPN BET, reconstructed)",
                     rec[(rec["season"] == s) & (rec["stage"] == "games 1-9")]))
    sets.append(("2025-26 games 10+ (DK, reconstructed)",
                 rec[(rec["season"] == 2026) & (rec["stage"] == "games 10+") &
                     (rec["close_book"] == "dk")]))
    sets.append(("2026-27 rebuilt v11 (DK)", reb))
    for lab, d in sets:
        x, q = lg(d["p_home"]), lg(d["close_q_home"])
        k, se = logit_slope(x, d["home_won"].to_numpy(float))
        L.append(f"  {lab:44s} n {len(d):4d}  SD model {x.std():.2f}  SD close {q.std():.2f}  "
                 f"close~model {np.polyfit(x, q, 1)[0]:.2f}  calibration {k:.2f} ± {se:.2f}")
    return L


def decompose(reb):
    """Rebuilt v11 rows with the margin split into the shipped report-route
    terms (av terms as the residual) and last/this-season-only deltas."""
    W = json.load(open(os.path.join(ROOT, "model", "weights.json")))
    M = json.load(open(os.path.join(ROOT, "model", "logit_avail.json")))
    coef = dict(zip(M["features"], M["coef"]))
    logs, prior = nc.load_logs(SEASON), nc.load_logs(SEASON - 1)
    lg3, pct0 = nc.league_3p(logs), nc.last_season_3p(prior)
    opening = nc.season_opening(logs)
    tal = pav.season_talent(SEASON, rookies=True)

    def comp(lh, ih, ph, la, ia, pa, rho):
        def side(log, i, pr):
            t = nc.fade_totals(log, i, pr, rho)
            return nc.features_from_totals(np.array([t[c] for c in nc.COLS], float),
                                           orb_pct=nc.is_orb_pct(W))
        return nc.composite(side(lh, ih, ph) - side(la, ia, pa), W["sd"], W["w"])

    rows = []
    for _, g in reb.iterrows():
        d, h, a = pd.Timestamp(g["slate_date"]), g["home"], g["away"]
        lh, la = logs[h], logs[a]
        ih = int((pd.to_datetime(lh["date"]) < d).sum())
        ia = int((pd.to_datetime(la["date"]) < d).sum())
        pct = lg3.get(d, np.nan)
        x = nc.fade_inputs(lh, ih, prior[h], la, ia, prior[a],
                           pct if np.isfinite(pct) else pct0, W)
        li = nc.logit_inputs(x["delta"], nc.rest_days(lh, ih, d), nc.rest_days(la, ia, d),
                             d, opening)
        v = dict(delta=x["delta"], b2b_net=li["b2b_net"], d_phase=li["d_phase"],
                 luck_def=x["luck_def"], talent_diff=tal(h, d) - tal(a, d),
                 ft_diff=x["ft_diff"])
        c = {k: coef[k] * val for k, val in v.items()}
        base = M["intercept"] + sum(c.values())
        rows.append(dict(
            slate_date=g["slate_date"], home=h, away=a, gp_home=ih, gp_away=ia,
            model=g["margin_home"], spread=-g["close_spread"],
            actual=g["home_pts"] - g["away_pts"], home_won=g["home_won"],
            p_home=g["p_home"], close_q_home=g["close_q_home"],
            **{"c_" + k: val for k, val in c.items()}, c_avail=g["margin_home"] - base,
            delta_last=comp(lh, 0, prior[h], la, 0, prior[a], 1.0),
            delta_now=comp(lh, ih, None, la, ia, None, 0.0) if min(ih, ia) > 0 else np.nan))
    return pd.DataFrame(rows), M


def section_teams(D):
    L = ["", "== 3. Per-team gap, rebuilt v11 (signed toward the team; points of margin "
         "and probability)"]
    rows = []
    for _, g in D.dropna(subset=["spread"]).iterrows():
        for t, s in ((g["home"], 1), (g["away"], -1)):
            rows.append((t, s * g["spread"], s * (g["model"] - g["spread"]),
                         s * (g["p_home"] - g["close_q_home"]),
                         s * (g["home_won"] - g["close_q_home"])))
    T = pd.DataFrame(rows, columns=["team", "market", "gap_pts", "gap_p", "res_p"]) \
        .groupby("team").mean().sort_values("gap_pts")
    L.append(f"  corr(team's mean market margin, model - market margin) "
             f"{np.corrcoef(T['market'], T['gap_pts'])[0, 1]:+.2f} (30 teams)")
    L.append(f"  corr(team's model - close P, result - close P) "
             f"{np.corrcoef(T['gap_p'], T['res_p'])[0, 1]:+.2f}")
    L.append("  team  market margin  model - market (pts)  model - close P  result - close P")
    for t, r in T.iterrows():
        L.append(f"  {t:4s}  {r['market']:+13.1f}  {r['gap_pts']:+20.1f}  {r['gap_p']:+15.3f}  "
                 f"{r['res_p']:+16.3f}")
    return L


def section_decomposition(D, M):
    L = ["", "== 5. Rebuilt v11 margin decomposition (shipped report-route coefficients; "
         "c_avail = margin - base terms; ± 1 SE)"]
    L.append("  coefficients " + ", ".join(f"{k} {v:.3f}" for k, v in
                                           zip(M["features"], M["coef"])) +
             f"; intercept {M['intercept']:.2f}; sigma {M['sigma']:.2f}")
    E = D.dropna(subset=["spread"])
    L.append(f"  n {len(E)}  SD model margin {E['model'].std():.2f}  SD close spread "
             f"{E['spread'].std():.2f}  corr {np.corrcoef(E['model'], E['spread'])[0, 1]:.3f}")
    L.append(f"  close spread = a + b·model           : {ols(E['spread'].values, [E['model']])}")
    L.append(f"  actual       = a + b·model           : {ols(E['actual'].values, [E['model']])}")
    L.append(f"  actual       = a + b·close spread    : {ols(E['actual'].values, [E['spread']])}")
    L.append(f"  actual       = a + b·model + c·close : "
             f"{ols(E['actual'].values, [E['model'], E['spread']])}")
    F = E.dropna(subset=["delta_now"])
    parts = [F["delta_last"], F["delta_now"], F["c_talent_diff"]]
    L.append(f"  games with both teams past game 0: n {len(F)}; terms a + "
             "b·delta_last (win-% pts) + c·delta_now + d·talent (model pts)")
    for lab, col in (("close spread", "spread"), ("model margin", "model"),
                     ("actual margin", "actual")):
        L.append(f"    {lab:14s}: {ols(F[col].values, parts)}")
    for team in ("SAS", "NYK"):
        s = E[(E["home"] == team) | (E["away"] == team)]
        sg = np.where(s["home"] == team, 1.0, -1.0)
        cols = ["model", "spread", "actual", "c_delta", "c_talent_diff", "c_luck_def",
                "c_ft_diff", "c_b2b_net", "c_d_phase", "c_avail"]
        T = s[cols].mul(sg, axis=0)
        home = np.where(s["home"] == team, M["intercept"], -M["intercept"])
        L.append(f"  {team} ({len(s)} games, signed toward {team}): " + ", ".join(
            f"{c} {T[c].mean():+.2f}" for c in cols) + f", home court {home.mean():+.2f}")
    return L


def main():
    nat, reb, rec = load()
    L = ["Early-season dispersion check (2026-11-05): v11 against the close, 2026-27.",
         "Hindsight description; nothing fitted. Native, rebuilt and reconstructed kept apart.",
         ""]
    L += section_gap(nat, reb, rec)
    L += section_continuity(reb, rec)
    D, M = decompose(reb)
    L += section_teams(D)
    L += section_dispersion(reb, rec)
    L += section_decomposition(D, M)
    os.makedirs(OUT, exist_ok=True)
    D.to_csv(os.path.join(OUT, "season_decomposition.csv"), index=False, float_format="%.4f")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "RESULTS_season.txt"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
