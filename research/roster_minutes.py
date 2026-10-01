#!/usr/bin/env python3
"""Roster and expected minutes: one lineup-strength term for talent and availability.

  python research/roster_minutes.py --seasons 2025 2026 \
      [--box-seasons 2023 2024 2025 2026] [--lead-minutes 30]

Model v5 counts roster strength twice, from two partial views:
  talent_diff  sum over the team's PREVIOUS box score (minutes > 0) of mean
               minutes/48 x last-season BPM value. A player back from injury
               counts only after his first game back; one hurt since the
               last game still counts.
  av_bpm       the injury report's Out/Doubtful players' value lost against
               their usual participation. Newcomers with no minutes for the
               team are not counted in pregame mode (trade debuts).
Neither allocates the team's 240 regulation minutes: a missing starter's
minutes vanish instead of going to the players who replace him.

This tests a single replacement term, xtal (home - away), per team-game k:
  roster    who could play: `prev` = listed on game k-1's box score (the v5
            view); `listed` = listed on game k's own box score (dressed or
            inactive), a proxy for the roster known before tip (trades and
            signings are announced before the game). The script prints how
            often report Out players appear in that listing: if they do,
            the listing is the roster and does not reveal who sits.
  expected  P(plays): `od` Out/Doubtful -> 0, else 1; `q` Out -> 0,
            Doubtful/Questionable/Probable -> their play rates in the
            TRAINING seasons; `hind` = who actually played (hindsight
            ceiling, not pregame).
  minutes   raw = decayed (half-life 25) mean minutes when he played for
            this team before k; a newcomer gets his mean minutes this season
            before the date for any team, else last season's MP/G
            (player_availability.arrival_roles). raw x P(plays) is scaled so
            the team's expected minutes sum to 240, no player above CAP.
  xtal      sum of expected minutes/48 x value (last-season BPM above
            replacement, shrunk; unmatched players = replacement, 0).
  position  `list_od_pos`: the listed roster is first allocated 240 as if
            everyone played (the same base as list_od). Each Out/Doubtful
            player's allocated minutes then go to the available players in
            proportion to their minutes x max(0, 1 - |pos - pos_out| / 2)
            instead of in proportion to minutes alone (list_od), on
            PG=1 SG=2 SF=3 PF=4 C=5
            (hybrids average: PF-C = 4.5). So a C's minutes go to C and PF
            (weight 1, 0.5), a PG's to PG and SG, an SF's to SF, SG and PF.
            Positions are last season's BBR Pos (no lookahead); a player
            without one gets weight 0.5 as a teammate and spreads like
            list_od when missing. If no teammate is near, proportional.

Arms (each a games-10+ logit; the same covered games, fit on the same
training rows; every feature uses games strictly before the date and the
report at least --lead-minutes before tip, except `hind`):
  v4          FEATURES_V4 (no talent / luck)
  v5          FEATURES_V5: the shipped availability logit (reference)
  x_prev_od   delta, b2b_net, d_phase, luck_def, xtal (prev roster, od)
  x_list_od   same, listed roster
  x_list_q    same, listed roster, q participation
  x_list_pos  same as x_list_od, minutes redistributed by position
  x_hind      same, who played (hindsight ceiling)
  v5_x        FEATURES_V5 + xtal (listed, od): does xtal add beyond v5?
The x_* arms drop talent_diff, av_bpm and av_min, so no adjustment is
counted twice.

Walk-forward: weights on WEIGHT_YEARS < Y; every logit and the q rates on
box seasons < Y. Reported per test season and closing book on identical
games: log loss and Brier vs v5 and vs the close (paired, ± 95%), and each
arm's fitted terms. Research only: no change to the model, the ledger or
MODEL_TAG. Per-game output in research/output/roster_minutes.csv.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backfill_history as bf  # noqa: E402
import ledger  # noqa: E402
import market as mk  # noqa: E402
import nba_composite as nc  # noqa: E402
import player_availability as pav  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output",
                   "roster_minutes.csv")
TEAM_MINUTES = 240.0     # regulation: 5 players x 48
CAP = 42.0               # most minutes one player is expected to play
VARIANTS = ("prev_od", "list_od", "list_q", "list_od_pos", "hind")
POSITIONS = {"PG": 1.0, "SG": 2.0, "SF": 3.0, "PF": 4.0, "C": 5.0,
             "G": 1.5, "F": 3.5}
UNKNOWN_POS_WEIGHT = 0.5
X_BASE = ["delta", "b2b_net", "d_phase", "luck_def"]
ARMS = {
    "v4": list(pav.FEATURES_V4),
    "v5": list(pav.FEATURES_V5),
    "x_prev_od": X_BASE + ["x_prev_od"],
    "x_list_od": X_BASE + ["x_list_od"],
    "x_list_q": X_BASE + ["x_list_q"],
    "x_list_pos": X_BASE + ["x_list_od_pos"],
    "x_hind": X_BASE + ["x_hind"],
    "v5_x": list(pav.FEATURES_V5) + ["x_list_od"],
}
REF = "v5"


def allocate(e, total=TEAM_MINUTES, cap=CAP):
    """Scale expected raw minutes e (>= 0) to sum to `total`, none above
    `cap` (excess goes to the others, in proportion). With too few players
    to fill `total` under the cap, each gets the cap."""
    e = np.asarray(e, float)
    m = np.zeros_like(e)
    free = e > 0
    left = total
    for _ in range(len(e) + 1):
        s = e[free].sum()
        if s <= 0 or left <= 0:
            break
        m[free] = left * e[free] / s
        over = free & (m > cap)
        if not over.any():
            break
        m[over] = cap
        left -= cap * over.sum()
        free &= ~over
    return m


def pos_number(s):
    """BBR Pos -> PG=1 .. C=5; hybrids ('PF-C', 'SG-PG') average; else NaN."""
    vals = [POSITIONS[t] for t in re.split(r"[-/ ,]+", str(s).strip().upper())
            if t in POSITIONS]
    return float(np.mean(vals)) if vals else float("nan")


def parse_positions(html):
    """BBR NBA_{y}_advanced.html -> {norm name: position number}, the row
    with the most minutes (multi-team total) as in parse_advanced."""
    m = re.search(r'<table[^>]*id="advanced(?:_stats)?".*?</table>',
                  nc.uncomment(html), re.S)
    if not m:
        return {}
    df = pd.read_html(io.StringIO(m.group(0)))[0]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]
    if "Pos" not in df:
        return {}
    df = df[df["Player"].notna() & (df["Player"] != "Player")].copy()
    df["MP"] = pd.to_numeric(df["MP"], errors="coerce").fillna(0)
    df["key"] = df["Player"].map(pav.norm_name)
    df = df.sort_values("MP", ascending=False).drop_duplicates("key")
    out = {k: pos_number(p) for k, p in zip(df["key"], df["Pos"])}
    return {k: v for k, v in out.items() if np.isfinite(v)}


def load_positions(y):
    """Season y's positions (use y - 1 for season y's games)."""
    html = nc.fetch(f"{nc.BASE}/leagues/NBA_{y}_advanced.html",
                    nc.CACHE / f"advanced_{y}.html")
    return parse_positions(html)


def positional_fill(e, miss, pos, total=TEAM_MINUTES, cap=CAP):
    """Minutes e of the available players plus the minutes `miss` of the
    missing ones (e + miss: the team allocated as if everyone played).
    Each missing player's minutes go to the available players in
    proportion to e x max(0, 1 - |pos - pos_missing| / 2)
    (UNKNOWN_POS_WEIGHT for a teammate without a position; in proportion
    to e when the missing player has none or nobody is near). Then scaled
    to `total` and capped as in allocate."""
    e, miss, pos = (np.asarray(x, float) for x in (e, miss, pos))
    if miss.sum() <= 0 or e.sum() <= 0:
        return allocate(e, total, cap)
    add = np.zeros_like(e)
    for i in np.where(miss > 0)[0]:
        if np.isfinite(pos[i]):
            with np.errstate(invalid="ignore"):
                k = np.where(np.isfinite(pos),
                             np.clip(1 - np.abs(pos - pos[i]) / 2, 0, None),
                             UNKNOWN_POS_WEIGHT)
            wgt = e * k
        else:
            wgt = e
        if wgt.sum() <= 0:
            wgt = e
        add += miss[i] * wgt / wgt.sum()
    return allocate(e + add, total, cap)


def team_lineups(tg, value, arrival_role, presence, half_life=nc.HALF_LIFE,
                 pos=None):
    """One team-season of box rows -> {game_id: {variant: xtal}}.

    presence: {"od": {(game_id, player_id): P}, "q": {...}} from the injury
    report (player_availability.present_map); a rostered player not on the
    report plays (1). Game k's own minutes are read only by `hind`; the
    `list_*` variants read game k's listing (who is on the roster), never
    its minutes. pos: {player_id: position number} for list_od_pos."""
    games = (tg[["game_id", "date"]].drop_duplicates("game_id")
             .sort_values(["date", "game_id"]).reset_index(drop=True))
    order = {g: k for k, g in enumerate(games["game_id"])}
    players = sorted(tg["player_id"].unique())
    pidx = {p: j for j, p in enumerate(players)}
    n, P = len(games), len(players)
    M = np.zeros((n, P))
    L = np.zeros((n, P), bool)
    for r in tg.itertuples(index=False):
        k, j = order[r.game_id], pidx[r.player_id]
        M[k, j] = r.minutes
        L[k, j] = True
    v = np.array([value.get(p, 0.0) for p in players])
    pn = np.array([(pos or {}).get(p, np.nan) for p in players], float)
    out = {}
    for k in range(1, n):
        gid, date = games.at[k, "game_id"], games.at[k, "date"]
        w = 0.5 ** (np.arange(k)[::-1] / half_life)
        played = M[:k] > 0
        wp = (w[:, None] * played).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = np.where(wp > 0, (w[:, None] * M[:k]).sum(0) / wp, 0.0)
        for j in np.where(wp == 0)[0]:
            raw[j] = 48.0 * arrival_role(players[j], date)

        def p_of(arm):
            mp = presence.get(arm, {})
            return np.array([mp.get((gid, pl), 1.0) for pl in players])

        rost = {"prev": L[k - 1], "list": L[k]}
        res = {}
        for var in VARIANTS:
            if var == "hind":
                p = (M[k] > 0).astype(float)
            else:
                ro, arm = var.split("_")[:2]
                p = rost[ro] * p_of(arm)
            if var.endswith("_pos"):
                full = allocate(raw * rost[ro])      # as if everyone played
                m = positional_fill(full * p, full * (1 - p), pn)
            else:
                m = allocate(raw * p)
            res[var] = float((m / 48.0 * v).sum()) if m.sum() > 0 else np.nan
        out[gid] = res
    return out


def game_lineups(box, value, arrival_role, presence, pos=None):
    """Box rows -> one row per game: slate_date, home, away, x_<variant>
    (home - away)."""
    feats = {}
    for tm, tg in box.groupby("team"):
        for gid, r in team_lineups(tg, value, arrival_role, presence,
                                     pos=pos).items():
            feats[(gid, tm)] = r
    g = box[box["home"]].drop_duplicates("game_id")[["game_id", "date", "team", "opp"]]
    rows = []
    for r in g.itertuples(index=False):
        h, a = feats.get((r.game_id, r.team)), feats.get((r.game_id, r.opp))
        if h is None or a is None:
            continue
        rows.append(dict(game_id=r.game_id,
                         slate_date=pd.Timestamp(r.date).strftime("%Y-%m-%d"),
                         home=r.team, away=r.opp,
                         **{f"x_{k}": h[k] - a[k] for k in VARIANTS}))
    return pd.DataFrame(rows)


def listing_check(st, box):
    """Share of report rows of each status whose player is on that game's
    box listing (any minutes): ~100% for Out means the listing is the
    roster, not the players who dressed."""
    listed = set(zip(box["game_id"], box["player_id"]))
    out = {}
    for status, g in st.groupby("status"):
        on = [(a, b) in listed for a, b in zip(g["game_id"], g["player_id"])]
        out[status] = (float(np.mean(on)), len(g))
    return out


def season(t, archive, lead_minutes, cache_dir=pav.DEFAULT_CACHE):
    """Everything per box season that does not depend on the test season."""
    box = pav.fetch_box(t, cache_dir=cache_dir)
    bpm = pav.load_bpm(t - 1)
    value, hit, hit_min = pav.player_values(box, bpm)
    role = pav.arrival_roles(box, bpm)
    names = box.drop_duplicates("player_id").set_index("player_id")["name"]
    pmap = load_positions(t - 1)
    pos = {pid: pmap[pav.norm_name(nm)] for pid, nm in names.items()
           if pav.norm_name(nm) in pmap}
    mins = box.groupby("player_id")["minutes"].sum()
    pos_min = float(mins[mins.index.isin(list(pos))].sum() / max(mins.sum(), 1.0))
    st, s = pav.game_statuses(box, pav.fetch_tips(t, cache_dir), archive, lead_minutes)
    archive.save()
    av = pav.game_availability(box, value, present=pav.present_map(st, "od"))
    av = av[av["game_id"].isin(s["covered"])]
    print(f"season {t}: {len(s['covered'])}/{s['games']} games covered by the "
          f"report; BPM matches {100 * hit:.0f}% of players, {100 * hit_min:.0f}% "
          f"of minutes; last-season position for {100 * pos_min:.0f}% of minutes",
          flush=True)
    chk = listing_check(st, box)
    print("  on that game's box listing: " + ", ".join(
        f"{k} {100 * v[0]:.0f}% (n={v[1]})" for k, v in sorted(chk.items())), flush=True)
    return dict(box=box, value=value, role=role, st=st, pos=pos, covered=s["covered"],
                av=av[["game_id", "slate_date", "home", "away", "av_min", "av_bpm"]],
                talent=pav.talent_fn(box, value, role))


def season_frame(t, S, weights, rates):
    """Games 10+ of season t covered by the report, with every arm's terms."""
    presence = {"od": pav.present_map(S["st"], "od"),
                "q": pav.present_map(S["st"], "q", rates)}
    x = game_lineups(S["box"], S["value"], S["role"], presence, S.get("pos"))
    g = nc.build_games(t, weights, talent=S["talent"])
    g["slate_date"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")
    g = g.merge(S["av"], on=["slate_date", "home", "away"], how="inner")
    return g.merge(x.drop(columns="game_id"), on=["slate_date", "home", "away"],
                   how="inner")


def common(df):
    feats = sorted({f for fs in ARMS.values() for f in fs})
    return df[np.isfinite(df[feats].to_numpy(float)).all(axis=1)]


def paired(a, b, y):
    la, lb = mk.logloss(a, y), mk.logloss(b, y)
    ba, bb = mk.brier(a, y), mk.brier(b, y)
    n = len(y)
    se = (lambda d: float(d.std(ddof=1) / np.sqrt(n))) if n > 1 else (lambda d: np.nan)
    return (float(la.mean()), float((la - lb).mean()), 1.96 * se(la - lb),
            float((ba - bb).mean()), 1.96 * se(ba - bb))


def report(m):
    lines = []
    for (yr, book), g in m.groupby(["year", "close_book"]):
        y = g["home_won"].to_numpy(float)
        q = g["close_q_home"].to_numpy(float)
        lines.append(f"\n{yr} {mk.BOOK_NAMES.get(book, book)} close · games 10+, "
                     f"report-covered (n={len(g)})")
        if len(g) < 30:
            lines.append("  (fewer than 30 games: not scored)")
            continue
        lq = float(mk.logloss(q, y).mean())
        lines.append(f"  {'close':10s} logloss {lq:.4f}")
        ref = g[f"p_{REF}"].to_numpy(float)
        for arm in ARMS:
            p = g[f"p_{arm}"].to_numpy(float)
            la, dm, dmc, _, _ = paired(p, q, y)
            if arm == REF:
                lines.append(f"  {arm:10s} logloss {la:.4f}  vs close {dm:+.4f} ± {dmc:.4f}")
                continue
            _, d, dc, br, brc = paired(p, ref, y)
            tag = "  (hindsight)" if arm == "x_hind" else ""
            lines.append(f"  {arm:10s} logloss {la:.4f}  vs {REF} {d:+.4f} ± {dc:.4f} "
                         f"(Brier {br:+.4f} ± {brc:.4f})  vs close {dm:+.4f} ± {dmc:.4f}"
                         + tag)
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--box-seasons", nargs="+", type=int, default=bf.REPORT_YEARS)
    ap.add_argument("--lead-minutes", type=int, default=pav.LEAD_MINUTES)
    a = ap.parse_args(argv)
    archive = pav.ReportArchive()
    try:
        S = {t: season(t, archive, a.lead_minutes) for t in a.box_seasons}
    finally:
        archive.save()
    recon = ledger.graded(ledger.load(ledger.RECON_PATH))
    recon = recon[["slate_date", "home", "away", "close_q_home", "close_book",
                   "home_won"]]
    allm = []
    for y in a.seasons:
        tr_years = bf.training_years(a.box_seasons, y, walk_forward=True)
        if not tr_years or y not in S:
            raise SystemExit(f"season {y}: needs its own and earlier box seasons "
                             f"(have {sorted(S)})")
        rates = pav.play_rates(pd.concat([S[t]["st"] for t in tr_years]))
        weights = nc.fit_weights(bf.training_years(bf.WEIGHT_YEARS, y, True))
        frames = {t: common(season_frame(t, S[t], weights, rates))
                  for t in tr_years + [y]}
        tr = pd.concat([frames[t] for t in tr_years], ignore_index=True)
        te = frames[y].copy()
        print(f"\n=== season {y}: logits on {tr_years} (n={len(tr)}), q rates " +
              ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(rates.items())) +
              f"; test games {len(te)}")
        for arm, feats in ARMS.items():
            fm = nc.fit_logit(tr[feats].to_numpy(float), tr["win"], feats)
            te[f"p_{arm}"] = nc.predict(fm, te[feats].to_numpy(float))
            print(f"  {arm:10s} " + "  ".join(
                f"{f}={c:+.4f}" for f, c in zip(fm["features"], fm["coef"])))
        sd = te[[f"x_{v}" for v in VARIANTS]].std()
        print("  xtal sd (home - away, points/48): " +
              ", ".join(f"{k[2:]} {v:.2f}" for k, v in sd.items()) +
              f"; corr(x_list_od, talent_diff) "
              f"{te['x_list_od'].corr(te['talent_diff']):+.2f}, "
              f"corr(x_list_od, av_bpm) {te['x_list_od'].corr(te['av_bpm']):+.2f}")
        dpos = te["x_list_od_pos"] - te["x_list_od"]
        print(f"  position vs proportional: sd of the difference {dpos.std():.3f}, "
              f"{100 * (dpos.abs() > 0.05).mean():.0f}% of games move > 0.05 point")
        m = te.merge(recon, on=["slate_date", "home", "away"], how="inner")
        print(f"  matched to reconstructed rows with a close: {len(m)}")
        allm.append(m)
    m = pd.concat(allm, ignore_index=True)
    print("\n== Same games, one book at a time (negative = better than v5 / "
          "the close; ± 95%)")
    print(report(m))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    m.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(m)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
