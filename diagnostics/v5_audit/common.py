"""v5 audit: shared walk-forward folds and statistics (HANDOFF_v5_audit.md).

Research only. Nothing here changes model/*.json, the routing in
build_site.score_game, the ledger or MODEL_TAG.

A fold for test season Y re-fits everything on seasons strictly before Y,
exactly as backfill_history.reconstruct_season(walk_forward=True, v5=True):

  * composite weights: nc.fit_weights on WEIGHT_YEARS < Y;
  * base logit (V5_FEATURES) on PHASE_YEARS < Y, games with every term;
  * availability logit (FEATURES_V5) on REPORT_YEARS < Y, covered games;
  * frozen-v4 fallback (PHASE_FEATURES) on PHASE_YEARS < Y;
  * test games 10+ routed as production: avail (report covers it, every
    term finite) > base (every v5 term finite) > v4.

Each season's games are built ONCE with the raw four-factor gaps (home -
away, 8 features before standardising) and the per-feature change that
opponent-3P% regression makes; delta and luck_def are linear in the
composite weights (nc.composite), so each fold computes them by a dot
product instead of rebuilding the season. `verify` checks this against
nc.build_games.

Extra per-game columns used by the tasks (all pregame, games before the
date only):
  ftp_gap      own decayed FT% (FTM/FTA), home - away       (Task D)
  ftmfga_gap   own decayed FTM/FGA, home - away             (Task D)
  newly_out_diff, returning_diff, returning_listed_diff     (Task E)
"""
from __future__ import annotations

import math
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import backfill_history as bf  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402
import nba_composite as nc  # noqa: E402
import player_availability as pav  # noqa: E402

OUT = os.path.join(HERE, "output")
GAPS = [f"g{k}" for k in range(8)]          # raw factor gaps, nc.FEATURES order
LUCKS = [f"l{k}" for k in range(8)]         # opponent-3P% regression change, h - a
ZS = [f"z{k}" for k in range(8)]            # gaps / fold sd (standardised)
V5 = nc.V5_FEATURES
AV5 = pav.FEATURES_V5
V4 = nc.PHASE_FEATURES
E_COLS = ["newly_out_diff", "returning_diff", "returning_listed_diff"]


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------ season data --
def _totals(lg, i, cols, hl=nc.HALF_LIFE):
    A = lg[cols].to_numpy(float)[:i]
    w = 0.5 ** (np.arange(i)[::-1] / hl)
    return dict(zip(cols, (A * w[:, None]).sum(0)))


def season_table(y, talent=None):
    """Games 10+ of season y with weight-free inputs (see module doc)."""
    logs = nc.load_logs(y)
    opening = nc.season_opening(logs)
    lg3 = nc.league_3p(logs)
    feat, tot = {}, {}

    def side(tm, i):
        if (tm, i) not in feat:
            lg = logs[tm]
            t = _totals(lg, i, nc.COLS)
            feat[(tm, i)] = nc.features_from_totals(np.array([t[c] for c in nc.COLS]))
            tot[(tm, i)] = t
        return feat[(tm, i)], tot[(tm, i)]

    def luck_vec(tm, i, pct):
        lg = logs[tm]
        if i <= 0 or not np.isfinite(pct) or not set(nc.X3) <= set(lg.columns):
            return np.full(8, np.nan)
        A = lg[nc.COLS + nc.X3].to_numpy(float)[:i]
        if not np.isfinite(A).all():
            return np.full(8, np.nan)
        t = _totals(lg, i, nc.COLS + nc.X3)
        r = nc.regress_3p(t, "O", pct)
        f = nc.features_from_totals(np.array([t[c] for c in nc.COLS], float))
        fr = nc.features_from_totals(np.array([r[c] for c in nc.COLS], float))
        return fr - f

    rows = []
    for tm, c in logs.items():
        for i in np.where(c["home"].values)[0]:
            r = c.iloc[i]
            opp = r["opp"]
            if opp not in logs:
                continue
            c2 = logs[opp]
            j = np.where(c2["date"].values == r["date"])[0]
            if not len(j):
                continue
            j = int(j[0])
            if i < nc.MIN_GAMES or j < nc.MIN_GAMES:
                continue
            fh, th = side(tm, i)
            fa, ta = side(opp, j)
            pct = lg3.get(pd.Timestamp(r["date"]), float("nan"))
            li = nc.logit_inputs(0.0, nc.rest_days(c, i), nc.rest_days(c2, j),
                                 r["date"], opening)
            tal = (talent(tm, r["date"]) - talent(opp, r["date"])) if talent \
                else float("nan")
            with np.errstate(invalid="ignore", divide="ignore"):
                ftp = th["TFT"] / th["TFTA"] - ta["TFT"] / ta["TFTA"]
                ftmfga = th["TFT"] / th["TFGA"] - ta["TFT"] / ta["TFGA"]
            rows.append({"year": y, "date": pd.Timestamp(r["date"]),
                         "home": tm, "away": opp, "gp_home": i, "gp_away": j,
                         "b2b_net": li["b2b_net"], "phase": li["phase"],
                         "talent_diff": tal, "win": int(r["pts"] > r["opp_pts"]),
                         "ftp_gap": ftp, "ftmfga_gap": ftmfga,
                         **dict(zip(GAPS, fh - fa)),
                         **dict(zip(LUCKS, luck_vec(tm, i, pct) - luck_vec(opp, j, pct)))})
    df = pd.DataFrame(rows)
    df["slate_date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def with_weights(tab, weights):
    """delta, d_phase, luck_def and z-gaps for one fold's composite weights."""
    t = tab.copy()
    sd, w = np.asarray(weights["sd"]), np.asarray(weights["w"])
    G = t[GAPS].to_numpy(float) / sd
    t["delta"] = 100 * G @ w
    t["d_phase"] = t["delta"] * t["phase"]
    t["luck_def"] = 100 * (t[LUCKS].to_numpy(float) / sd) @ w
    for k, z in enumerate(ZS):
        t[z] = G[:, k]
    return t


# --------------------------------------------- Task E: stale talent terms --
def stale_talent(t, archive, cache_dir=pav.DEFAULT_CACHE):
    """(slate_date, home, away) -> newly_out / returning sums (home - away),
    same per-player value as talent_diff (role(pid, date) x value).

    newly_out: players on the team's previous box score (minutes > 0) who are
      Out/Doubtful on the report v5's availability fit uses (last report at
      least LEAD_MINUTES before tip).
    returning: players who played for the team earlier this season, are not
      on the previous box score, are not Out/Doubtful on that report, and
      whose latest appearance before the date (any team) was for this team
      -- the roster approximation (no roster data: a player waived or sent
      down without a later appearance elsewhere still counts).
    returning_listed: returning AND listed on that report with a status other
      than Out/Doubtful (the report is evidence he is on the roster).
    Never uses the game's own box score."""
    box = pav.fetch_box(t, cache_dir=cache_dir)
    bpm = pav.load_bpm(t - 1)
    value, _, _ = pav.player_values(box, bpm)
    role = pav.arrival_roles(box, bpm)
    st, s = pav.game_statuses(box, pav.fetch_tips(t, cache_dir), archive,
                              pav.LEAD_MINUTES)
    archive.save()
    od = {(g, p) for g, p, x in zip(st["game_id"], st["player_id"], st["status"])
          if x in ("Out", "Doubtful")}
    ok_listed = {(g, p) for g, p, x in zip(st["game_id"], st["player_id"], st["status"])
                 if x not in ("Out", "Doubtful")}
    played = box[box["minutes"] > 0]
    apps = defaultdict(list)                 # pid -> [(date, team)] sorted
    for r in played.sort_values("date").itertuples(index=False):
        apps[r.player_id].append((pd.Timestamp(r.date), r.team))

    def last_team(pid, date):
        team = None
        for d, tm in apps.get(pid, ()):
            if d >= date:
                break
            team = tm
        return team

    games = (box.drop_duplicates(["game_id", "team"])
             [["game_id", "date", "team", "home"]].copy())
    games["date"] = pd.to_datetime(games["date"])
    on_box = {k: set(g["player_id"]) for k, g in played.groupby(["game_id", "team"])}
    vals = {}
    for tm, g in games.sort_values(["date", "game_id"]).groupby("team"):
        seen = set()
        prev = None
        for r in g.itertuples(index=False):
            date = pd.Timestamp(r.date)
            prev_set = on_box.get((prev, tm), set()) if prev else set()
            no = sum(role(p, date) * value.get(p, 0.0) for p in prev_set
                     if (r.game_id, p) in od)
            ret = [p for p in seen - prev_set
                   if (r.game_id, p) not in od and last_team(p, date) == tm]
            rv = sum(role(p, date) * value.get(p, 0.0) for p in ret)
            rl = sum(role(p, date) * value.get(p, 0.0) for p in ret
                     if (r.game_id, p) in ok_listed)
            vals[(r.game_id, tm)] = (no, rv, rl, bool(r.home), date)
            seen |= on_box.get((r.game_id, tm), set())
            prev = r.game_id
    rows = []
    for gid, g in games.groupby("game_id"):
        h = g[g["home"].astype(bool)]
        a = g[~g["home"].astype(bool)]
        if len(h) != 1 or len(a) != 1:
            continue
        vh, va = vals.get((gid, h["team"].iloc[0])), vals.get((gid, a["team"].iloc[0]))
        if vh is None or va is None:
            continue
        rows.append(dict(slate_date=vh[4].strftime("%Y-%m-%d"),
                         home=h["team"].iloc[0], away=a["team"].iloc[0],
                         newly_out_diff=vh[0] - va[0], returning_diff=vh[1] - va[1],
                         returning_listed_diff=vh[2] - va[2]))
    return pd.DataFrame(rows), s


# ------------------------------------------------------------ fitting ------
def norm_cdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2)))


def norm_pdf(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / math.sqrt(2 * math.pi)


def fit(X, y, w=None, offset=None, l2=1e-3, pen=None, link="logit", iters=100):
    """Penalised GLM for P(y=1), Newton (logit) / Fisher scoring (probit).
    l2 matches nc.fit_logit (2*l2 on every coefficient, none on the
    intercept); `pen` (per coefficient) overrides it. Game weights `w`.
    Returns dict(intercept, coef, cov, link)."""
    X = np.asarray(X, float).reshape(len(y), -1)
    X1 = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    pv = np.full(X.shape[1], 2 * l2) if pen is None else np.asarray(pen, float)
    P = np.diag(np.concatenate([[0.0], pv]))
    b = np.zeros(X1.shape[1])
    for _ in range(iters):
        eta = X1 @ b + off
        if link == "logit":
            mu = 1 / (1 + np.exp(-eta))
            g = X1.T @ (w * (mu - y)) + P @ b
            H = X1.T @ (X1 * (w * mu * (1 - mu))[:, None]) + P
        else:
            mu = np.clip(norm_cdf(eta), 1e-10, 1 - 1e-10)
            d = norm_pdf(eta)
            g = -X1.T @ (w * (y - mu) * d / (mu * (1 - mu))) + P @ b
            H = X1.T @ (X1 * (w * d * d / (mu * (1 - mu)))[:, None]) + P
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-10:
            break
    return dict(intercept=float(b[0]), coef=b[1:].tolist(),
                cov=np.linalg.inv(H), link=link)


def predict(m, X, offset=None):
    X = np.asarray(X, float).reshape(len(X), -1)
    eta = m["intercept"] + X @ np.asarray(m["coef"], float)
    if offset is not None:
        eta = eta + offset
    return 1 / (1 + np.exp(-eta)) if m.get("link", "logit") == "logit" else norm_cdf(eta)


def finite(df, cols):
    return np.isfinite(df[cols].to_numpy(float)).all(axis=1)


# ------------------------------------------------------------ folds --------
class Fold:
    """Everything one walk-forward test season needs."""

    def __init__(self, y, weights, base_train, avail_train, v4_train, test):
        self.y, self.weights = y, weights
        self.base_train, self.avail_train, self.v4_train = base_train, avail_train, v4_train
        self.test = test

    def routed(self, fitter, base_feats=V5, avail_feats=AV5):
        """p for every test game under the production routing, with the base
        and availability logits fitted by `fitter(train, feats, route)` (a
        callable returning a predictor frame -> p); v4 rows keep the
        production v4 fallback."""
        t = self.test
        p = np.full(len(t), np.nan)
        r = t["route"].to_numpy()
        if (r == "avail").any():
            pa = fitter(self.avail_train, avail_feats, "avail")
            p[r == "avail"] = pa(t[r == "avail"])
        if (r == "base").any():
            pb = fitter(self.base_train, base_feats, "base")
            p[r == "base"] = pb(t[r == "base"])
        if (r == "v4").any():
            m = fit(self.v4_train[V4].to_numpy(float), self.v4_train["win"])
            p[r == "v4"] = predict(m, t.loc[r == "v4", V4].to_numpy(float))
        return p


def plain(train, feats, route):
    m = fit(train[feats].to_numpy(float), train["win"])
    return lambda d: predict(m, d[feats].to_numpy(float))


def build_folds(seasons, cache=pav.DEFAULT_CACHE, with_e=True):
    """{Y: Fold} for test seasons `seasons` (BBR years), all fits on earlier
    seasons only. Cached to output/folds.pkl."""
    os.makedirs(OUT, exist_ok=True)
    need = sorted({t for Y in seasons for t in bf.PHASE_YEARS if t <= Y} | set(seasons))
    tabs = {}
    for t in need:
        try:
            tal = pav.season_talent(t)
        except Exception as e:  # noqa: BLE001
            log(f"season {t}: no talent ({e!r}); v5 terms NaN -> v4 route")
            tal = None
        tabs[t] = season_table(t, tal)
        log(f"season {t}: {len(tabs[t])} games 10+")
    archive = pav.ReportArchive(os.path.join(cache, "injury_reports"))
    terms, stale = {}, {}
    try:
        for t in [t for t in bf.REPORT_YEARS if t <= max(seasons)]:
            terms[t], s = pav.season_terms(t, archive, cache)
            terms[t] = terms[t].drop_duplicates(["slate_date", "home", "away"])
            log(f"report season {t}: {len(s['covered'])}/{s['games']} covered")
            if with_e:
                stale[t], _ = stale_talent(t, archive, cache)
                stale[t] = stale[t].drop_duplicates(["slate_date", "home", "away"])
    finally:
        archive.save()
    key = ["slate_date", "home", "away"]
    for t in tabs:
        tab = tabs[t]
        if t in terms:
            tab = tab.merge(terms[t][key + ["av_min", "av_bpm"]], on=key, how="left")
        else:
            tab["av_min"] = tab["av_bpm"] = np.nan
        if t in stale:
            tab = tab.merge(stale[t], on=key, how="left")
        else:
            for c in E_COLS:
                tab[c] = np.nan
        tabs[t] = tab
    folds = {}
    for Y in seasons:
        wts = nc.fit_weights(bf.training_years(bf.WEIGHT_YEARS, Y, True))
        tw = {t: with_weights(tabs[t], wts) for t in tabs if t <= Y}
        ph = [t for t in bf.PHASE_YEARS if t < Y]
        rep = [t for t in bf.REPORT_YEARS if t < Y and t in terms]
        allb = pd.concat([tw[t] for t in ph], ignore_index=True)
        base_train = allb[finite(allb, V5)].reset_index(drop=True)
        v4_train = allb[finite(allb, V4)].reset_index(drop=True)
        if rep:
            alla = pd.concat([tw[t] for t in rep], ignore_index=True)
            avail_train = alla[finite(alla, AV5)].reset_index(drop=True)
        else:
            avail_train = None
        test = tw[Y].copy()
        test["route"] = np.where(finite(test, V5), "base", "v4")
        if avail_train is not None and len(avail_train):
            test.loc[finite(test, AV5), "route"] = "avail"
        f = Fold(Y, wts, base_train, avail_train, v4_train, test.reset_index(drop=True))
        f.test["p_v5"] = f.routed(plain)
        folds[Y] = f
        log(f"fold {Y}: base n={len(base_train)} avail n="
            f"{0 if avail_train is None else len(avail_train)}  routes "
            f"{f.test['route'].value_counts().to_dict()}")
    with open(os.path.join(OUT, "folds.pkl"), "wb") as fh:
        pickle.dump(folds, fh)
    return folds


def load_folds():
    with open(os.path.join(OUT, "folds.pkl"), "rb") as fh:
        return pickle.load(fh)


def verify(y, weights_path=os.path.join(ROOT, "model", "weights.json")):
    """Max |difference| of delta / luck_def / talent_diff between
    season_table + with_weights and nc.build_games (production weights)."""
    import json
    w = json.load(open(weights_path))
    tal = pav.season_talent(y)
    a = with_weights(season_table(y, tal), w)
    b = nc.build_games(y, w, talent=tal)
    b["slate_date"] = pd.to_datetime(b["date"]).dt.strftime("%Y-%m-%d")
    m = a.merge(b, on=["slate_date", "home", "away"], suffixes=("", "_ref"))
    out = {"n": len(m), "n_a": len(a), "n_b": len(b)}
    for c in ("delta", "luck_def", "talent_diff", "d_phase", "b2b_net"):
        d = (m[c] - m[c + "_ref"]).abs()
        out[c] = float(np.nanmax(d.to_numpy(float))) if len(d) else float("nan")
    return out


# ------------------------------------------------- market rows & stats -----
def with_market(test):
    """Attach the reconstructed row's result, open and close (one book)."""
    r = ledger.graded(ledger.load(ledger.RECON_PATH))
    r = r[["slate_date", "home", "away", "home_won", "close_q_home", "close_book",
           "open_home_ml", "open_away_ml", "close_home_ml", "close_away_ml"]].copy()
    for c in ("open_home_ml", "open_away_ml"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r["open_q_home"] = [market.devig(h, a) for h, a in zip(r["open_home_ml"],
                                                            r["open_away_ml"])]
    m = test.merge(r, on=["slate_date", "home", "away"], how="left")
    bad = m["home_won"].notna() & (m["home_won"].astype(float) != m["win"])
    if bad.any():
        log(f"WARNING: {int(bad.sum())} games where BBR and ESPN winners disagree")
    return m


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired(a, b, y):
    """(mean ll a - ll b, SE): candidate a vs reference b on the same games."""
    d = ll(a, y) - ll(b, y)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def slope(p, y):
    """Calibration: logit P(y) = a + b logit(p); (a, b, se_a, se_b)."""
    m = fit(logit(p)[:, None], y, l2=0.0)
    se = np.sqrt(np.diag(m["cov"]))
    return m["intercept"], m["coef"][0], se[0], se[1]


def books(m):
    """[(label, frame)] of graded rows, one closing book at a time."""
    out = []
    for (y, b), g in m[m["close_q_home"].notna()].groupby(["year", "close_book"]):
        out.append((f"{y - 1}-{str(y)[2:]} {market.BOOK_NAMES.get(b, b)}", g))
    return out


def picks(m, p_col, price="open"):
    """H2-H4 style value-side picks at `price`: one row per game with the
    picked side, its model P, no-vig q, edge, American price, result and
    unit P/L; side home/away and favourite flag (q > 0.5)."""
    q = m[f"{price}_q_home"].to_numpy(float)
    p = m[p_col].to_numpy(float)
    ok = np.isfinite(q) & np.isfinite(p) & m["home_won"].notna().to_numpy()
    d = m[ok]
    q, p = q[ok], p[ok]
    home = p > q
    keep = p != q
    hml = d[f"{price}_home_ml"].to_numpy(float)
    aml = d[f"{price}_away_ml"].to_numpy(float)
    won = d["home_won"].to_numpy(float)
    out = pd.DataFrame(dict(
        year=d["year"].to_numpy(), book=d["close_book"].to_numpy(),
        home=home, model_p=np.where(home, p, 1 - p), q=np.where(home, q, 1 - q),
        ml=np.where(home, hml, aml), won=np.where(home, won, 1 - won),
        close_q=np.where(home, d["close_q_home"], 1 - d["close_q_home"])))[keep]
    out["edge"] = out["model_p"] - out["q"]
    out["fav"] = out["q"] > 0.5
    out["pl"] = [market.unit_profit(a, b) for a, b in zip(out["ml"], out["won"])]
    out["be"] = [market.implied(a) for a in out["ml"]]
    return out


HYP = {"H2": dict(min_edge=0.0, fav=True), "H3": dict(min_edge=0.12, fav=False),
       "H4 (open proxy)": dict(min_edge=0.12, fav=False)}


def hyp_rows(pk, h):
    r = HYP[h]
    d = pk[pk["edge"] >= r["min_edge"]] if r["min_edge"] > 0 else pk[pk["edge"] > 0]
    if r["fav"]:
        d = d[d["fav"]]
    return d


def table(df, nd=4):
    """Markdown table (no tabulate dependency) for the log and RESULTS.md."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.{nd}f}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in r) + " |")
    return "\n".join(lines)
