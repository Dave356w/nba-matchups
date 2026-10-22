"""Calibration and market comparisons, shared by the site and the report.

Every function takes graded ledger rows (ledger.graded) and uses ONLY rows
with a valid closing pair, so the model and the market are always compared on
identical games. Callers pass one basis (native or reconstructed) and one
closing book (`book_split`) at a time; nothing here pools either.
"""
from __future__ import annotations

import os
from statistics import NormalDist

import numpy as np
import pandas as pd

import market

PROB_BINS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)


def lean_is_home(g):
    """The recorded lean (the `lean` column) as a home/away mask.

    Every grade of "the lean" reads this one definition, so the record in the
    tiles, the ROI tables, the accuracy and the ATS lean always agree (a game
    at exactly p_home = 0.5 is the side the row recorded). Rows without a
    recorded lean fall back to p_home >= 0.5.
    """
    lean = g["lean"].astype(str)
    ph = pd.to_numeric(g["p_home"], errors="coerce")
    home = (lean == g["home"].astype(str)).to_numpy()
    known = (home | (lean == g["away"].astype(str)).to_numpy())
    return np.where(known, home, (ph >= 0.5).to_numpy())


def price_q_home(g, price):
    """No-vig home probability at one price ("pre", "open" or "close")."""
    col = f"{price}_q_home"
    if col in g.columns:
        return pd.to_numeric(g[col], errors="coerce")
    return pd.Series([market.devig(h, a) for h, a in
                      zip(g[f"{price}_home_ml"], g[f"{price}_away_ml"])],
                     index=g.index, dtype=float)


def with_close(g):
    """Graded rows with a usable close; adds q/ml/won for the lean side."""
    if not len(g):
        return g
    q = pd.to_numeric(g["close_q_home"], errors="coerce")
    hml = pd.to_numeric(g["close_home_ml"], errors="coerce")
    aml = pd.to_numeric(g["close_away_ml"], errors="coerce")
    ok = q.between(0, 1, inclusive="neither") & hml.notna() & aml.notna() \
        & pd.to_numeric(g["p_home"], errors="coerce").notna()
    h = g.loc[ok].copy()
    lean_home = lean_is_home(h)
    h["lean_home"] = lean_home
    h["lean_q"] = np.where(lean_home, h["close_q_home"], 1 - h["close_q_home"])
    h["lean_ml"] = np.where(lean_home, h["close_home_ml"], h["close_away_ml"])
    h["lean_p"] = np.where(lean_home, h["p_home"], 1 - h["p_home"])
    h["lean_won"] = np.where(lean_home, h["home_won"], 1 - h["home_won"])
    return h


# The bases market_view prices by Kalshi: none. Live (native) and rebuilt
# games are graded at their sportsbook, DraftKings (owner's decisions,
# 2026-10-21; Kalshi was the pages' market from 2026-10-06). Kalshi's prices
# stay in the data (kalshi_* columns; research/kalshi_benchmark.py) and on the
# Preseason page; naming a basis here grades it by Kalshi again.
KALSHI_BASES = ()


def market_view(g):
    """The ledger as the pages grade it. Rows of a basis in KALSHI_BASES
    (none since 2026-10-21) are Kalshi-priced when graded with a Kalshi
    close, or still pending with a Kalshi pregame price; such a row reads
    every price from Kalshi (pre, first, open, close; book "kalshi") and
    keeps its model columns, result and sportsbook closing spread (the ATS
    line; `spread_book` keeps the sportsbook that priced it). Every other
    row keeps its sportsbook prices, so a row is always priced by one
    market. The pre-registered HYPOTHESES read the raw rows: they were
    registered on sportsbook prices."""
    if g is None or not len(g) or "kalshi_close_q_home" not in g.columns:
        return g
    num = lambda c: pd.to_numeric(g[c], errors="coerce")
    won = num("home_won").isin([0, 1])
    k = (won & num("kalshi_close_q_home").between(0, 1, inclusive="neither")) | (
        ~won & num("kalshi_pre_q_home").between(0, 1, inclusive="neither"))
    k &= g["basis"].astype(str).isin(KALSHI_BASES)
    if not k.any():
        return g
    v = g.copy()
    for c in ("pre_book", "close_book", "first_book"):
        v[c] = v[c].astype(object)
    v["spread_book"] = v["close_book"]
    for kind in ("pre", "first", "close", "open"):
        rows = k & won if kind in ("close", "open") else k   # closes: graded only
        cols = ([f"{kind}_home_ml", f"{kind}_away_ml"]
                + ([f"{kind}_q_home"] if kind != "open" else []))
        for c in cols:
            v.loc[rows, c] = num(f"kalshi_{c}")[rows]
    has_first = k & v["first_snapshot_utc"].notna()
    v.loc[k, "pre_book"] = "kalshi"
    v.loc[k & won, "close_book"] = "kalshi"
    v.loc[has_first, "first_book"] = "kalshi"
    return v


# Books the pages and the report leave out (owner's decisions, 2026-10-21):
# ESPN BET is no longer offered. The data keep its rows and prices; only what
# is shown is filtered (shown()).
HIDDEN_BOOKS = ("espnbet",)


def shown(items, key="book"):
    """Sections, season rows or (book, ...) tuples whose book is shown (not
    in HIDDEN_BOOKS)."""
    return [x for x in items
            if (x[0] if isinstance(x, tuple) else x[key]) not in HIDDEN_BOOKS]


def book_split(h):
    """[(book label, rows)] by closing book, in market.BOOKS order."""
    if h is None or not len(h):
        return []
    labels = [b for _, b in market.BOOKS] + list(market.EXCHANGES)
    return [(b, h[h["close_book"] == b]) for b in labels
            if (h["close_book"] == b).any()]


def _agg(p, won):
    p = np.asarray(p, float)
    won = np.asarray(won, float)
    n = len(p)
    if not n:
        return None
    act = float(won.mean())
    imp = float(p.mean())
    return dict(n=n, w=int(won.sum()), implied=imp, actual=act,
                diff=act - imp, se=market.excess_se(p))


def market_calibration(h):
    """Devigged close vs realised, per side and price rung. Grades the MARKET.

    Each game gives two observations (home at its close, away at its close).
    No pooled both-sides total: the two devigged sides sum to 1 and exactly
    one wins, so that total is 50% vs 50% by construction. The favourite pool
    asks once per game instead (pick'ems excluded).
    """
    if h is None or not len(h):
        return [], {}
    obs = []
    for _, r in h.iterrows():
        for side, ml, p, w in (
                ("home", r["close_home_ml"], r["close_q_home"], r["home_won"]),
                ("away", r["close_away_ml"], 1 - r["close_q_home"],
                 1 - r["home_won"])):
            rung = market.ladder_rung(ml)
            if rung is not None:
                obs.append((rung, side, float(p), int(w)))
    rows = []
    for _lo, _hi, label in market.ODDS_LADDER:
        at = [o for o in obs if o[0] == label]
        if not at:
            continue
        rows.append(dict(
            rung=label,
            home=_agg(*zip(*[(o[2], o[3]) for o in at if o[1] == "home"]))
            if any(o[1] == "home" for o in at) else None,
            away=_agg(*zip(*[(o[2], o[3]) for o in at if o[1] == "away"]))
            if any(o[1] == "away" for o in at) else None,
            all=_agg([o[2] for o in at], [o[3] for o in at]),
        ))
    home = [o for o in obs if o[1] == "home"]
    fav = [o for o in obs if o[2] > 0.5]
    totals = dict(
        home=_agg([o[2] for o in home], [o[3] for o in home]),
        favourite=_agg([o[2] for o in fav], [o[3] for o in fav]),
    )
    return rows, totals


def model_calibration(h, bins=PROB_BINS):
    """Model P(home win) bins vs realised, with the market's q on the same rows."""
    if h is None or not len(h):
        return []
    p = h["p_home"].to_numpy(float)
    q = h["close_q_home"].to_numpy(float)
    y = h["home_won"].to_numpy(float)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, len(bins) - 2)
    out = []
    for j in range(len(bins) - 1):
        m = idx == j
        if not m.any():
            continue
        out.append(dict(
            lo=bins[j], hi=bins[j + 1], n=int(m.sum()),
            model=float(p[m].mean()), market=float(q[m].mean()),
            actual=float(y[m].mean()), se=market.excess_se(p[m]),
        ))
    return out


def reliability(p, y, bins=PROB_BINS):
    """Stated probability vs realised rate, binned on p. `se` is the spread
    of the realised rate if p were right (market.excess_se), so a point more
    than ~2 se off the diagonal is a calibration miss, not noise."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, len(bins) - 2)
    out = []
    for j in range(len(bins) - 1):
        m = idx == j
        if m.any():
            out.append(dict(lo=bins[j], hi=bins[j + 1], n=int(m.sum()),
                            stated=float(p[m].mean()), actual=float(y[m].mean()),
                            se=market.excess_se(p[m])))
    return out


def scoring(h):
    """Brier, log loss and accuracy for model and market on identical rows.

    The paired differences (model - market; negative = model better) carry a
    per-game SE. With few games this is the honest summary of whether the
    composite adds anything the close did not already price.
    """
    if h is None or not len(h):
        return None
    p = h["p_home"].to_numpy(float)
    q = h["close_q_home"].to_numpy(float)
    y = h["home_won"].to_numpy(float)
    n = len(y)
    lean_home = lean_is_home(h)
    bm, bq = market.brier(p, y), market.brier(q, y)
    lm, lq = market.logloss(p, y), market.logloss(q, y)

    def se(d):
        return float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return dict(
        n=n,
        model=dict(brier=float(bm.mean()), logloss=float(lm.mean()),
                   acc=float((lean_home == (y == 1)).mean())),
        market=dict(brier=float(bq.mean()), logloss=float(lq.mean()),
                    acc=float(((q >= 0.5) == (y == 1)).mean())),
        d_brier=float((bm - bq).mean()), d_brier_se=se(bm - bq),
        d_logloss=float((lm - lq).mean()), d_logloss_se=se(lm - lq),
    )


def calibration_slope(p, y, min_n=30, iters=50):
    """(slope, se, intercept, se) of the outcome on logit(p): 1 / 0 for a
    calibrated forecast (slope < 1: too confident, > 1: too timid). NaNs
    below min_n rows, without both outcomes, or if the fit separates."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    k = np.isfinite(p) & np.isfinite(y)
    p, y = p[k], y[k]
    nan = (float("nan"),) * 4
    if len(y) < min_n or y.min() == y.max():
        return nan
    X = np.column_stack([np.ones(len(p)), np.log(p / (1 - p))])
    b = np.zeros(2)
    try:
        for _ in range(iters):
            mu = 1 / (1 + np.exp(-X @ b))
            H = X.T @ (X * (mu * (1 - mu))[:, None])
            step = np.linalg.solve(H, X.T @ (y - mu))
            b += step
            if np.abs(step).max() < 1e-10:
                break
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return nan
    if not np.isfinite(b).all() or abs(b[1]) > 50:
        return nan
    return float(b[1]), float(np.sqrt(cov[1, 1])), float(b[0]), float(np.sqrt(cov[0, 0]))


def lean_calibration(h):
    """The lean's mean model P against its win rate on graded rows:
    dict(n, predicted, actual, diff, se) with se = sqrt(sum p(1-p)) / n,
    the spread of the win rate if the model were right. None without rows."""
    if h is None or not len(h):
        return None
    p = h["p_home"].to_numpy(float)
    y = h["home_won"].to_numpy(float)
    home = lean_is_home(h)
    pl = np.where(home, p, 1 - p)
    won = np.where(home, y == 1, y == 0).astype(float)
    n = len(pl)
    return dict(n=n, predicted=float(pl.mean()), actual=float(won.mean()),
                diff=float(won.mean() - pl.mean()),
                se=float(np.sqrt((pl * (1 - pl)).sum()) / n))


def _lean_row(label, s):
    n = len(s)
    q = s["lean_q"].to_numpy(float)
    ml = s["lean_ml"].to_numpy(float)
    won = s["lean_won"].to_numpy(float)
    be = market.breakeven_prob(ml)
    units = np.array([market.unit_profit(m, w) for m, w in zip(ml, won)])
    wins = int(won.sum())
    return dict(
        band=label, n=n, w=wins, l=n - wins, win=wins / n,
        q=float(q.mean()), excess=float(won.mean() - q.mean()),
        se=market.excess_se(q), ev=float(won.mean() - be.mean()),
        ev_null=market.ev_null(q, be), units=float(np.nansum(units)),
        roi=float(np.nansum(units) / n), model_p=float(s["lean_p"].mean()),
    )


def lean_by_price(h):
    """The model's lean graded at the lean side's closing price, by rung.

    excess = win% - mean no-vig q (calibration vs the market);
    ev     = win% - mean break-even (at the posted price) and is centred on
             `ev_null` (= q - break-even, minus the hold), NOT on zero.
    Descriptive only: the bands are monitoring dimensions, not a filter.
    """
    if h is None or not len(h):
        return [], None
    rung = h["lean_ml"].map(market.ladder_rung)
    rows = []
    for _lo, _hi, label in market.ODDS_LADDER:
        s = h[rung == label]
        if len(s):
            rows.append(_lean_row(label, s))
    return rows, _lean_row("Pooled", h)


PICK_RULES = (
    ("lean", "Lean — the model's favourite"),
    ("value", "Value — side where model P > no-vig market P"),
)
# Rules graded on the pages and in the report. The generic value side is
# retired from display (owner's decision, 2026-10-06): on the reconstructed
# rows its claimed EV at the close was +11 to +16% in every book-season while
# it returned -6 to -11% (null about -4%), and it is picked against the close
# itself. picks() still builds it: the pre-registered HYPOTHESES are
# value-side rules at bettable prices and stay on the scoreboard.
SHOWN_RULES = ("lean",)
EARLY_BELOW = 10     # nba_composite.MIN_GAMES: below it v2 uses the carryover model
# The share of the model's disagreement with the price that outcomes bore out
# at the opening line, games 10+ (research/open_shrinkage.py, 2026-10-22: w
# fitted on the other priced season +0.43 / +0.36; at the close it is ~0).
# Display only: the Picks card's edge and EV read the lean's probability
# pulled toward the market by it (shrink_p); p_home, the lean and the
# hypotheses are never changed.
SHRINK_W = 0.40


def shrink_p(p, q, w=SHRINK_W):
    """sigma(logit q + w (logit p - logit q)): the model's probability with
    only the share w of its disagreement with the no-vig price q kept. Same
    side of q as p; w = 1 gives p, w = 0 gives q. NaN where p or q is."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lq = np.log(q / (1 - q))
        x = lq + w * (np.log(p / (1 - p)) - lq)
    return 1 / (1 + np.exp(-x))


def picks(g, price="close"):
    """One flat 1-unit bet per game per pick rule, graded at one price.

    price="open": the opening pair (graded rows; q devigged from it).
    price="first": the first pregame snapshot with a model P and a price
      (native rows, `first_*`), with the model P written at that snapshot:
      what an early bettor had. Its lean is that P's side.
    price="close": the closing pair (every basis; for reconstructed rows the
      value side is chosen against the close itself, i.e. with hindsight on
      the price). price="pre": the pregame snapshot pair -- the price that
      could actually be bet when the row was written (native rows only).

    Returns rows with, per rule and game: the picked side's model P, no-vig
    market P (q), break-even at the posted price, result, and unit P/L, plus
    `fav_units`: 1u on the market favourite of the same game at the same
    price (q > 0.5; exactly 0.5 ties to home), the same-row baseline, and
    `home_units`: 1u on the home side at the same price.
    """
    if g is None or not len(g):
        return pd.DataFrame()
    hml = pd.to_numeric(g[f"{price}_home_ml"], errors="coerce")
    aml = pd.to_numeric(g[f"{price}_away_ml"], errors="coerce")
    qh = price_q_home(g, price)
    ph = pd.to_numeric(g["first_p_home" if price == "first" else "p_home"],
                       errors="coerce")
    won = pd.to_numeric(g["home_won"], errors="coerce")
    ok = (qh.between(0, 1, inclusive="neither") & hml.notna() & aml.notna()
          & ph.notna() & won.isin([0, 1])).to_numpy()
    g, hml, aml, qh, ph, won = (x[ok] for x in (g, hml, aml, qh, ph, won))
    fav_home = (qh >= 0.5).to_numpy()
    fav_units = [market.unit_profit(m, w) for m, w in
                 zip(np.where(fav_home, hml, aml),
                     np.where(fav_home, won, 1 - won))]
    home_units = [market.unit_profit(m, w) for m, w in zip(hml, won)]
    out = []
    for rule, _label in PICK_RULES:
        if rule == "lean":
            home = (ph >= 0.5).to_numpy() if price == "first" else lean_is_home(g)
            keep = np.ones(len(ph), bool)
        else:
            home = (ph > qh).to_numpy()
            keep = (ph != qh).to_numpy()
        ml = np.where(home, hml, aml)
        d = pd.DataFrame(dict(
            rule=rule, game_id=g["game_id"].to_numpy(),
            early=(np.fmin(pd.to_numeric(g["gp_home"], errors="coerce"),
                           pd.to_numeric(g["gp_away"], errors="coerce"))
                   < EARLY_BELOW).to_numpy(),
            side=np.where(home, g["home"], g["away"]),
            model_p=np.where(home, ph, 1 - ph), q=np.where(home, qh, 1 - qh),
            ml=ml, won=np.where(home, won, 1 - won),
            model_tag=g["first_model_tag" if price == "first"
                        else "model_tag"].astype(str).to_numpy(),
            fav_units=fav_units, home_units=home_units))
        d = d[keep]
        d["breakeven"] = market.breakeven_prob(d["ml"])
        d["units"] = [market.unit_profit(m, w) for m, w in zip(d["ml"], d["won"])]
        d["units_null"] = d["q"] * [market.decimal_payout(m) for m in d["ml"]] - 1
        d["edge"] = d["model_p"] - d["q"]
        out.append(d)
    return pd.concat(out, ignore_index=True)


def roi_row(label, d):
    """Flat-stake summary. roi_null is the ROI if the no-vig market is right
    (about minus the hold, ~-4%); an ROI is judged against it, not zero.
    fav_roi is the market favourite on the same games at the same price."""
    n = len(d)
    if not n:
        return None
    u = d["units"].to_numpy(float)
    fav = (d["fav_units"].to_numpy(float) if "fav_units" in d
           else np.full(n, np.nan))
    return dict(
        label=label, n=n, w=int(d["won"].sum()), l=n - int(d["won"].sum()),
        model_p=float(d["model_p"].mean()), q=float(d["q"].mean()),
        breakeven=float(d["breakeven"].mean()), actual=float(d["won"].mean()),
        units=float(u.sum()), roi=float(u.mean()),
        roi_se=float(u.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        roi_null=float(d["units_null"].mean()),
        fav_units=float(fav.sum()), fav_roi=float(fav.mean()),
    )


def z_vs_null(r):
    """(ROI - ROI null) / SE; NaN when the SE is undefined."""
    se = r.get("roi_se", float("nan"))
    return ((r["roi"] - r["roi_null"]) / se if np.isfinite(se) and se > 0
            else float("nan"))


def roi_summary(g, price="close"):
    """[(rule label, [roi rows])] for SHOWN_RULES: all picks, and early
    (games 1-9) vs later when both exist. Descriptive, not filters."""
    p = picks(g, price)
    if not len(p):
        return []
    out = []
    for rule, label in PICK_RULES:
        if rule not in SHOWN_RULES:
            continue
        d = p[p["rule"] == rule]
        rows = [roi_row("All picks", d)]
        if d["early"].any() and (~d["early"]).any():
            rows += [roi_row("Games 10+", d[~d["early"]]),
                     roi_row("Games 1–9", d[d["early"]])]
        out.append((label, [r for r in rows if r]))
    return out


def roi_by_band(g, price="close", tags=None):
    """[(rule label, [roi rows])] by the picked side's price rung.

    Each row adds ev = actual - break-even, its null (q - break-even, about
    minus the hold, not zero) and z = (roi - roi_null) / roi_se. `tags`
    keeps only rows with those model tags (e.g. the v3 rows alone).
    Descriptive monitoring dimensions, not a filter: a band that looks good
    in-sample is a hypothesis to test forward.
    """
    p = picks(g, price)
    if not len(p):
        return []
    if tags is not None:
        p = p[p["model_tag"].isin(list(tags))]
        if not len(p):
            return []
    rung = p["ml"].map(market.ladder_rung)
    out = []
    for rule, label in PICK_RULES:
        if rule not in SHOWN_RULES:
            continue
        rows = []
        for _lo, _hi, band in market.ODDS_LADDER:
            d = p[(p["rule"] == rule) & (rung == band)]
            r = roi_row(band, d)
            if r:
                r["ev"] = r["actual"] - r["breakeven"]
                r["ev_null"] = r["q"] - r["breakeven"]
                r["z"] = z_vs_null(r)
                rows.append(r)
        if rows:
            out.append((label, rows))
    return out



def season_rows(g, price="close"):
    """One summary per closing book x season of graded rows `g` (one basis).

    Each: book, season, n, scoring (model vs the close, analysis.scoring)
    and a roi row per SHOWN_RULES rule at `price` (with the same-row favourite). The
    shared source of the Model page's season table and ledger_report.txt.
    """
    h = with_close(g)
    out = []
    for book, hb in book_split(h):
        for season in sorted(hb["season"].dropna().unique(), reverse=True):
            hs = hb[hb["season"] == season]
            p = picks(hs, price)
            rows = {rule: (roi_row(rule, p[p["rule"] == rule]) if len(p) else None)
                    for rule, _label in PICK_RULES if rule in SHOWN_RULES}
            out.append(dict(book=book, season=season, n=len(hs),
                            scoring=scoring(hs), **rows))
    return out

# Pre-registered forward hypotheses (CLAUDE.md; H1-H3 fixed 2026-09-30, H4
# 2026-10-01, amendment A1 (H2-first) 2026-10-01, all before any native rows). The thresholds are frozen: never
# tune them on native data. `hindsight` is the price the reconstructed scan
# found them at; `native` is the price natives are graded at: the latest
# pregame snapshot ("pre", near the close) or the first one ("first", the
# early price, with the model P written then).
HYPOTHESES = (
    dict(key="H1", rule="Games 1–9 · value side · model P − no-vig q ≥ 0.08",
         early=True, favourite=False, min_edge=0.08, hindsight="close",
         native="pre"),
    dict(key="H2", rule="Games 10+ · value side that is the favourite",
         early=False, favourite=True, min_edge=0.0, hindsight="open",
         native="pre"),
    dict(key="H3", rule="Games 10+ · value side · model P − no-vig q ≥ 0.12",
         early=False, favourite=False, min_edge=0.12, hindsight="open",
         native="pre"),
    dict(key="H4", rule="Games 10+ · value side · model P − no-vig q ≥ 0.12",
         early=False, favourite=False, min_edge=0.12, hindsight="open",
         native="first"),
    # Amendment A1 (v5 audit Task A): H2-H4 also scored at the decision-time
    # (first-snapshot) price. H3 there is H4; H2 there is this row. H2 and H3
    # themselves are unchanged.
    dict(key="H2·F", rule="Games 10+ · value side that is the favourite",
         early=False, favourite=True, min_edge=0.0, hindsight="open",
         native="first"),
    # H5 (owner's request, fixed 2026-10-20, before any native row had a model
    # P; research/ats_signal.py): against the spread. The model's home margin
    # sigma * Phi^-1(P), sigma frozen at 13.5; bet its side when |margin +
    # home line| >= 1 pt, main-line prices only (ATS_MAIN_LINE). Native at
    # the first snapshot (first_spread, its prices, first_p_home); hindsight
    # at the same book's open (OPEN_SPREADS).
    dict(key="H5", rule="Games 10+ · ATS: model margin vs the line ≥ 1 pt",
         market="ats", early=False, favourite=False, min_edge=1.0, sigma=13.5,
         hindsight="open", native="first"),
    # H6 (owner's request, fixed 2026-10-20 with H5, before any native row had
    # a model P): H5's rule unchanged, on games 1-9 (the carryover model, no
    # injury-report terms). Same sigma, threshold, price filter and prices.
    dict(key="H6", rule="Games 1–9 · ATS: model margin vs the line ≥ 1 pt",
         market="ats", early=True, favourite=False, min_edge=1.0, sigma=13.5,
         hindsight="open", native="first"),
)
PRICE_NAMES = {"pre": "pregame", "first": "first snapshot", "open": "open",
               "close": "close"}


def carry_view(g):
    """Rows with the carryover model's P in place of p_home / first_p_home
    wherever it was recorded (carry_p_home, first_carry_p_home; games 1-9 from
    model v7 on). H1 and H6 were registered on the carryover model and stay
    on it (owner's decision 2026-10-22); rows written before v7 have no carry
    column filled and their P already is the carryover model's."""
    if g is None or not len(g):
        return g
    g = g.copy()
    for col, carry in (("p_home", "carry_p_home"), ("first_p_home", "first_carry_p_home")):
        if carry in g.columns and col in g.columns:
            c = pd.to_numeric(g[carry], errors="coerce")
            g[col] = c.where(c.notna(), pd.to_numeric(g[col], errors="coerce"))
    return g


def hypothesis_picks(g, hyp, price):
    """The value-side bets one pre-registered rule makes, graded at `price`.
    The edge and "favourite" (q > 0.5) are read at that same price. An ATS
    rule (market "ats") bets the model's side against that price's spread,
    its edge in points (ats_side_picks). The games 1-9 rules (H1, H6) read
    the carryover model's P (carry_view)."""
    if hyp["early"]:
        g = carry_view(g)
    if hyp.get("market") == "ats":
        p = ats_side_picks(g, price, hyp["sigma"])
        if not len(p):
            return p
        return p[(p["early"] == hyp["early"]) & (p["edge"] >= hyp["min_edge"])]
    p = picks(g, price)
    if not len(p):
        return p
    d = p[(p["rule"] == "value") & (p["early"] == hyp["early"])
          & (p["edge"] >= hyp["min_edge"])]
    if hyp["favourite"]:
        d = d[d["q"] > 0.5]
    return d


# The book behind each graded price (the hypotheses read raw rows: sportsbooks).
PRICE_BOOK = {"first": "first_book", "pre": "pre_book", "open": "close_book",
              "close": "close_book"}
# The book the pages grade the hypotheses on (owner's decision, 2026-10-21):
# DraftKings only, as ESPN BET is no longer offered. The rules are unchanged;
# a basis keeps the bets whose graded price came from this book (PRICE_BOOK).
# The reconstructed DraftKings rows are 2025-26 from 2025-11-25 (ESPN lists
# DraftKings from then), all games 10+, so H1 and H6 have no hindsight there.
# The registered pooled figures are in CLAUDE.md and research/dk_hypotheses.py.
HYPOTHESIS_BOOK = "dk"


def on_book(g, price, book):
    """Rows of `g` whose `price` came from `book` (every row if book is None)."""
    if book is None or g is None or not len(g):
        return g
    return g[g[PRICE_BOOK[price]].astype(str) == book]


def hypothesis_rows(native, recon, book=None):
    """[(hypothesis, hindsight roi row | None, native roi row | None)].

    Hindsight: reconstructed rows at the price the scan used (pooled over
    seasons and books, as pre-registered, unless `book` keeps one book's
    prices). Native: pregame-locked rows at the hypothesis's native price.
    Never pooled with each other.
    """
    out = []
    for hyp in HYPOTHESES:
        ats = hyp.get("market") == "ats"
        hind = with_open_spreads(recon) if ats and hyp["hindsight"] == "open" else recon
        rows = []
        for g, price, label in ((hind, hyp["hindsight"], "hindsight"),
                                (native, hyp["native"], "native")):
            g = on_book(g, price, book)
            d = hypothesis_picks(g, hyp, price) if g is not None and len(g) else None
            r = ((ats_row if ats else roi_row)(label, d)
                 if d is not None and len(d) else None)
            if r:
                r["z"] = z_vs_null(r)
            rows.append(r)
        out.append((hyp, *rows))
    return out


def hypothesis_by_season(native, recon, keys=("H5", "H6"), book=None):
    """[(hypothesis, [part])] for the hypotheses in `keys`: the same bets as
    hypothesis_rows (with the same `book`), split by basis x the graded
    price's book x season and never pooled (native first, newest season
    first). part = dict(basis, book, season, price, row); a basis with no
    bets gives one part with book, season and row None."""
    order = {b: i for i, (_, b) in enumerate(market.BOOKS)}
    out = []
    for hyp in HYPOTHESES:
        if hyp["key"] not in keys:
            continue
        ats = hyp.get("market") == "ats"
        hind = with_open_spreads(recon) if ats and hyp["hindsight"] == "open" else recon
        parts = []
        for basis, g, price in (("native", native, hyp["native"]),
                                ("reconstructed", hind, hyp["hindsight"])):
            g = on_book(g, price, book)
            d = hypothesis_picks(g, hyp, price) if g is not None and len(g) else None
            if d is None or not len(d):
                parts.append(dict(basis=basis, book=None, season=None, price=price,
                                  row=None))
                continue
            meta = pd.DataFrame(dict(game_id=g["game_id"].astype(str).to_numpy(),
                                     book=g[PRICE_BOOK[price]].to_numpy(),
                                     season=pd.to_numeric(g["season"],
                                                          errors="coerce").to_numpy()))
            d = d.assign(game_id=d["game_id"].astype(str)).merge(meta, on="game_id",
                                                                 how="left")
            groups = sorted(d.groupby(["book", "season"], dropna=False),
                            key=lambda t: (-t[0][1], order.get(t[0][0], 99)))
            for (book, season), x in groups:
                r = (ats_row if ats else roi_row)(
                    "", x.drop(columns=["book", "season"]).reset_index(drop=True))
                r["z"] = z_vs_null(r)
                parts.append(dict(basis=basis, book=book, season=season,
                                  price=price, row=r))
        out.append((hyp, parts))
    return out


def hypothesis_groups():
    """HYPOTHESES as distinct bets in hindsight: H4 and H2·F are H3 and H2 at
    the first snapshot, which reconstructed rows lack, so each pair is one
    set of bets there. [(label, hypothesis)], e.g. ("H3 / H4", H3), in
    registration order."""
    out = {}
    for h in HYPOTHESES:
        k = (h.get("market"), h["early"], h["favourite"], h["min_edge"],
             h.get("sigma"))
        out.setdefault(k, []).append(h)
    return [(" / ".join(h["key"] for h in hs), hs[0]) for hs in out.values()]


def reprice_at_close(g, d, hyp):
    """Bets `d` (hypothesis_picks at the open, rows of `g`) re-priced at the
    close: same side, the close's price (and, for ATS, its line and the
    cover at that line). `move` is how far the market moved toward each
    bet from open to close: no-vig pp (moneyline) or points (ATS). Drops
    the open's favourite baseline (it priced the open)."""
    if not len(d):
        return d
    m = g.assign(game_id=g["game_id"].astype(str)).set_index("game_id")
    gid = d["game_id"].astype(str)
    home = (d["side"].to_numpy() == m.loc[gid, "home"].to_numpy())
    num = lambda c: pd.to_numeric(m.loc[gid, c], errors="coerce").to_numpy()  # noqa: E731
    out = d.drop(columns=["fav_units"], errors="ignore")
    if hyp.get("market") == "ats":
        line, hso, aso = (num(c) for c in ("close_spread", "close_home_spread_odds",
                                            "close_away_spread_odds"))
        res = np.array([market.ats_result(x, s) for x, s in
                        zip(num("home_pts") - num("away_pts"), line)])
        out["move"] = d["line"].to_numpy() - np.where(home, line, -line)
        out["ml"] = np.where(home, hso, aso)
        qh = np.array([market.devig(h, a) for h, a in zip(hso, aso)])
        out["q"] = np.where(home, qh, 1 - qh)
        out["result"] = np.where(home, res, 1 - res)
        out["units"] = [0.0 if r == 0.5 else market.unit_profit(x, r == 1)
                        for x, r in zip(out["ml"], out["result"])]
    else:
        qh = num("close_q_home")
        q_side = np.where(home, qh, 1 - qh)
        out["move"] = 100 * (q_side - d["q"].to_numpy())
        out["ml"] = np.where(home, num("close_home_ml"), num("close_away_ml"))
        out["q"] = q_side
        out["units"] = [market.unit_profit(x, w) for x, w in zip(out["ml"], out["won"])]
    out = out[np.isfinite(out["q"]) & out["ml"].notna()]
    out["breakeven"] = market.breakeven_prob(out["ml"])
    out["units_null"] = out["q"] * [market.decimal_payout(x) for x in out["ml"]] - 1
    return out


HALF_SEASON = "-02-01"   # the backtest's split: before / from 1 Feb of the end year


def hypothesis_backtest(recon, book):
    """The hypotheses' hindsight on one book's reconstructed rows (the
    Ledger's DraftKings backtest; research/dk_hypotheses.py). Per distinct
    rule (hypothesis_groups): dict(label, hyp, open, close, open_at_close,
    move, halves). open / close: roi rows of its bets at that price;
    open_at_close: the open bets re-priced at the close (reprice_at_close),
    move = dict(mean, se, toward, unit) of the market's move toward them;
    halves: [(name, roi row)] at the registered hindsight price, split at
    1 Feb. Rows are None without bets. Hindsight: these games were in the
    scans that chose the rules."""
    base = on_book(recon, "close", book)
    out = []
    for label, hyp in hypothesis_groups():
        ats = hyp.get("market") == "ats"
        g = with_open_spreads(base) if ats else base
        row_fn = ats_row if ats else roi_row

        def grade(x, price):
            d = hypothesis_picks(x, hyp, price) if x is not None and len(x) else None
            if d is None or not len(d):
                return None, None
            r = row_fn("", d)
            r["z"] = z_vs_null(r)
            return d, r
        d_open, r_open = grade(g, "open")
        r_moved = move = None
        if r_open:
            c = reprice_at_close(g, d_open, hyp)
            if len(c):
                r_moved = row_fn("", c)
                r_moved["z"] = z_vs_null(r_moved)
                v = c["move"].to_numpy(float)
                move = dict(mean=float(v.mean()),
                            se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1
                            else float("nan"),
                            toward=float((v > 0).mean()), unit="pts" if ats else "pp")
        halves = []
        if g is not None and len(g):
            end = pd.to_numeric(g["season"], errors="coerce").astype("Int64").astype(str)
            first = (g["slate_date"].astype(str) < end + HALF_SEASON).to_numpy()
            halves = [(name, grade(g[m], hyp["hindsight"])[1])
                      for name, m in (("to 31 Jan", first), ("from 1 Feb", ~first))]
        out.append(dict(label=label, hyp=hyp, open=r_open, close=grade(g, "close")[1],
                        open_at_close=r_moved, move=move, halves=halves))
    return out


def clv(g):
    """Closing-line value of native leans: close q - pregame q, lean side.

    Only rows whose pregame and closing prices come from the same book; a
    move between two books' lines is not a line move.
    """
    if g is None or not len(g):
        return None
    g = g[g["pre_book"].astype(str) == g["close_book"].astype(str)]
    if not len(g):
        return None
    pre = pd.to_numeric(g["pre_q_home"], errors="coerce")
    cls = pd.to_numeric(g["close_q_home"], errors="coerce")
    lean_home = (g["lean"].astype(str) == g["home"].astype(str))
    d = np.where(lean_home, cls - pre, pre - cls)
    d = d[np.isfinite(d)]
    if not d.size:
        return None
    return dict(n=int(d.size), mean=float(d.mean()),
                se=float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1
                else float("nan"),
                beat=float((d > 0).mean()))


# --------------------------------------------------------- against the spread
ATS_SIGMA = 13.5     # pts: SD of margin about the market's expectation. Maps a
                     # stored p_home to a margin for the model's cover %; it
                     # does not change any prediction (p_home is as written).
# The ATS value side (the side the model's margin favours against the line)
# is retired with the moneyline value side (SHOWN_RULES, 2026-10-06): -6.1%,
# -7.4% and -2.7% at the close against a -4.5% null, never above it.
ATS_RULES = (
    ("lean", "Lean ATS — the model's moneyline favourite against the spread"),
)


def ats_picks(g, sigma=ATS_SIGMA):
    """One flat 1-unit spread bet per game per rule at the closing spread.

    Rows need the closing spread line and both spread prices (one book, the
    same as the moneyline close). The model's implied home margin is
    sigma * Phi^-1(p_home); its cover probability for a side is
    Phi((margin + line) / sigma). q is the no-vig cover probability from the
    two spread prices. A push refunds the stake (0 units) and is left out of
    the cover rate.
    """
    if g is None or not len(g):
        return pd.DataFrame()
    line = pd.to_numeric(g["close_spread"], errors="coerce")
    hso = pd.to_numeric(g["close_home_spread_odds"], errors="coerce")
    aso = pd.to_numeric(g["close_away_spread_odds"], errors="coerce")
    ph = pd.to_numeric(g["p_home"], errors="coerce")
    margin = (pd.to_numeric(g["home_pts"], errors="coerce")
              - pd.to_numeric(g["away_pts"], errors="coerce"))
    ok = (line.notna() & hso.notna() & aso.notna() & margin.notna()
          & ph.between(0, 1, inclusive="neither")).to_numpy()
    if not ok.any():
        return pd.DataFrame()
    g, line, hso, aso, ph, margin = (x[ok] for x in (g, line, hso, aso, ph, margin))
    nd = NormalDist()
    mdl_margin = np.array([sigma * nd.inv_cdf(p) for p in ph])
    p_cover_home = np.array([nd.cdf(z) for z in (mdl_margin + line) / sigma])
    q_home = np.array([market.devig(h, a) for h, a in zip(hso, aso)])
    res_home = np.array([market.ats_result(m, s) for m, s in zip(margin, line)])
    out = []
    for rule, _label in ATS_RULES:
        home = lean_is_home(g)
        d = pd.DataFrame(dict(
            rule=rule, game_id=g["game_id"].to_numpy(),
            side=np.where(home, g["home"], g["away"]),
            line=np.where(home, line, -line),
            model_p=np.where(home, p_cover_home, 1 - p_cover_home),
            q=np.where(home, q_home, 1 - q_home),
            ml=np.where(home, hso, aso),
            result=np.where(home, res_home, 1 - res_home)))
        d = d[np.isfinite(d["q"])]
        d["breakeven"] = market.breakeven_prob(d["ml"])
        d["units"] = [0.0 if r == 0.5 else market.unit_profit(m, r == 1)
                      for m, r in zip(d["ml"], d["result"])]
        d["units_null"] = d["q"] * [market.decimal_payout(m) for m in d["ml"]] - 1
        out.append(d)
    return pd.concat(out, ignore_index=True)


def ats_row(label, d):
    """Flat-stake ATS summary. `actual` is covers / (covers + misses), judged
    against break-even at the posted spread price (~52.4% at -110); roi_null
    is the ROI if the no-vig spread is right (minus the hold, ~-4.5%)."""
    n = len(d)
    if not n:
        return None
    r = d["result"].to_numpy(float)
    w, l = int((r == 1).sum()), int((r == 0).sum())
    u = d["units"].to_numpy(float)
    return dict(
        label=label, n=n, w=w, l=l, push=n - w - l,
        model_p=float(d["model_p"].mean()), q=float(d["q"].mean()),
        breakeven=float(d["breakeven"].mean()),
        actual=w / (w + l) if w + l else float("nan"),
        units=float(u.sum()), roi=float(u.mean()),
        roi_se=float(u.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        roi_null=float(d["units_null"].mean()),
        **({"fav_roi": float(d["fav_units"].mean())} if "fav_units" in d else {}),
    )


ATS_PRICES = ("first", "pre", "open", "close")   # *_spread, *_home/away_spread_odds
# A main-line spread price sits in this band (99.96% of the reconstructed
# closes); ESPN's "open" is sometimes an alternate line (-20.5 at +320 for
# a game that closed -11.5 at -110), which is no main-line bet.
ATS_MAIN_LINE = (-135, 115)
# The same book's opening spreads for the reconstructed rows (research/
# ats_signal.py), read only to show H5's hindsight at the open.
OPEN_SPREADS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "diagnostics", "ats_signal", "open_spreads.csv")


def with_open_spreads(g, path=None):
    """`g` with open_spread and its prices from the research cache
    (OPEN_SPREADS), or None without the cache (rows that already carry them
    are returned as is)."""
    if g is None or not len(g) or "open_spread" in g.columns:
        return g
    path = path or OPEN_SPREADS
    if not os.path.exists(path):
        return None
    o = pd.read_csv(path, dtype={"game_id": str})
    cols = ["open_spread", "open_home_spread_odds", "open_away_spread_odds"]
    return g.assign(game_id=g["game_id"].astype(str)).merge(
        o[["game_id"] + cols], on="game_id", how="left")


def ats_sides(g, price, sigma=ATS_SIGMA):
    """The model's side against one price's spread, per game; no result
    needed (ats_side_picks grades it, ats_card shows it on the Today page).

    price "first": the first snapshot's spread, prices and first_p_home;
    "pre", "open", "close": that spread with p_home. The model's home margin
    is sigma * Phi^-1(P); edge = margin + home line (points): home if
    edge > 0, else away; `edge` is |edge|. Only main-line prices (both sides
    in ATS_MAIN_LINE) and a model P. Per game: early (games 1-9), side,
    home, edge, line (the side's), model_line (the model's line for that
    side, -its margin), ml, q (no-vig cover probability), model_p, and the
    home line and prices; indexed like `g`.
    """
    if g is None or not len(g) or f"{price}_spread" not in g.columns:
        return pd.DataFrame()
    num = lambda c: pd.to_numeric(g[c], errors="coerce")  # noqa: E731
    line, hso, aso = (num(f"{price}_{c}") for c in
                      ("spread", "home_spread_odds", "away_spread_odds"))
    ph = num("first_p_home" if price == "first" else "p_home")
    ok = (line.notna() & hso.between(*ATS_MAIN_LINE) & aso.between(*ATS_MAIN_LINE)
          & ph.between(0, 1, inclusive="neither")).to_numpy()
    if not ok.any():
        return pd.DataFrame()
    g, line, hso, aso, ph = (x[ok] for x in (g, line, hso, aso, ph))
    nd = NormalDist()
    m = np.array([sigma * nd.inv_cdf(p) for p in ph])          # model home margin
    edge = m + line.to_numpy()
    home = edge > 0
    q_home = np.array([market.devig(h, a) for h, a in zip(hso, aso)])
    return pd.DataFrame(dict(
        game_id=g["game_id"].to_numpy(),
        early=(np.fmin(pd.to_numeric(g["gp_home"], errors="coerce"),
                       pd.to_numeric(g["gp_away"], errors="coerce"))
               < EARLY_BELOW).to_numpy(),
        side=np.where(home, g["home"], g["away"]), home=home, edge=np.abs(edge),
        line=np.where(home, line, -line), model_line=np.where(home, -m, m),
        model_p=[nd.cdf(abs(e) / sigma) for e in edge],
        q=np.where(home, q_home, 1 - q_home), ml=np.where(home, hso, aso),
        home_line=line.to_numpy(), hso=hso.to_numpy(), aso=aso.to_numpy()),
        index=g.index)


def ats_side_picks(g, price, sigma=ATS_SIGMA):
    """Flat 1u spread bets on the model's side at one price (ats_sides),
    graded: result, units, its no-vig null and break-even, and fav_units,
    1u on the spread favourite of the same game at the same price (pick'em:
    the shorter price; tie: home). Finished games only."""
    if g is None or not len(g):
        return pd.DataFrame()
    margin = (pd.to_numeric(g["home_pts"], errors="coerce")
              - pd.to_numeric(g["away_pts"], errors="coerce"))
    done = margin.notna() & pd.to_numeric(g["home_won"], errors="coerce").isin([0, 1])
    s = ats_sides(g[done], price, sigma)
    if not len(s):
        return pd.DataFrame()
    margin, line = margin[s.index], s["home_line"]
    res_home = np.array([market.ats_result(m, x) for m, x in zip(margin, line)])
    fav_home = np.where(line < 0, True, np.where(line > 0, False, s["hso"] <= s["aso"]))

    def units(mls, res):
        return [0.0 if r == 0.5 else market.unit_profit(m, r == 1)
                for m, r in zip(mls, res)]
    d = s[["game_id", "early", "side", "line", "edge", "model_p", "q", "ml"]].assign(
        result=np.where(s["home"], res_home, 1 - res_home),
        fav_units=units(np.where(fav_home, s["hso"], s["aso"]),
                        np.where(fav_home, res_home, 1 - res_home)))
    d = d[(d["edge"] > 0) & np.isfinite(d["q"])].reset_index(drop=True)
    d["breakeven"] = market.breakeven_prob(d["ml"])
    d["units"] = units(d["ml"], d["result"])
    d["units_null"] = d["q"] * [market.decimal_payout(m) for m in d["ml"]] - 1
    return d


def ats_card(row):
    """The ATS hypothesis (H5 or H6) that applies to one ledger row, at its
    native price, exactly as the Ledger grades it: dict(hyp, bet, side,
    line, model_line, ml, edge, model_p, q), bet = edge >= the rule's
    threshold. None when the row has no first snapshot with a main-line
    spread, or no ATS rule covers its games played."""
    g = pd.DataFrame([row])
    for hyp in HYPOTHESES:
        if hyp.get("market") != "ats":
            continue
        s = ats_sides(carry_view(g) if hyp["early"] else g, hyp["native"], hyp["sigma"])
        if not len(s) or bool(s["early"].iloc[0]) != hyp["early"]:
            continue
        r = s.iloc[0]
        return dict(hyp=hyp, bet=bool(r["edge"] >= hyp["min_edge"]),
                    **{k: r[k] for k in ("side", "line", "model_line", "ml",
                                         "edge", "model_p", "q")})
    return None


def ats_summary(g, sigma=ATS_SIGMA):
    """[(rule label, [ats rows])] at the closing spread; [] without spreads."""
    p = ats_picks(g, sigma)
    if not len(p):
        return []
    out = []
    for rule, label in ATS_RULES:
        row = ats_row("All picks", p[p["rule"] == rule])
        if row:
            out.append((label, [row]))
    return out


# ATS bands (owner's request, 2026-10-21): the picked side's line (negative =
# laying points) and its spread price, inclusive (lo, hi, label). Descriptive
# monitoring dimensions, not filters (evidence rule 4); every main-line price
# (ATS_MAIN_LINE) and every half-point line falls in exactly one band. Plus
# prices are rare at the main line (under 2% of bets), so they share a band.
ATS_LINE_BANDS = ((-99, -10, "Fav −10 or more"), (-9.5, -6, "Fav −6 to −9.5"),
                  (-5.5, -3, "Fav −3 to −5.5"), (-2.5, 2.5, "Pick'em −2.5 to +2.5"),
                  (3, 5.5, "Dog +3 to +5.5"), (6, 9.5, "Dog +6 to +9.5"),
                  (10, 99, "Dog +10 or more"))
ATS_PRICE_BANDS = ((-1000, -116, "−116 or shorter"), (-115, -111, "−115 to −111"),
                   (-110, -110, "−110"), (-109, 1000, "−109 or longer"))
# Below this many bets a band's SE and z are not shown: a few bets with the
# same result give a near-zero spread and a meaningless z.
BAND_MIN_N = 10


def ats_by_band(d):
    """ATS bets `d` (ats_picks or ats_side_picks rows) by the picked side's
    line and by its spread price: {"line": [row], "price": [row]}, each an
    ats_row labelled with its band plus z vs the ROI null (SE and z left
    out below BAND_MIN_N bets); empty bands are left out. Descriptive,
    never a filter."""
    out = {"line": [], "price": []}
    if d is None or not len(d):
        return out
    for key, col, bands in (("line", "line", ATS_LINE_BANDS),
                            ("price", "ml", ATS_PRICE_BANDS)):
        v = pd.to_numeric(d[col], errors="coerce").to_numpy()
        for lo, hi, label in bands:
            r = ats_row(label, d[(v >= lo) & (v <= hi)])
            if r:
                if r["n"] < BAND_MIN_N:
                    r["roi_se"] = float("nan")
                r["z"] = z_vs_null(r)
                out[key].append(r)
    return out


def ats_lean_by_band(g, sigma=ATS_SIGMA):
    """The lean ATS bets at the closing spread (ats_picks) by band."""
    p = ats_picks(g, sigma)
    return ats_by_band(p[p["rule"] == "lean"] if len(p) else p)


def hypothesis_ats_bands(native, recon, book=None):
    """[(hypothesis, {"native": bands, "hindsight": bands})] for the ATS
    hypotheses: the same bets as hypothesis_rows (with `book`) by the
    picked side's line and spread price (ats_by_band). Descriptive: the
    rules themselves stay as registered."""
    out = []
    for hyp in HYPOTHESES:
        if hyp.get("market") != "ats":
            continue
        hind = with_open_spreads(recon) if hyp["hindsight"] == "open" else recon
        parts = {}
        for basis, g, price in (("native", native, hyp["native"]),
                                ("hindsight", hind, hyp["hindsight"])):
            g = on_book(g, price, book)
            d = hypothesis_picks(g, hyp, price) if g is not None and len(g) else None
            parts[basis] = ats_by_band(d)
        out.append((hyp, parts))
    return out


# ------------------------------------------------------ market performance --
# The market alone (owner's request, 2026-10-22; build_site's Performance
# page): how the closing prices themselves did, read from the graded rows'
# prices and results only, never a model column. Favourite = the side the
# no-vig close makes more likely (a 50/50 close has none); the spread
# favourite is the side giving points (a pick'em line falls back to the
# moneyline favourite).
FAV_LINE_BANDS = ATS_LINE_BANDS[:4]   # the favourite's line: −10+ ... pick'em


def _se_mean(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def market_games(g):
    """One row per graded game with a usable close: the moneyline favourite
    (fav_home, fav_ml, fav_q, fav_won), flat 1-unit P/L at the close for the
    favourite and the underdog with their nulls (q x decimal - 1), and,
    where the closing spread and both its prices are recorded from the same
    book as the moneyline, the spread favourite's line (fav_line <= 0), its
    cover (1 / 0 / 0.5 push), units and null at its spread price, the home
    side's cover, and the line's error (home margin + home line: > 0 means
    home beat the line). Empty frame without rows."""
    if g is None or not len(g):
        return pd.DataFrame()
    num = lambda c: pd.to_numeric(g[c], errors="coerce").to_numpy(float)  # noqa: E731
    q, y = num("close_q_home"), num("home_won")
    hml, aml = num("close_home_ml"), num("close_away_ml")
    margin = num("home_pts") - num("away_pts")
    ok = np.isfinite(q) & np.isfinite(y) & np.isfinite(hml) & np.isfinite(aml) & (q != 0.5)
    fav_home = q > 0.5
    fav_ml = np.where(fav_home, hml, aml)
    dog_ml = np.where(fav_home, aml, hml)
    fav_q = np.where(fav_home, q, 1 - q)
    fav_won = np.where(fav_home, y, 1 - y)
    dec = np.array([market.decimal_payout(m) for m in fav_ml])
    ddec = np.array([market.decimal_payout(m) for m in dog_ml])
    out = pd.DataFrame(dict(
        game_id=g["game_id"].astype(str).to_numpy(),
        slate_date=g["slate_date"].astype(str).to_numpy(),
        fav_home=fav_home, fav_ml=fav_ml, fav_q=fav_q, fav_won=fav_won,
        fav_units=[market.unit_profit(m, w == 1) for m, w in zip(fav_ml, fav_won)],
        fav_null=fav_q * dec - 1,
        dog_units=[market.unit_profit(m, w == 0) for m, w in zip(dog_ml, fav_won)],
        dog_null=(1 - fav_q) * ddec - 1,
        home_won=y))
    hs = num("close_spread") if "close_spread" in g else np.full(len(g), np.nan)
    hso = num("close_home_spread_odds") if "close_home_spread_odds" in g else hs * np.nan
    aso = num("close_away_spread_odds") if "close_away_spread_odds" in g else hs * np.nan
    same = np.ones(len(g), bool)
    if "spread_book" in g and "close_book" in g:
        same = (g["spread_book"].astype(str) == g["close_book"].astype(str)).to_numpy()
    has = np.isfinite(hs) & np.isfinite(hso) & np.isfinite(aso) & np.isfinite(margin) & same
    home_cover = np.array([market.ats_result(mg, s) if h else np.nan
                           for mg, s, h in zip(margin, hs, has)])
    sfav_home = np.where(hs != 0, hs < 0, fav_home)
    sml = np.where(sfav_home, hso, aso)
    sq = np.array([market.devig(h_, a_) if h else np.nan
                   for h_, a_, h in zip(hso, aso, has)])
    sq_fav = np.where(sfav_home, sq, 1 - sq)
    cover = np.where(sfav_home, home_cover, 1 - home_cover)
    out["has_spread"] = has
    out["fav_line"] = np.where(has, -np.abs(hs), np.nan)
    out["fav_cover"] = np.where(has, cover, np.nan)
    out["home_cover"] = home_cover
    out["fav_margin_vs_line"] = np.where(
        has, np.where(sfav_home, margin + hs, -margin - hs), np.nan)
    out["ats_units"] = [0.0 if c == 0.5 else market.unit_profit(m, c == 1) if h else np.nan
                        for c, m, h in zip(cover, sml, has)]
    out["ats_null"] = np.where(has, sq_fav * np.array(
        [market.decimal_payout(m) if h else np.nan for m, h in zip(sml, has)]) - 1, np.nan)
    out["line_error"] = np.where(has, margin + hs, np.nan)
    return out[ok].reset_index(drop=True)


def _su_row(m):
    n = len(m)
    p, w = m["fav_q"].to_numpy(float), m["fav_won"].to_numpy(float)
    return dict(n=n, w=int(w.sum()), l=n - int(w.sum()), actual=float(w.mean()),
                implied=float(p.mean()), diff=float(w.mean() - p.mean()),
                se=float(np.sqrt((p * (1 - p)).sum()) / n),
                fav_roi=float(m["fav_units"].mean()), fav_se=_se_mean(m["fav_units"]),
                fav_null=float(m["fav_null"].mean()),
                dog_roi=float(m["dog_units"].mean()), dog_se=_se_mean(m["dog_units"]),
                dog_null=float(m["dog_null"].mean()))


def _ats_row(m, col="fav_cover"):
    s = m[m["has_spread"]]
    c = s[col].to_numpy(float)
    w, l = int((c == 1).sum()), int((c == 0).sum())
    out = dict(n=len(s), w=w, l=l, push=len(s) - w - l,
               rate=w / (w + l) if w + l else float("nan"),
               se=float(np.sqrt(0.25 / (w + l))) if w + l else float("nan"))
    if col == "fav_cover" and len(s):
        out.update(roi=float(s["ats_units"].mean()), roi_se=_se_mean(s["ats_units"]),
                   null=float(s["ats_null"].mean()),
                   margin=float(s["fav_margin_vs_line"].mean()),
                   margin_se=_se_mean(s["fav_margin_vs_line"]))
    return out


def market_summary(m):
    """The market's season in one dict (market_games rows): su (favourites
    straight up and at the moneyline, underdogs at the moneyline), ats
    (spread favourites), home_ats (home sides) and line (closing line vs the
    final margin: mae, rmse, bias = mean home margin + home line, its se).
    None without rows."""
    if m is None or not len(m):
        return None
    s = m[m["has_spread"]]
    e = s["line_error"].to_numpy(float)
    line = (dict(n=len(e), mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                 bias=float(e.mean()), bias_se=_se_mean(e)) if len(e) else None)
    return dict(su=_su_row(m), ats=_ats_row(m) if len(s) else None,
                home_ats=_ats_row(m, "home_cover") if len(s) else None, line=line)


def market_ml_bands(m):
    """[(rung, su row)] by the favourite's closing price (market.ODDS_LADDER),
    empty rungs left out. Descriptive, never a filter."""
    if m is None or not len(m):
        return []
    rung = np.array([market.ladder_rung(x) for x in m["fav_ml"]], dtype=object)
    return [(label, _su_row(m[rung == label])) for _lo, _hi, label in market.ODDS_LADDER
            if (rung == label).any()]


def market_ats_bands(m):
    """[(band, ats row)] by the spread favourite's closing line
    (FAV_LINE_BANDS), empty bands left out."""
    if m is None or not len(m):
        return []
    s = m[m["has_spread"]]
    out = []
    for lo, hi, label in FAV_LINE_BANDS:
        x = s[(s["fav_line"] >= lo) & (s["fav_line"] <= hi)]
        if len(x):
            out.append((label, _ats_row(x)))
    return out


def market_by_side(m):
    """[(label, su row, ats row)] for home and road moneyline favourites."""
    if m is None or not len(m):
        return []
    out = []
    for label, k in (("Home favourites", m["fav_home"]), ("Road favourites", ~m["fav_home"])):
        x = m[k.to_numpy(bool)]
        if len(x):
            out.append((label, _su_row(x),
                        _ats_row(x) if x["has_spread"].any() else None))
    return out
