"""Calibration and market comparisons, shared by the site and the report.

Every function takes graded ledger rows (ledger.graded) and uses ONLY rows
with a valid DraftKings closing pair, so the model and the market are always
compared on identical games. Callers pass one basis at a time (native or
reconstructed); nothing here pools the two.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import market

PROB_BINS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)


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
    lean_home = (h["lean"].astype(str) == h["home"].astype(str)).to_numpy()
    h["lean_home"] = lean_home
    h["lean_q"] = np.where(lean_home, h["close_q_home"], 1 - h["close_q_home"])
    h["lean_ml"] = np.where(lean_home, h["close_home_ml"], h["close_away_ml"])
    h["lean_p"] = np.where(lean_home, h["p_home"], 1 - h["p_home"])
    h["lean_won"] = np.where(lean_home, h["home_won"], 1 - h["home_won"])
    return h


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
    bm, bq = market.brier(p, y), market.brier(q, y)
    lm, lq = market.logloss(p, y), market.logloss(q, y)

    def se(d):
        return float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return dict(
        n=n,
        model=dict(brier=float(bm.mean()), logloss=float(lm.mean()),
                   acc=float(((p >= 0.5) == (y == 1)).mean())),
        market=dict(brier=float(bq.mean()), logloss=float(lq.mean()),
                    acc=float(((q >= 0.5) == (y == 1)).mean())),
        d_brier=float((bm - bq).mean()), d_brier_se=se(bm - bq),
        d_logloss=float((lm - lq).mean()), d_logloss_se=se(lm - lq),
    )


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


def clv(g):
    """Closing-line value of native leans: close q - pregame q, lean side."""
    if g is None or not len(g):
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
