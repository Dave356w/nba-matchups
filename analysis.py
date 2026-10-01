"""Calibration and market comparisons, shared by the site and the report.

Every function takes graded ledger rows (ledger.graded) and uses ONLY rows
with a valid closing pair, so the model and the market are always compared on
identical games. Callers pass one basis (native or reconstructed) and one
closing book (`book_split`) at a time; nothing here pools either.
"""
from __future__ import annotations

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


def book_split(h):
    """[(book label, rows)] by closing book, in market.BOOKS order."""
    if h is None or not len(h):
        return []
    return [(b, h[h["close_book"] == b]) for _, b in market.BOOKS
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
EARLY_BELOW = 10     # nba_composite.MIN_GAMES: below it v2 uses the carryover model
EDGE_BINS = ((0.0, 0.02, "0–2 pp"), (0.02, 0.05, "2–5 pp"),
             (0.05, 0.10, "5–10 pp"), (0.10, 1.0, "10+ pp"))


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
    market P (q), break-even at the posted price, result, and unit P/L.
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
                        else "model_tag"].astype(str).to_numpy()))
        d = d[keep]
        d["breakeven"] = market.breakeven_prob(d["ml"])
        d["units"] = [market.unit_profit(m, w) for m, w in zip(d["ml"], d["won"])]
        d["units_null"] = d["q"] * [market.decimal_payout(m) for m in d["ml"]] - 1
        d["edge"] = d["model_p"] - d["q"]
        out.append(d)
    return pd.concat(out, ignore_index=True)


def roi_row(label, d):
    """Flat-stake summary. roi_null is the ROI if the no-vig market is right
    (about minus the hold, ~-4%); an ROI is judged against it, not zero."""
    n = len(d)
    if not n:
        return None
    u = d["units"].to_numpy(float)
    return dict(
        label=label, n=n, w=int(d["won"].sum()), l=n - int(d["won"].sum()),
        model_p=float(d["model_p"].mean()), q=float(d["q"].mean()),
        breakeven=float(d["breakeven"].mean()), actual=float(d["won"].mean()),
        units=float(u.sum()), roi=float(u.mean()),
        roi_se=float(u.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        roi_null=float(d["units_null"].mean()),
    )


def z_vs_null(r):
    """(ROI - ROI null) / SE; NaN when the SE is undefined."""
    se = r.get("roi_se", float("nan"))
    return ((r["roi"] - r["roi_null"]) / se if np.isfinite(se) and se > 0
            else float("nan"))


def roi_summary(g, price="close"):
    """[(rule label, [roi rows])]: all picks, early (games 1-9) vs later when
    both exist, and value picks by model edge. Descriptive, not filters."""
    p = picks(g, price)
    if not len(p):
        return []
    out = []
    for rule, label in PICK_RULES:
        d = p[p["rule"] == rule]
        rows = [roi_row("All picks", d)]
        if d["early"].any() and (~d["early"]).any():
            rows += [roi_row("Games 10+", d[~d["early"]]),
                     roi_row("Games 1–9 (carryover)", d[d["early"]])]
        if rule == "value":
            for lo, hi, lab in EDGE_BINS:
                s = d[(d["edge"] >= lo) & (d["edge"] < hi)]
                if len(s):
                    rows.append(roi_row(f"Edge {lab}", s))
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
)
PRICE_NAMES = {"pre": "pregame", "first": "first snapshot", "open": "open",
               "close": "close"}


def hypothesis_picks(g, hyp, price):
    """The value-side bets one pre-registered rule makes, graded at `price`.
    The edge and "favourite" (q > 0.5) are read at that same price."""
    p = picks(g, price)
    if not len(p):
        return p
    d = p[(p["rule"] == "value") & (p["early"] == hyp["early"])
          & (p["edge"] >= hyp["min_edge"])]
    if hyp["favourite"]:
        d = d[d["q"] > 0.5]
    return d


def hypothesis_rows(native, recon):
    """[(hypothesis, hindsight roi row | None, native roi row | None)].

    Hindsight: reconstructed rows at the price the scan used (pooled over
    seasons and books, as pre-registered). Native: pregame-locked rows at
    the hypothesis's native price. Never pooled with each other.
    """
    out = []
    for hyp in HYPOTHESES:
        rows = []
        for g, price, label in ((recon, hyp["hindsight"], "hindsight"),
                                (native, hyp["native"], "native")):
            r = roi_row(label, hypothesis_picks(g, hyp, price)) \
                if g is not None and len(g) else None
            if r:
                r["z"] = z_vs_null(r)
            rows.append(r)
        out.append((hyp, *rows))
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
                     # stored p_home to a margin for the ATS value side only;
                     # it does not change any prediction (p_home is as written).
ATS_RULES = (
    ("lean", "Lean ATS — the model's moneyline favourite against the spread"),
    ("value", "Value ATS — side the model's margin favours against the line"),
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
        home = lean_is_home(g) if rule == "lean" else p_cover_home > 0.5
        keep = np.ones(len(ph), bool) if rule == "lean" else p_cover_home != 0.5
        d = pd.DataFrame(dict(
            rule=rule, game_id=g["game_id"].to_numpy(),
            side=np.where(home, g["home"], g["away"]),
            line=np.where(home, line, -line),
            model_p=np.where(home, p_cover_home, 1 - p_cover_home),
            q=np.where(home, q_home, 1 - q_home),
            ml=np.where(home, hso, aso),
            result=np.where(home, res_home, 1 - res_home)))[keep]
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
    )


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
