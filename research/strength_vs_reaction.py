#!/usr/bin/env python3
"""In-game Kalshi: strength prior vs market reaction (HINDSIGHT model rows).

The owner's notebook nba_strength_vs_reaction.ipynb (2026-10-22), ported to
the repo's data. A live game's fair home price, from a Brownian margin whose
drift is fixed by a pregame probability p0 (z0 = Phi^-1(p0)):

  p* = Phi( delta / (sigma sqrt(tau)) + z0 sqrt(tau) )

delta = home - away now, tau = share of regulation left, sigma = the
full-game margin SD. Three priors for z0, the notebook's arms:

  MKT   : Kalshi's pregame mid (the home market's last 1-minute candle with
          a bid and an ask at or before the first play)
  MODEL : the model's P (reconstructed v6 rows; the notebook named v5)
  BLEND : w MODEL + (1 - w) MKT in z, w fitted on the training games

Two edges are kept apart. REACTION: with the MKT prior, p* differs from the
live mid only in how hard the market reacts to the score. STRENGTH: MODEL or
BLEND beats MKT only if the model knows something the pregame mid does not.
Each is scored on held-out games (chronological split at TRAIN_END; sigma
and w are fitted on the earlier games only): pregame log loss; in-game log
loss of p* against the live mid and the outcome's weight on their
disagreement; and trades, one per game per prior, the first quiet minute
where p* clears the ask + taker fee by ENTRY_EDGE, filled at the worst
quote of a LATER minute, beside the null (the same contract valued at the
fill minute's mid) and the live favourite bought at the same minute.

Reaction, two further tests (owner's request, 2026-10-22). The prior's
weight set free: p* = Phi(delta / (sigma sqrt(tau)) + k z0 sqrt(tau)), a
probit on delta / sqrt(tau) and z0 sqrt(tau), fitted to the outcome and to
the live mid on the same minutes (sigma and k for each, and their gap with a
joint clustered ±). And with no p* form: the outcome's weight on the score
(a) and the prior (b) beyond the live mid. Both are scored out of sample like
the arms above: p* with the train outcome fit's (sigma, k), and the mid
corrected by the train fit's (a, b) or by a alone.

Live fluctuations (owner's question, 2026-10-22): each minute's mid move
against the spread and the move the score and clock justify, by time
remaining; how far the mid follows a score move in that minute, the next
and minutes 2-6 after (the last is 0 for a fair price, whatever p*'s scale);
the bounce when the margin does not move; and the drift after big score
moves against the cost to cross.

Both sides at tip (owner's question, 2026-10-22): buy home and away at the
last pregame quotes, rest a sell on each at entry + 5 / 10 / 15 / 20c, let an
unsold side settle; profit per $1 pair beside holding both, and how far the
mid had already passed the target when a sell filled. Then, at each +10c
fill, whether the side wins more often than its price says and whether the
overshoot, the minute's jump, the price - p* residual, the lead, time left or
a second fill adds anything; and how far eventual winners and losers run
from their entry price. Last, the quarter-by-quarter price path: whether
the moves so far and the choppiness predict the winner beyond the price at
the end of Q1, Q2 and Q3, and the Q3-end favourite's up/down patterns
against its price. And the favourite-longshot bias late in games on raw
quotes (the extremes kept): the favourite's win rate against its price by
band, the same as loss counts per game with exact Poisson intervals, and a
buy-the-favourite trade by ask band. Three late-game tests were then
pre-registered for the 2025 and 2026 play-in and playoff games
(diagnostics/strength_vs_reaction/PREREG_holdout.md) before any of that
data was read.

Data (2025-26, the first season Kalshi lists): every reconstructed game with
a Kalshi event (diagnostics/kalshi_benchmark/games.csv), ESPN's play-by-play
with wallclock times (summary endpoint) and the home market's 1-minute
candles from 3 h before the first play to 10 min after the last (Kalshi's
historical archive, else the live API). Cached compactly under
research/output/strength_vs_reaction/.

  python research/strength_vs_reaction.py [--limit N] [--fetch-only]

The model rows are reconstructed (hindsight; for 2025-26 every fit uses
earlier seasons only); no native row has a model P yet. Research only:
nothing here touches the ledgers, the pages, the model or the hypotheses.
Writes diagnostics/strength_vs_reaction/{RESULTS.txt,games.csv,trades.csv}.
± is 95%, game-clustered.

Changes from the notebook (fixes, or a stricter reading):
  - Kalshi's archive quotes dollars as strings ("0.5500") in close / high /
    low; the notebook read them as cents, which drops every quote.
  - A fill is the candle ending exactly (1 + LAG) minutes after the signal,
    matched by time, not by row, so a missing candle never stretches the lag.
  - Plays keep ESPN's order, with stray wallclock stamps interpolated and
    time made non-decreasing, instead of being re-sorted by wallclock (one
    play stamped 5 h late froze a game's state).
  - The two sigma diagnostics are replaced by one like-for-like test: sigma
    fitted to the outcome and to the live mid on the same minutes with the
    same p* form, overall and by part of the game, with a game bootstrap of
    the difference. The notebook's market-implied regression dropped mids
    outside [.05, .95], which truncates the response and biases sigma up,
    and its outcome fit sat on the (8, 18) bound. The score-process sigma
    is still the one every p* uses.
  - Every trade figure carries its null and the live-favourite baseline;
    every log loss difference its ± 95%.
  - The fee is market.kalshi_cost's (the large-order limit; the notebook
    rounds each 100-contract order up to the cent, under 0.01c a contract).
  - scipy is replaced by the standard library (Phi, Phi^-1) and a
    golden-section search; the repo does not depend on scipy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import kalshi                                     # noqa: E402
import kalshi_benchmark as kb                     # noqa: E402
import ledger                                     # noqa: E402
import market                                     # noqa: E402
import model_review as mr                         # noqa: E402

CACHE = ROOT / "research" / "output" / "strength_vs_reaction"
OUT = ROOT / "diagnostics" / "strength_vs_reaction"
BENCH = ROOT / "diagnostics" / "kalshi_benchmark" / "games.csv"
Z95 = 1.96
EPS = 1e-6

# The notebook's settings, unchanged.
TRAIN_END = "2026-01-31"          # chronological split; sigma and w fitted here only
END_DATE = "2026-04-12"           # regular season end
SIGMA_GRID = np.round(np.arange(6.0, 40.0001, 0.05), 2)  # sigma fits (points)
N_BOOT = 1000                     # game bootstrap for the sigma fits
STRAY_SEC = 300                   # a wallclock this far from its neighbours' median is stray
TAU_MIN, TAU_MAX = 0.10, 0.90     # trade window (share of regulation left)
QUIET_SEC = 45                    # skip if the score changed in the last N seconds
LAG_CANDLES = 1                   # decide on candle t, fill on candle t + 1 + LAG
ENTRY_EDGE = 0.04                 # min (p* - ask - fee) on the signal candle
MIN_PRICE, MAX_PRICE = 0.03, 0.97 # ignore dead quotes
PRE_WINDOW = 3 * 3600             # candles read before the first play
POST_WINDOW = 600                 # and after the last
EDGE_BINS = [ENTRY_EDGE, 0.06, 0.08, 0.12, 1.0]
STRADDLE_TARGETS = (0.05, 0.10, 0.15, 0.20)   # sell each side this far above entry
MAKER_FEE = 0.0175                # Kalshi's maker fee coefficient, where it applies
QUARTER_ENDS = (0.75, 0.5, 0.25)  # tau at the end of Q1, Q2, Q3
LONGSHOT_BANDS = ((0.50, 0.70), (0.70, 0.85), (0.85, 0.93), (0.93, 0.97), (0.97, 0.995))
TAU_BUCKETS = ((0.5, 1.0, "1st half"), (0.25, 0.5, "3rd quarter"),
               (0.0, 0.25, "4th quarter"))
PRIORS = (("z_mkt", "MKT"), ("z_model", "MODEL"), ("z_blend", "BLEND"))

_erfc = np.frompyfunc(math.erfc, 1, 1)
_N = NormalDist()


def Phi(x):
    return (0.5 * _erfc(-np.asarray(x, float) / math.sqrt(2))).astype(float)


def Phi_inv(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.array([_N.inv_cdf(v) for v in p.ravel()]).reshape(p.shape)


def clip(p):
    return np.clip(np.asarray(p, float), EPS, 1 - EPS)


def ll(p, y):
    p, y = clip(p), np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def p_star(delta, tau, z0, sigma):
    """In-game home win probability: Brownian margin, drift fixed by z0."""
    tau = np.maximum(np.asarray(tau, float), 1e-4)
    return Phi(np.asarray(delta, float) / (sigma * np.sqrt(tau)) + np.asarray(z0) * np.sqrt(tau))


def fee(p):
    """Kalshi's taker fee per contract bought at p (market.kalshi_cost - p)."""
    p = np.asarray(p, float)
    return np.vectorize(market.kalshi_cost, otypes=[float])(p) - p


def golden(f, lo, hi, tol=1e-6):
    """Minimiser of a unimodal f on [lo, hi]."""
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


# ------------------------------------------------------------- fetching ----
def _clock_sec(s):
    s = str(s)
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    try:
        return float(s)
    except ValueError:
        return float("nan")


def espn_game(gid):
    """{'final': [home, away], 'plays': [[ts, period, clock s, home, away]]}
    in ESPN's order, plays without a wallclock dropped (cached)."""
    path = CACHE / "espn" / f"{gid}.json"
    if path.exists():
        return json.loads(path.read_text())
    js = market.get_json(market.SUMMARY.format(eid=gid))
    plays = []
    for p in js.get("plays") or []:
        wc = p.get("wallclock")
        if not wc or p.get("homeScore") is None or p.get("awayScore") is None:
            continue
        clk = _clock_sec((p.get("clock") or {}).get("displayValue"))
        per = (p.get("period") or {}).get("number")
        if per is None or clk != clk:
            continue
        plays.append([pd.Timestamp(wc).timestamp(), int(per), clk,
                      int(p["homeScore"]), int(p["awayScore"])])
    comp = js["header"]["competitions"][0]["competitors"]
    fin = {c["homeAway"]: int(c.get("score") or 0) for c in comp}
    res = {"final": [fin.get("home"), fin.get("away")], "plays": plays}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    return res


def px(q, field):
    """A candle quote field (close / high / low) in dollars. Live payloads
    carry f'{field}_dollars', older live payloads integer cents, the
    historical archive a dollar string ("0.5500")."""
    q = q or {}
    v = q.get(f"{field}_dollars")
    if v is None and q.get(field) is not None:
        raw = q[field]
        v = float(raw) if isinstance(raw, str) and "." in raw else float(raw) / 100.0
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if 0 < v < 1 else None


def kalshi_game(gid, ticker, start, end):
    """{'ticker', 'candles': [[end ts, ask close, ask high, bid close, bid
    low, volume]]} for the home market (cached)."""
    path = CACHE / "kalshi" / f"{gid}.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = []
    for c in kb.candles(ticker, start, end, 1):
        ya, yb = c.get("yes_ask"), c.get("yes_bid")
        try:
            vol = float(c.get("volume") or 0)
        except (TypeError, ValueError):
            vol = 0.0
        rows.append([c.get("end_period_ts"), px(ya, "close"), px(ya, "high"),
                     px(yb, "close"), px(yb, "low"), vol])
    res = {"ticker": ticker, "candles": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    return res


def load_games(limit=None):
    """2025-26 reconstructed regular-season games with a result and a Kalshi
    event (the benchmark's map), with the home market's ticker."""
    r = ledger.load(ledger.RECON_PATH)
    r = r[(r["season"].astype(float) == 2026) & r["home_won"].isin([0, 1])
          & (r["slate_date"] <= END_DATE) & r["p_home"].notna()].copy()
    k = pd.read_csv(BENCH, dtype={"game_id": str})[["game_id", "kalshi_event"]]
    r = r.merge(k, on="game_id", how="left")
    r = r[r["kalshi_event"].notna()].sort_values(["slate_date", "tip_utc"])
    r["ticker"] = [f"{e}-{kalshi.BBR2KALSHI.get(h, h)}"
                   for e, h in zip(r["kalshi_event"], r["home"])]
    if limit:                       # evenly spaced, so both splits are present
        r = r.iloc[np.unique(np.linspace(0, len(r) - 1, limit).astype(int))]
    return r


def fetch(games, workers=8):
    """ESPN play-by-play for every game, then each home market's candles
    over [first play - 3 h, last play + 10 min]."""
    ids = list(games["game_id"])

    def one_espn(gid):
        try:
            return gid, espn_game(gid)
        except Exception as e:  # noqa: BLE001 - one game; keep going
            print(f"espn {gid}: {e!r}", flush=True)
            return gid, None

    espn = {}
    with ThreadPoolExecutor(workers) as ex:
        for i, (gid, js) in enumerate(ex.map(one_espn, ids)):
            espn[gid] = js
            if i % 200 == 0:
                print(f"espn {i}/{len(ids)}", flush=True)

    def one_kalshi(row):
        js = espn.get(row.game_id)
        if not js or not js["plays"]:
            return row.game_id, None
        ts = [p[0] for p in js["plays"]]
        try:
            return row.game_id, kalshi_game(row.game_id, row.ticker,
                                            min(ts) - PRE_WINDOW, max(ts) + POST_WINDOW)
        except Exception as e:  # noqa: BLE001 - one game; keep going
            print(f"kalshi {row.game_id}: {e!r}", flush=True)
            return row.game_id, None

    kal = {}
    with ThreadPoolExecutor(min(workers, 4)) as ex:
        for i, (gid, js) in enumerate(ex.map(one_kalshi, games.itertuples())):
            kal[gid] = js
            if i % 200 == 0:
                print(f"kalshi {i}/{len(ids)}", flush=True)
    return espn, kal


# --------------------------------------------------------------- states ----
def clean_wallclock(t):
    """ESPN's play times, in ESPN's order. A few plays carry a stray stamp
    (one 2025-26 play is 5 h late); any stamp more than STRAY_SEC from the
    median of its 10 neighbours is interpolated, then time is made
    non-decreasing (stamps a few seconds out of order are common). A play
    beside a real gap (halftime, a review) has most of its window on its own
    side, so gaps are kept. (times, stray stamps replaced)."""
    s = pd.Series(np.asarray(t, float))
    bad = (s - s.rolling(11, center=True, min_periods=3).median()).abs() > STRAY_SEC
    s[bad] = np.nan
    s = s.interpolate(limit_direction="both")
    return np.maximum.accumulate(s.to_numpy()), int(bad.sum())


def game_states(row, espn, kal):
    """(pregame record, in-game states) for one game, or (reason, None).
    State = the last play at or before each live candle's end; the fill is
    the candle ending (1 + LAG) minutes later. Nothing after a decision is
    used to make it."""
    if not espn or len(espn["plays"]) < 50:
        return "no play-by-play", None
    if not kal or not kal["candles"]:
        return "no candles", None
    pl = np.array(espn["plays"], float)
    ts, stray = clean_wallclock(pl[:, 0])
    tip = ts[0]
    sched = ledger.parse_utc(row.tip_utc).timestamp()
    if not (-1800 <= tip - sched <= 7200 and 1.5 * 3600 <= ts[-1] - tip <= 4.5 * 3600):
        return "implausible wallclock", None
    c = pd.DataFrame(kal["candles"], columns=["ts", "ask_c", "ask_h", "bid_c", "bid_l", "vol"])
    c = c.dropna(subset=["ts"]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    for col in ("ask_c", "ask_h", "bid_c", "bid_l"):
        c[col] = pd.to_numeric(c[col], errors="coerce")
        c[col + "_r"] = c[col]                 # raw: the extremes, for section 10
        c.loc[(c[col] < MIN_PRICE) | (c[col] > MAX_PRICE), col] = np.nan
    pre = c[(c["ts"] <= tip) & c["ask_c"].notna() & c["bid_c"].notna()]
    if pre.empty:
        return "no pregame quote", None
    p_close = float((pre["ask_c"].iloc[-1] + pre["bid_c"].iloc[-1]) / 2)

    margin = pl[:, 3] - pl[:, 4]
    chg = np.r_[True, np.diff(margin) != 0]
    last_change = pd.Series(np.where(chg, ts, np.nan)).ffill().to_numpy()
    live = c[c["ts"] > tip].reset_index(drop=True)
    j = np.searchsorted(ts, live["ts"].to_numpy(), side="right") - 1
    per, clk = pl[j, 1], pl[j, 2]
    by_ts = c.set_index("ts")
    fill_ts = live["ts"] + 60 * (1 + LAG_CANDLES)
    fill = by_ts.reindex(fill_ts.to_numpy())
    st = pd.DataFrame({
        "game_id": row.game_id, "ts": live["ts"].to_numpy(),
        "delta": margin[j], "tau": ((4 - per) * 720 + clk) / 2880.0,
        "reg": per <= 4,
        "quiet": (live["ts"].to_numpy() - last_change[j]) >= QUIET_SEC,
        "ask_c": live["ask_c"].to_numpy(), "bid_c": live["bid_c"].to_numpy(),
        "fill_ask": fill["ask_h"].to_numpy(), "fill_bid": fill["bid_l"].to_numpy(),
        "fill_mid": ((fill["ask_c"] + fill["bid_c"]) / 2).to_numpy(),
        "ask_r": live["ask_c_r"].to_numpy(), "bid_r": live["bid_c_r"].to_numpy(),
        "fill_ask_r": fill["ask_h_r"].to_numpy(), "fill_bid_r": fill["bid_l_r"].to_numpy(),
        "fill_mid_r": ((fill["ask_c_r"] + fill["bid_c_r"]) / 2).to_numpy(),
        "y": int(row.home_won)})
    st = st[st["reg"] & (st["tau"] > 0)].drop(columns="reg")
    st["mid"] = (st["ask_c"] + st["bid_c"]) / 2
    if st["mid"].notna().sum() < 30:
        return "no live quotes", None
    fin = espn["final"]
    rec = dict(game_id=row.game_id, slate_date=row.slate_date, home=row.home,
               away=row.away, y=int(row.home_won), p_model=float(row.p_home),
               p_close=p_close, kalshi_close_q_home=row.kalshi_close_q_home,
               close_book=row.close_book, close_spread=float(row.close_spread),
               tip_ts=tip, tip_delay_min=round((tip - sched) / 60, 1),
               pre_quote_min=round((tip - pre["ts"].iloc[-1]) / 60, 1),
               live_spread=float((st["ask_c"] - st["bid_c"]).median()),
               n_states=int(st["mid"].notna().sum()), stray_stamps=stray,
               espn_result_agrees=(fin[0] is not None and fin[1] is not None
                                   and int(fin[0] > fin[1]) == int(row.home_won)))
    return rec, st


def assemble(games, espn, kal):
    S, P, drops = [], [], {}
    for row in games.itertuples():
        rec, st = game_states(row, espn.get(row.game_id), kal.get(row.game_id))
        if st is None:
            drops[rec] = drops.get(rec, 0) + 1
            continue
        S.append(st)
        P.append(rec)
    S, P = pd.concat(S, ignore_index=True), pd.DataFrame(P)
    P["split"] = np.where(P["slate_date"] <= TRAIN_END, "train", "test")
    P["z_mkt"], P["z_model"] = Phi_inv(P["p_close"]), Phi_inv(P["p_model"])
    S = S.merge(P[["game_id", "split", "z_mkt", "z_model"]], on="game_id")
    return S, P, drops


# ----------------------------------------------------------------- fits ----
def sigma_process(espn, P):
    """Sigma from the score process alone: within-game variance of the four
    quarter margin changes (drift removed per game), scaled to 48 minutes.
    No outcomes are fitted and no prices read. (sigma, ±95, games)."""
    resid = []
    for gid in P["game_id"]:
        js = espn.get(gid)
        if not js:
            continue
        pl = np.array(js["plays"], float)
        m = pl[:, 3] - pl[:, 4]
        ends = [m[pl[:, 1] == k][-1] for k in (1, 2, 3, 4) if (pl[:, 1] == k).any()]
        if len(ends) < 4:
            continue
        inc = np.diff(np.r_[0, ends])
        resid.append(((inc - inc.mean()) ** 2).sum() / 3.0)
    resid = np.array(resid)
    s = math.sqrt(4 * resid.mean())
    se = 2 * resid.std(ddof=1) / math.sqrt(len(resid)) / s
    return s, Z95 * se, len(resid)


def spread_check(espn, P):
    """The score-process sigma's yardstick: quarter margins net of the
    closing spread's share (home margin expected -spread / 4 a quarter).
    (SD of the regulation margin around the spread, its ±95, the mean
    correlation of consecutive quarters' net margins, games). Negatively
    correlated quarters (leads shrink) make the within-game estimate above
    run higher than the game's SD."""
    q = []
    for gid, spread in zip(P["game_id"], P["close_spread"]):
        js = espn.get(gid)
        if not js or not np.isfinite(spread):
            continue
        pl = np.array(js["plays"], float)
        m = pl[:, 3] - pl[:, 4]
        ends = [m[pl[:, 1] == k][-1] for k in (1, 2, 3, 4) if (pl[:, 1] == k).any()]
        if len(ends) < 4:
            continue
        q.append(np.diff(np.r_[0, ends]) + spread / 4)
    q = np.array(q)
    tot = q.sum(1)
    sd = tot.std(ddof=1)
    se = sd / math.sqrt(2 * (len(tot) - 1))
    r = np.mean([np.corrcoef(q[:, k], q[:, k + 1])[0, 1] for k in range(3)])
    return sd, Z95 * se, r, len(tot)


def part_of_game(tau):
    """Index into TAU_BUCKETS for each state."""
    tau = np.asarray(tau, float)
    return np.select([tau > b[0] for b in TAU_BUCKETS], range(len(TAU_BUCKETS)), 0)


def loss_curves(S, zcol, target):
    """Cross-entropy of p*(sigma) against `target` (the outcome y, or the live
    mid as a soft label) on SIGMA_GRID, each game weighted equally: array
    (games, parts of the game, grid). Only states with a mid, so the two
    targets are scored on the same minutes."""
    s = S.dropna(subset=["mid", zcol])
    gi, games = pd.factorize(s["game_id"])
    code = gi * len(TAU_BUCKETS) + part_of_game(s["tau"])
    n = len(games) * len(TAU_BUCKETS)
    wt = per_game(s)
    t = s[target].to_numpy(float)
    d, tau, z = (s[c].to_numpy(float) for c in ("delta", "tau", zcol))
    M = np.empty((n, len(SIGMA_GRID)))
    for k, sg in enumerate(SIGMA_GRID):
        p = clip(p_star(d, tau, z, sg))
        M[:, k] = np.bincount(code, wt * -(t * np.log(p) + (1 - t) * np.log(1 - p)), n)
    return M.reshape(len(games), len(TAU_BUCKETS), len(SIGMA_GRID))


def sigma_fits(S, zcol, boot=N_BOOT, seed=7):
    """Sigma fitted to the outcome and to the live mid (same states, same
    p* form), for all minutes and each part of the game, with game-bootstrap
    ±95 and the same for mid - outcome. {part: (outcome, ±, mid, ±, diff, ±)}."""
    cy, cm = loss_curves(S, zcol, "y"), loss_curves(S, zcol, "mid")
    rng = np.random.default_rng(seed)
    W = rng.multinomial(cy.shape[0], np.full(cy.shape[0], 1 / cy.shape[0]), boot)
    out = {}
    for name, sl in [("all minutes", slice(None))] + [
            (b[2], slice(k, k + 1)) for k, b in enumerate(TAU_BUCKETS)]:
        gy, gm = cy[:, sl].sum(1), cm[:, sl].sum(1)
        fy, fm = SIGMA_GRID[gy.sum(0).argmin()], SIGMA_GRID[gm.sum(0).argmin()]
        by, bm = SIGMA_GRID[(W @ gy).argmin(1)], SIGMA_GRID[(W @ gm).argmin(1)]
        out[name] = (fy, Z95 * by.std(), fm, Z95 * bm.std(), fm - fy, Z95 * (bm - by).std())
    return out


def sigma_text(f):
    s = f"{f[0]:5.2f} ± {f[1]:4.2f}"
    return s + (" (grid edge)" if f[0] in (SIGMA_GRID[0], SIGMA_GRID[-1]) else "")


def phi_tails(eta):
    """(Phi(eta), 1 - Phi(eta), phi(eta)), each tail from erfc, so neither
    rounds to 0 or 1 within |eta| <= 8."""
    eta = np.clip(np.asarray(eta, float), -8, 8)
    P = (0.5 * _erfc(-eta / math.sqrt(2))).astype(float)
    Q = (0.5 * _erfc(eta / math.sqrt(2))).astype(float)
    return P, Q, np.exp(-eta * eta / 2) / math.sqrt(2 * math.pi)


def probit_fit(X, t, wts, start, iters=100):
    """Weighted probit of t (0/1, or a price in (0, 1) as a soft label) on X,
    no intercept, by Fisher scoring. (coefs, Fisher information, per-row
    scores), the last two for a clustered covariance."""
    b = np.asarray(start, float)
    for _ in range(iters):
        P, Q, f = phi_tails(X @ b)
        info = X.T @ (X * (wts * f * f / (P * Q))[:, None])
        step = np.linalg.solve(info, X.T @ (wts * f * (t - P) / (P * Q)))
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    P, Q, f = phi_tails(X @ b)
    info = X.T @ (X * (wts * f * f / (P * Q))[:, None])
    return b, info, X * (wts * f * (t - P) / (P * Q))[:, None]


def free_fits(S, zcol):
    """The prior's weight set free: p* = Phi(delta / (sigma sqrt(tau)) +
    k z0 sqrt(tau)), a probit on delta / sqrt(tau) and z0 sqrt(tau) with
    coefficients 1 / sigma and k (k = 1 is the notebook's form). Fitted to
    the outcome and to the live mid on the same minutes, all minutes and
    each part of the game, each game weighted equally; the two fits share a
    game-clustered covariance, so mid - outcome carries its own ±95.
    {part: {'out': (1/sigma, k) of the outcome fit, 'sig_y', 'k_y', 'sig_m',
    'k_m', 'd_sig', 'd_k': (estimate, ±95)}}."""
    s = S.dropna(subset=["mid", zcol])
    X = np.c_[s["delta"] / np.sqrt(s["tau"]), s[zcol] * np.sqrt(s["tau"])]
    y, mid, wt = s["y"].to_numpy(float), s["mid"].to_numpy(float), per_game(s)
    part, games = part_of_game(s["tau"]), s["game_id"].to_numpy()
    out = {}
    for name, m in [("all minutes", np.ones(len(s), bool))] + [
            (b[2], part == k) for k, b in enumerate(TAU_BUCKETS)]:
        by, iy, uy = probit_fit(X[m], y[m], wt[m], (1 / 17.0, 1.0))
        bm, im, um = probit_fit(X[m], mid[m], wt[m], (1 / 17.0, 1.0))
        H = np.zeros((4, 4))
        H[:2, :2], H[2:, 2:] = np.linalg.inv(iy), np.linalg.inv(im)
        U = pd.DataFrame(np.c_[uy, um]).groupby(games[m]).sum().to_numpy()
        V = H @ (U.T @ U) @ H
        sy, sm = 1 / by[0], 1 / bm[0]

        def ci(grad):
            g = np.asarray(grad, float)
            return Z95 * math.sqrt(float(g @ V @ g))
        out[name] = dict(
            out=by, sig_y=(sy, ci([-sy ** 2, 0, 0, 0])), k_y=(by[1], ci([0, 1, 0, 0])),
            sig_m=(sm, ci([0, 0, -sm ** 2, 0])), k_m=(bm[1], ci([0, 0, 0, 1])),
            d_sig=(sm - sy, ci([sy ** 2, 0, -sm ** 2, 0])), d_k=(bm[1] - by[1], ci([0, -1, 0, 1])))
    return out


def score_beyond_mid(S):
    """The score's and the prior's weight beyond the live mid, no p* form:
    y ~ logit(mid) [offset] + a delta / sqrt(tau) + b z0 sqrt(tau), each game
    weighted equally, all minutes and each part of the game.
    {part: ((a, b), ±95, games)}."""
    s = S.dropna(subset=["mid"])
    out = {}
    for name, x in [("all minutes", s)] + [
            (n, s[(s["tau"] > lo) & (s["tau"] <= hi)]) for lo, hi, n in TAU_BUCKETS]:
        X = np.c_[x["delta"] / np.sqrt(x["tau"]), x["z_mkt"] * np.sqrt(x["tau"])]
        b, ci = offset_fit(X, x["y"], mr.logit(x["mid"]), per_game(x), x["game_id"])
        out[name] = (b, ci, x["game_id"].nunique())
    return out


def p_mid_adj(df, fit, by_part=False, score_only=False):
    """The live mid corrected by a fit of score_beyond_mid: all minutes, or
    each part of the game with its own (a, b); score_only drops b (the
    prior's term, which flips with the season-phase calibration). NaN where
    there is no mid."""
    u = (df["delta"] / np.sqrt(df["tau"])).to_numpy()
    v = (df["z_mkt"] * np.sqrt(df["tau"])).to_numpy()
    if by_part:
        coef = np.array([fit[b[2]][0] for b in TAU_BUCKETS])[part_of_game(df["tau"])]
    else:
        coef = np.tile(fit["all minutes"][0], (len(df), 1))
    eta = mr.logit(df["mid"]) + coef[:, 0] * u + (0 if score_only else coef[:, 1] * v)
    return np.where(df["mid"].notna(), 1 / (1 + np.exp(-eta)), np.nan)


def p_free(df, fit, by_part=False):
    """p* with the train outcome fit's (sigma, k): all minutes, or each part
    of the game with its own pair."""
    if not by_part:
        s, k = fit["all minutes"]["out"]
        return p_star(df["delta"], df["tau"], k * df["z_mkt"], 1 / s)
    part = part_of_game(df["tau"])
    out = np.full(len(df), np.nan)
    for j, b in enumerate(TAU_BUCKETS):
        s, k = fit[b[2]]["out"]
        m = part == j
        out[m] = p_star(df["delta"][m], df["tau"][m], k * df["z_mkt"][m], 1 / s)
    return out


def blend_w(P, lo=0.0, hi=1.0):
    """w maximising the pregame likelihood of Phi(w z_model + (1-w) z_mkt);
    (w, ±95) from the observed information (one row per game)."""
    y = P["y"].to_numpy(float)
    f = lambda w: float(ll(Phi(w * P["z_model"] + (1 - w) * P["z_mkt"]), y).sum())  # noqa: E731
    w = golden(f, lo, hi)
    h = 1e-3
    hess = (f(w + h) - 2 * f(w) + f(w - h)) / h ** 2
    return w, (Z95 / math.sqrt(hess) if hess > 0 else float("nan"))


def offset_fit(X, y, off, wts, groups, iters=50):
    """Weighted logistic fit with an offset; (coefs, game-clustered ±95)."""
    X, y, off, wts = (np.asarray(v, float) for v in (X, y, off, wts))
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(off + X @ b)))
        H = X.T @ (X * (wts * mu * (1 - mu))[:, None])
        step = np.linalg.solve(H, X.T @ (wts * (y - mu)))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    mu = 1 / (1 + np.exp(-(off + X @ b)))
    H = X.T @ (X * (wts * mu * (1 - mu))[:, None])
    U = pd.DataFrame(X * (wts * (y - mu))[:, None]).groupby(np.asarray(groups)).sum().to_numpy()
    Hi = np.linalg.inv(H)
    V = Hi @ (U.T @ U) @ Hi
    return b, Z95 * np.sqrt(np.diag(V))


# ---------------------------------------------------------------- tests ----
def per_game(s):
    """Row weights that give each game the same total weight."""
    return 1 / s.groupby("game_id")["ts"].transform("count").to_numpy(float)


def per_game_diff(S, a, b):
    """Game-clustered mean of log loss(a) - log loss(b) (negative = a better),
    and of Brier; each game weighted equally."""
    s = S.dropna(subset=[a, b])
    d = pd.Series(ll(s[a], s["y"]) - ll(s[b], s["y"]), index=s.index)
    db = pd.Series((s[a] - s["y"]) ** 2 - (s[b] - s["y"]) ** 2, index=s.index)
    g = s["game_id"]
    return mr.mean_ci(d.groupby(g).mean()), mr.mean_ci(db.groupby(g).mean()), g.nunique()


def diff_line(label, S, a, b):
    (m, ci), (mb, cib), n = per_game_diff(S, a, b)
    return (f"  {label:<38} log loss {m:+.4f} ± {ci:.4f}   Brier {mb:+.4f} ± {cib:.4f}"
            f"   games {n}")


def weight_line(label, S, pcol):
    """The outcome's weight on p*'s disagreement with the live mid:
    y ~ logit(mid) [offset] + w (logit p* - logit mid); 0 = p* adds nothing
    to the mid, 1 = p* is right where they differ."""
    s = S.dropna(subset=["mid", pcol])
    lm = mr.logit(s["mid"])
    d = mr.logit(s[pcol]) - lm
    b, ci = offset_fit(d[:, None], s["y"], lm, per_game(s), s["game_id"])
    return f"  {label:<38} w {b[0]:+.2f} ± {ci[0]:.2f}"


def slope_line(label, s, pcol):
    """Calibration of a live price against the outcome, game-clustered:
    slope > 1 = too timid (under-reacts), < 1 = too confident (over-reacts)."""
    s = s.dropna(subset=[pcol])
    X = np.c_[np.ones(len(s)), mr.logit(s[pcol])]
    b, ci = offset_fit(X, s["y"], np.zeros(len(s)), per_game(s), s["game_id"])
    return (f"  {label:<38} slope {b[1]:.2f} ± {ci[1]:.2f}   intercept {b[0]:+.3f} ± {ci[0]:.3f}"
            f"   games {s['game_id'].nunique()}")


def run_trades(S, pcol, label):
    """First qualifying signal per game: quiet minute, tau in the window, the
    better side's p* - ask - fee >= ENTRY_EDGE on the signal candle; filled at
    the worst quote of the candle (1 + LAG) minutes later (YES at its high
    ask, NO at 1 - the low bid). pnl per contract, its null (the same
    contract valued at the fill minute's mid, less price and fee) and the
    live favourite (side with mid >= .5 at the fill minute, same fill rule)."""
    s = S[S["tau"].between(TAU_MIN, TAU_MAX) & S["quiet"]].copy()
    s = s.dropna(subset=["ask_c", "bid_c", pcol])
    p = s[pcol].to_numpy()
    no_ask = 1 - s["bid_c"].to_numpy()
    yes_sig = p - s["ask_c"].to_numpy() - fee(s["ask_c"])
    no_sig = (1 - p) - no_ask - fee(no_ask)
    s["side"] = np.where(yes_sig >= no_sig, "YES", "NO")
    s["sig_edge"] = np.maximum(yes_sig, no_sig)
    s["p"] = p
    s = s[s["sig_edge"] >= ENTRY_EDGE]
    s["fill"] = np.where(s["side"] == "YES", s["fill_ask"], 1 - s["fill_bid"])
    s = s[s["fill"].between(MIN_PRICE, MAX_PRICE) & s["fill_mid"].notna()]
    t = s.sort_values("ts").groupby("game_id").head(1).copy()
    if t.empty:
        return t
    yes = (t["side"] == "YES").to_numpy()
    y = t["y"].to_numpy()
    t["won"] = np.where(yes, y, 1 - y)
    t["fee"] = fee(t["fill"])
    t["pnl"] = t["won"] - t["fill"] - t["fee"]
    t["null"] = np.where(yes, t["fill_mid"], 1 - t["fill_mid"]) - t["fill"] - t["fee"]
    fav_yes = t["fill_mid"].to_numpy() >= 0.5
    fav_px = np.where(fav_yes, t["fill_ask"], 1 - t["fill_bid"])
    fav_won = np.where(fav_yes, y, 1 - y)
    t["fav_pnl"] = np.where(np.isfinite(fav_px), fav_won - fav_px - fee(np.nan_to_num(fav_px, nan=.5)),
                            np.nan)
    t["prior"] = label
    return t[["game_id", "split", "prior", "ts", "side", "tau", "delta", "p", "ask_c",
              "bid_c", "sig_edge", "fill", "fee", "fill_mid", "y", "won", "pnl",
              "null", "fav_pnl"]]


def trade_line(label, t):
    if len(t) < 20:
        return f"  {label:<14} trades {len(t):<4} (too few)"
    m, ci = mr.mean_ci(t["pnl"])
    mn, cin = mr.mean_ci(t["pnl"] - t["null"])
    fav = t["fav_pnl"].dropna()
    roi = t["pnl"].sum() / (t["fill"] + t["fee"]).sum()
    return (f"  {label:<14} trades {len(t):<4} YES/NO {int((t['side'] == 'YES').sum())}/"
            f"{int((t['side'] == 'NO').sum())}  fill {t['fill'].mean():.3f}  "
            f"hit {t['won'].mean():.3f}  claimed {100 * t['sig_edge'].mean():+.1f}c  "
            f"pnl {100 * m:+.1f}c ± {100 * ci:.1f}  null {100 * t['null'].mean():+.1f}c  "
            f"pnl - null {100 * mn:+.1f}c ± {100 * cin:.1f}  ROI {100 * roi:+.1f}%  "
            f"fav {100 * fav.mean():+.1f}c")


def edge_rows(t):
    out = []
    t = t.assign(edge_bin=pd.cut(t["sig_edge"], EDGE_BINS))
    for b, x in t.groupby("edge_bin", observed=True):
        m, ci = mr.mean_ci(x["pnl"]) if len(x) > 1 else (x["pnl"].mean(), float("nan"))
        out.append(f"    {str(b):<14} n {len(x):<4} claimed {100 * x['sig_edge'].mean():+5.1f}c"
                   f"  realised {100 * m:+5.1f}c ± {100 * ci:4.1f}"
                   f"  null {100 * x['null'].mean():+5.1f}c")
    return out


def timing_lines(S, pcol="p_z_mkt"):
    """Correlation of minute-on-minute changes in logit(mid) with changes in
    logit(p*) L minutes earlier (L > 0: the mid moves after ESPN's wallclock
    says the score changed). A peak at L = 0 means the two clocks agree to
    within a minute."""
    s = S.dropna(subset=["mid"]).sort_values(["game_id", "ts"]).copy()
    s["lm"], s["lp"] = mr.logit(s["mid"]), mr.logit(s[pcol])
    g = s.groupby("game_id")
    s["dm"] = g["lm"].diff().where(g["ts"].diff() == 60)
    s["dp"] = g["lp"].diff().where(g["ts"].diff() == 60)
    out = []
    for L in (-2, -1, 0, 1, 2):
        dp = s.groupby("game_id")["dp"].shift(L).where(
            s.groupby("game_id")["ts"].shift(L) == s["ts"] - 60 * L)
        ok = s["dm"].notna() & dp.notna()
        out.append(f"    L = {L:+d} min   corr {np.corrcoef(s['dm'][ok], dp[ok])[0, 1]:+.3f}"
                   f"   (n {int(ok.sum())})")
    return out


def ols_cluster(x, y, groups):
    """Slope of y on x (with an intercept) and its game-clustered ±95."""
    X = np.c_[np.ones(len(x)), np.asarray(x, float)]
    y = np.asarray(y, float)
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    U = pd.DataFrame(X * (y - X @ b)[:, None]).groupby(np.asarray(groups)).sum().to_numpy()
    V = XtXi @ (U.T @ U) @ XtXi
    return b[1], Z95 * math.sqrt(V[1, 1])


def minute_panel(S, pcol):
    """One row per live minute t (the candle ending at ts) with the minutes
    around it, matched by time within the game: the mid's change over
    minute t (dm), the next minute (dm1) and minutes t+2..t+6 (drift), the
    score-implied change over minute t (dp, from pcol), the spread at t, and
    whether the margin moved in minute t or t+1. Prices in cents."""
    s = S[["game_id", "ts", "tau", "delta", "mid", "ask_c", "bid_c", pcol]].copy()
    key = s.set_index(["game_id", "ts"])

    def at(k, col):
        idx = pd.MultiIndex.from_arrays([s["game_id"], s["ts"] + 60 * k])
        return key[col].reindex(idx).to_numpy()
    m = s["mid"].to_numpy()
    s["dm"] = 100 * (m - at(-1, "mid"))
    s["dm1"] = 100 * (at(1, "mid") - m)
    s["drift"] = 100 * (at(6, "mid") - at(1, "mid"))
    s["dp"] = 100 * (s[pcol].to_numpy() - at(-1, pcol))
    s["spread"] = 100 * (s["ask_c"] - s["bid_c"])
    d = s["delta"].to_numpy()
    s["still"] = (d == at(-1, "delta")) & (d == at(1, "delta"))
    return s.dropna(subset=["dm", "dm1", "dp", "spread"])


def fluctuation_lines(S, pcol):
    """The live mid's minute moves against the spread and the move the score
    and clock justify (pcol), by time remaining; whether the mid keeps
    following a score move after it (drift > 0: under-reaction) or gives it
    back (< 0: over-reaction); and the drift after big score moves against
    the cost of crossing (half-spread + fee)."""
    panel = minute_panel(S, pcol)
    out = []
    for name, x in [("all minutes", panel)] + [
            (n, panel[(panel["tau"] > lo) & (panel["tau"] <= hi)]) for lo, hi, n in TAU_BUCKETS]:
        g = x["game_id"]
        b0, c0 = ols_cluster(x["dp"], x["dm"], g)
        b1, c1 = ols_cluster(x["dp"], x["dm1"], g)
        xd = x.dropna(subset=["drift"])
        b5, c5 = ols_cluster(xd["dp"], xd["drift"], xd["game_id"])
        q = x[x["still"]]
        cov = float(np.cov(q["dm"], q["dm1"])[0, 1])
        within = (x["dm"].abs() <= x["spread"]).mean()
        out.append(f"  {name:<12} minutes {len(x):>6,}  SD mid move {x['dm'].std():4.2f}c  "
                   f"SD score move {x['dp'].std():4.2f}c  spread {x['spread'].median():.1f}c  "
                   f"within spread {100 * within:3.0f}%")
        out.append(f"  {'':<12} per cent of score move: same minute {b0:.2f} ± {c0:.2f}, next "
                   f"{b1:+.2f} ± {c1:.2f}, minutes 2-6 {b5:+.2f} ± {c5:.2f};  noise SD "
                   f"{math.sqrt(max(0.0, -cov)):.2f}c (half-spread "
                   f"{x['spread'].median() / 2:.2f}c)")
        big = xd[xd["dp"].abs() >= 3]
        if len(big) >= 50:
            sgn = np.sign(big["dp"])
            dm1 = (sgn * big["dm1"]).groupby(big["game_id"]).mean()
            dr = (sgn * big["drift"]).groupby(big["game_id"]).mean()
            cost = (big["spread"] / 2 + 100 * fee(big["mid"])).mean()
            m1, ci1 = mr.mean_ci(dm1)
            m5, ci5 = mr.mean_ci(dr)
            out.append(f"  {'':<12} after score moves >= 3c ({len(big):,}): next minute "
                       f"{m1:+.2f}c ± {ci1:.2f}, minutes 2-6 {m5:+.2f}c ± {ci5:.2f} the score's "
                       f"way; cost to cross {cost:.2f}c")
    return out


def straddle_game(candles, tip, y, up):
    """Both sides bought at the last pregame candle at or before the first
    play (home YES at its ask; the away side at 1 - the home bid, Kalshi's
    other market priced through the home book), each with a resting sell at
    its entry price + `up`, filled at the first later candle whose bid for
    that side (home: bid; away: 1 - ask) reaches it; an unsold side settles.
    Raw candles, so late quotes near 0 or 1 count. Per $1 pair: dict or None."""
    c = pd.DataFrame(candles, columns=["ts", "ask_c", "ask_h", "bid_c", "bid_l", "vol"])
    c = c.dropna(subset=["ts"]).sort_values("ts")
    pre = c[(c["ts"] <= tip) & c["ask_c"].notna() & c["bid_c"].notna()]
    if pre.empty:
        return None
    e = pre.iloc[-1]
    live = c[c["ts"] > e["ts"]]
    mid = (live["ask_c"] + live["bid_c"]) / 2
    legs = (("home", float(e["ask_c"]), live["bid_c"], mid, y),
            ("away", 1 - float(e["bid_c"]), 1 - live["ask_c"], 1 - mid, 1 - y))
    out = dict(cost=sum(market.kalshi_cost(a) for _, a, _, _, _ in legs), sold=[], won_sold=[],
               overshoot=[], fills=[], paths=[], proceeds=0.0, maker=0.0)
    for side, a, bid, fair, pays in legs:
        target = round(a + up, 2)
        b = bid.to_numpy(float)
        out["paths"].append(dict(side=side, won=pays, peak=np.nanmax(b) - a if np.isfinite(b).any()
                                 else np.nan, trough=np.nanmin(b) - a if np.isfinite(b).any()
                                 else np.nan))
        hit = b >= target - 1e-9
        if target <= 0.99 and hit.any():
            k = int(hit.argmax())
            out["proceeds"] += target
            out["maker"] += MAKER_FEE * target * (1 - target)
            out["sold"].append(side)
            out["won_sold"].append(pays)
            out["overshoot"].append(float(fair.to_numpy()[k]) - target)
            out["fills"].append(dict(side=side, ts=float(live["ts"].to_numpy()[k]), target=target,
                                     fair=float(fair.to_numpy()[k]), won=pays))
        else:
            out["proceeds"] += pays
    out["profit"] = out["proceeds"] - out["cost"]
    out["profit_maker"] = out["profit"] - out["maker"]
    out["hold"] = 1 - out["cost"]
    return out


def straddle_lines(P, kal):
    """Buy both sides at tip, sell each at +up: profit per $1 pair beside
    holding both (the cost of the pair), all games and each period."""
    out = []
    rows = []
    for up in STRADDLE_TARGETS:
        for g in P.itertuples():
            r = straddle_game(kal[g.game_id]["candles"], g.tip_ts, g.y, up)
            if r:
                rows.append(dict(up=up, split=g.split, n_sold=len(r["sold"]),
                                 sold_winner=int(len(r["sold"]) == 1 and r["won_sold"][0] == 1),
                                 overshoot=sum(v for v in r["overshoot"] if v == v),
                                 n_over=sum(1 for v in r["overshoot"] if v == v),
                                 **{k: r[k] for k in ("profit", "profit_maker", "hold")}))
    R = pd.DataFrame(rows)
    for up in STRADDLE_TARGETS:
        for lab, x in (("all", R[R["up"] == up]),
                       ("Oct-Jan", R[(R["up"] == up) & (R["split"] == "train")]),
                       ("Feb-Apr", R[(R["up"] == up) & (R["split"] == "test")])):
            m, ci = mr.mean_ci(x["profit"])
            mm, cim = mr.mean_ci(x["profit_maker"])
            d, cid = mr.mean_ci(x["profit"] - x["hold"])
            share = [100 * (x["n_sold"] == k).mean() for k in (2, 1, 0)]
            out.append(f"  +{100 * up:2.0f}c {lab:<8} games {len(x):<5} both sold {share[0]:3.0f}%  "
                       f"one {share[1]:3.0f}%  none {share[2]:3.0f}%   profit {100 * m:+5.2f}c ± "
                       f"{100 * ci:4.2f} (maker fee {100 * mm:+5.2f}c ± {100 * cim:4.2f})   hold both "
                       f"{100 * x['hold'].mean():+5.2f}c   profit - hold {100 * d:+5.2f}c ± {100 * cid:4.2f}")
        x = R[R["up"] == up]
        parts = []
        for k, name in ((2, "both sold"), (1, "one sold"), (0, "none sold")):
            xx = x[x["n_sold"] == k]
            if len(xx):
                parts.append(f"{name} {100 * xx['profit'].mean():+.1f}c")
        one = x[x["n_sold"] == 1]
        if len(one):
            parts.append(f"one sold and it won {100 * one['sold_winner'].mean():.0f}% "
                         f"({100 * one[one['sold_winner'] == 1]['profit'].mean():+.1f}c) / it lost "
                         f"({100 * one[one['sold_winner'] == 0]['profit'].mean():+.1f}c)")
        out.append("        mean profit per pair: " + ";  ".join(parts))
        mo, cio = mr.mean_ci(x["overshoot"])
        out.append(f"        the mid was already past the target at the fill minute by "
                   f"{100 * x['overshoot'].sum() / x['n_over'].sum():+.2f}c a sold side "
                   f"({x['n_sold'].sum()} sales, {x['n_over'].sum()} with a mid); "
                   f"{100 * mo:+.2f}c ± {100 * cio:.2f} a pair, against profit - hold "
                   f"{100 * (x['profit'] - x['hold']).mean():+.2f}c")
    return out


def trigger_frames(P, kal, S, up=0.10):
    """The +up fills of straddle_game as events, with the side's state at
    the fill minute from S (lead, tau, the mid's move that minute, p* with
    the train (sigma, k) fit; NaN in overtime or beyond the dead-quote
    bounds), and every side's path extremes from entry. (events, paths)."""
    ev, paths = [], []
    for g in P.itertuples():
        r = straddle_game(kal[g.game_id]["candles"], g.tip_ts, g.y, up)
        if not r:
            continue
        for k, f in enumerate(sorted(r["fills"], key=lambda f: f["ts"])):
            ev.append(dict(game_id=g.game_id, split=g.split, order=k + 1, **f))
        paths += [dict(game_id=g.game_id, split=g.split, **p) for p in r["paths"]]
    ev, paths = pd.DataFrame(ev), pd.DataFrame(paths)
    s = S.set_index(["game_id", "ts"])
    here = s.reindex(pd.MultiIndex.from_arrays([ev["game_id"], ev["ts"]]))
    before = s.reindex(pd.MultiIndex.from_arrays([ev["game_id"], ev["ts"] - 60]))
    home = (ev["side"] == "home").to_numpy()
    sgn = np.where(home, 1.0, -1.0)
    ev["lead"] = sgn * here["delta"].to_numpy() / np.sqrt(here["tau"].to_numpy())
    ev["tau"] = here["tau"].to_numpy()
    ev["jump"] = 100 * sgn * (here["mid"].to_numpy() - before["mid"].to_numpy())
    p = here["p_z_mkt_fk"].to_numpy()
    ev["resid"] = 100 * (ev["fair"] - np.where(home, p, 1 - p))
    ev["over"] = 100 * (ev["fair"] - ev["target"])
    return ev, paths


def cluster_mean_ci(d, groups):
    """Mean of d over rows, its ±95 clustered by game (a per-game average
    would weight games by their number of fills, which depends on later
    play)."""
    d = np.asarray(d, float)
    m = d.mean()
    u = pd.Series(d - m).groupby(np.asarray(groups)).sum().to_numpy()
    return m, Z95 * math.sqrt(float((u ** 2).sum())) / len(d)


def trigger_lines(ev, paths):
    """Is the side that reached +up the eventual winner more often than its
    price at that minute says, and does anything at that minute (overshoot,
    the minute's jump, the mid's residual from p*, the lead, time left, first
    or second fill) predict it beyond the price? Then how far winners and
    losers run from entry."""
    out = []
    ev = ev.dropna(subset=["fair"])
    ev = ev[ev["fair"].between(0, 1, inclusive="neither")]
    for lab, x in (("all", ev), ("Oct-Jan", ev[ev["split"] == "train"]),
                   ("Feb-Apr", ev[ev["split"] == "test"])):
        m, ci = cluster_mean_ci(x["won"] - x["fair"], x["game_id"])
        out.append(f"  {lab:<8} fills {len(x):<5} won {100 * x['won'].mean():4.1f}%  price said "
                   f"{100 * x['fair'].mean():4.1f}%  won - price {100 * m:+5.1f} pp ± {100 * ci:.1f}")
    out.append("  by price at the fill (all):")
    for lo, hi in ((0, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.0)):
        x = ev[(ev["fair"] > lo) & (ev["fair"] <= hi)]
        if len(x) >= 20:
            m, ci = cluster_mean_ci(x["won"] - x["fair"], x["game_id"])
            out.append(f"    {lo:.1f}-{hi:.1f}  fills {len(x):<5} won {100 * x['won'].mean():4.1f}%  "
                       f"price {100 * x['fair'].mean():4.1f}%  gap {100 * m:+5.1f} pp ± {100 * ci:.1f}")
    out.append("  beyond the price: won ~ logit(price) [offset] + beta x, one x at a time, game-")
    out.append("  clustered (beta > 0 = more x, more often the winner than the price says):")
    ev = ev.assign(second=(ev["order"] == 2).astype(float), late=1 - ev["tau"])
    for col, name, unit in (("over", "overshoot past target", "per c"),
                            ("jump", "price jump that minute", "per c"),
                            ("resid", "price - p* residual", "per c"),
                            ("lead", "lead / sqrt(tau)", "per pt"),
                            ("late", "share of game played", "per game"),
                            ("second", "second fill (comeback)", "")):
        parts = []
        for lab, x in (("all", ev), ("Oct-Jan", ev[ev["split"] == "train"]),
                       ("Feb-Apr", ev[ev["split"] == "test"])):
            x = x.dropna(subset=[col])
            if len(x) < 50 or x[col].std() == 0:
                parts.append(f"{lab} n/a")
                continue
            b, ci = offset_fit(x[[col]].to_numpy(float), x["won"], mr.logit(x["fair"]),
                               np.ones(len(x)), x["game_id"])
            parts.append(f"{lab} {b[0]:+.3f} ± {ci[0]:.3f} (n {len(x)})")
        out.append(f"    {name:<24} {unit:<8} " + "   ".join(parts))
    out.append("  How far each side runs from its entry price (its bid; all games):")
    for won, name in ((1, "eventual winners"), (0, "eventual losers")):
        x = paths[paths["won"] == won].dropna(subset=["peak"])
        ups = "  ".join(f"+{c}c {100 * (x['peak'] >= c / 100 - 1e-9).mean():3.0f}%"
                        for c in (5, 10, 20, 30))
        downs = "  ".join(f"-{c}c {100 * (x['trough'] <= -c / 100 + 1e-9).mean():3.0f}%"
                          for c in (10, 20, 30))
        out.append(f"    {name:<17} n {len(x):<5} reach {ups}   median peak "
                   f"{100 * x['peak'].median():+.0f}c;  dip {downs}   median low "
                   f"{100 * x['trough'].median():+.0f}c")
    return out


def quarter_frame(S, P):
    """One row per game: the home mid at the tip (p_close) and at the end of
    each of the first three quarters (the last live minute with a mid before
    the next quarter's clock runs, the break included), and the SD of the
    mid's minute moves (cents) from tip to each quarter's end."""
    s = S.dropna(subset=["mid"]).sort_values(["game_id", "ts"]).copy()
    g = s.groupby("game_id")
    s["dm"] = (100 * g["mid"].diff()).where(g["ts"].diff() == 60)
    rows = []
    for gid, x in s.groupby("game_id"):
        r = {"game_id": gid}
        for q, b in enumerate(QUARTER_ENDS, 1):
            xx = x[x["tau"] >= b - 1e-9]
            r[f"mid{q}"] = xx["mid"].iloc[-1] if len(xx) else np.nan
            r[f"vol{q}"] = xx["dm"].std() if xx["dm"].notna().sum() >= 10 else np.nan
        rows.append(r)
    return P[["game_id", "split", "y", "p_close"]].merge(pd.DataFrame(rows), on="game_id")


def shape_lines(Q):
    """Does the price path so far predict the winner beyond the price at the
    end of Q1, Q2 and Q3? One feature at a time, home side's view: y ~
    logit(home mid then) [offset] + beta x. Then the Q3-end favourite's
    up/down quarters against its price, and (hindsight) winners' paths."""
    out = []
    mids = ["p_close", "mid1", "mid2", "mid3"]
    for q in (1, 2, 3):
        x = Q.dropna(subset=mids[:q + 1] + [f"vol{q}"])
        x = x[np.logical_and.reduce([x[c].between(0, 1, inclusive="neither")
                                      for c in mids[:q + 1]])].copy()
        lm = mr.logit(x[f"mid{q}"])
        x["last"] = 10 * (x[f"mid{q}"] - x[mids[q - 1]])
        x["since"] = 10 * (x[f"mid{q}"] - x["p_close"])
        x["chop"] = x[f"vol{q}"] * lm
        feats = [("last", "this quarter's move", "per 10c"),
                 ("chop", "choppiness x logit(price)", "per c")]
        if q >= 2:                     # at Q1's end the move since tip is this quarter's
            x["prev"] = 10 * (x[mids[q - 1]] - x[mids[q - 2]])
            feats[1:1] = [("prev", "previous quarter's move", "per 10c"),
                          ("since", "move since tip", "per 10c")]
        X = np.c_[np.ones(len(x)), lm]
        b, ci = offset_fit(X, x["y"], np.zeros(len(x)), np.ones(len(x)), x["game_id"])
        m, cim = mr.mean_ci(x["y"] - x[f"mid{q}"])
        out.append(f"  end of Q{q}: games {len(x)}  home won {100 * x['y'].mean():.1f}%  price "
                   f"{100 * x[f'mid{q}'].mean():.1f}%  (won - price {100 * m:+.1f} pp ± "
                   f"{100 * cim:.1f})  price calibration slope {b[1]:.2f} ± {ci[1]:.2f}")
        for col, name, unit in feats:
            parts = []
            for lab, xx in (("all", x), ("Oct-Jan", x[x["split"] == "train"]),
                            ("Feb-Apr", x[x["split"] == "test"])):
                bb, cc = offset_fit(xx[[col]].to_numpy(float), xx["y"], mr.logit(xx[f"mid{q}"]),
                                    np.ones(len(xx)), xx["game_id"])
                parts.append(f"{lab} {bb[0]:+.3f} ± {cc[0]:.3f}")
            out.append(f"    {name:<27} {unit:<8} " + "   ".join(parts))
    x = Q.dropna(subset=mids)
    x = x[np.logical_and.reduce([x[c].between(0, 1, inclusive="neither") for c in mids])]
    fav = (x["mid3"] >= 0.5).to_numpy()
    side = lambda c: np.where(fav, x[c], 1 - x[c])                 # noqa: E731
    path = np.c_[side("mid1") - side("p_close"), side("mid2") - side("mid1"),
                 side("mid3") - side("mid2")]
    pat = ["".join("+" if v > 0 else "-" for v in r) for r in path]
    t = pd.DataFrame(dict(pat=pat, won=np.where(fav, x["y"], 1 - x["y"]), price=side("mid3")))
    out.append("  The Q3-end favourite's price by quarter (+ up, - down or flat: Q1 Q2 Q3), won vs")
    out.append("  its price then:")
    for p_, xx in t.groupby("pat"):
        if len(xx) >= 20:
            m, ci = mr.mean_ci(xx["won"] - xx["price"])
            out.append(f"    {p_}  games {len(xx):<4} won {100 * xx['won'].mean():4.1f}%  price "
                       f"{100 * xx['price'].mean():4.1f}%  gap {100 * m:+5.1f} pp ± {100 * ci:.1f}")
    won = np.where(x["y"] == 1, 1.0, -1.0)[:, None] * np.c_[
        x["mid1"] - x["p_close"], x["mid2"] - x["mid1"], x["mid3"] - x["mid2"]]
    out.append("  Hindsight: the eventual winner's price move by quarter, mean (share up): " + ",  ".join(
        f"Q{k + 1} {100 * won[:, k].mean():+.1f}c ({100 * (won[:, k] > 0).mean():.0f}%)"
        for k in range(3)))
    return out


def fav_frame(S, tau_max):
    """Live minutes with tau <= tau_max and a raw two-sided quote, from the
    favourite's side (the side with the raw mid >= .5): its price, whether
    it won, its ask now and its worst quote and mid (1 + LAG) min later."""
    s = S[(S["tau"] <= tau_max)].dropna(subset=["ask_r", "bid_r"])
    s = s[s["ask_r"] > s["bid_r"]].copy()
    mid = (s["ask_r"] + s["bid_r"]) / 2
    home = (mid >= 0.5).to_numpy()
    s["price"] = np.where(home, mid, 1 - mid)
    s["won"] = np.where(home, s["y"], 1 - s["y"])
    s["ask_f"] = np.where(home, s["ask_r"], 1 - s["bid_r"])
    s["fill_f"] = np.where(home, s["fill_ask_r"], 1 - s["fill_bid_r"])
    s["fill_mid_f"] = np.where(home, s["fill_mid_r"], 1 - s["fill_mid_r"])
    return s


def poisson_ci(k, conf=0.95):
    """Exact interval for a Poisson mean given k events (Garwood), by
    bisection on the Poisson CDF; no scipy."""
    def cdf(n, lam):
        term = total = math.exp(-lam)
        for i in range(1, n + 1):
            term *= lam / i
            total += term
        return total

    def solve(f, lo, hi):
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        return (lo + hi) / 2
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else solve(lambda lam: a - (1 - cdf(k - 1, lam)), 0.0, 10.0 * k + 10)
    hi = solve(lambda lam: cdf(k, lam) - a, 0.0, 10.0 * k + 20)
    return lo, hi


def poisson_p_low(k, lam):
    """One-sided p-value: P(X <= k) for X ~ Poisson(lam)."""
    term = total = math.exp(-lam)
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return total


def entry_losses(S, lo, hi, tau_max=0.25):
    """The first live minute in each game with tau <= tau_max whose raw
    favourite mid is in (lo, hi]: games, the favourite's expected losses
    (sum of 1 - price) and its actual losses."""
    s = fav_frame(S, tau_max)
    t = s[(s["price"] > lo) & (s["price"] <= hi)].sort_values("ts").groupby("game_id").head(1)
    return len(t), float((1 - t["price"]).sum()), int((1 - t["won"]).sum())


def longshot_lines(S):
    """Favourite-longshot bias late in games: the favourite's win rate
    against its price by price band (minutes, clustered by game), and one
    trade per game per band, buying the favourite at the first 4th-quarter
    minute its ask is in the band, filled at the worst quote (1 + LAG) min
    later and held to settlement, beside its null (the fill minute's mid)."""
    out = []
    for tau_max, name in ((0.25, "4th quarter"), (5 / 48, "last 5 minutes")):
        s = fav_frame(S, tau_max)
        out.append(f"  {name}: favourite won vs its price, by price (minutes; ± clustered by game)")
        for lo, hi in LONGSHOT_BANDS:
            parts = []
            for lab, x in (("all", s), ("Oct-Jan", s[s["split"] == "train"]),
                           ("Feb-Apr", s[s["split"] == "test"])):
                x = x[(x["price"] > lo) & (x["price"] <= hi)]
                if len(x) < 50:
                    parts.append(f"{lab} n/a")
                    continue
                m, ci = cluster_mean_ci(x["won"] - x["price"], x["game_id"])
                parts.append(f"{lab} {100 * m:+5.2f} pp ± {100 * ci:.2f} ({x['game_id'].nunique()} g)")
            out.append(f"    {lo:.3f}-{hi:.3f}  " + "   ".join(parts))
    out.append("  The same as counts, one entry per game (the first 4th-quarter minute the favourite's")
    out.append("  mid is in the band): favourite losses, expected from its price vs actual, exact")
    out.append("  Poisson 95% for actual / expected (the minute-level ± understates rare losses):")
    for lo, hi in LONGSHOT_BANDS[2:] + ((0.93, 0.995),):
        parts = []
        for lab, x in (("all", S), ("Oct-Jan", S[S["split"] == "train"]),
                       ("Feb-Apr", S[S["split"] == "test"])):
            n, exp_, k = entry_losses(x, lo, hi)
            if n < 30:
                parts.append(f"{lab} n/a")
                continue
            plo, phi = poisson_ci(k)
            parts.append(f"{lab} {n} g: expected {exp_:.1f} lost {k} (ratio {k / exp_:.2f}, "
                         f"{plo / exp_:.2f}-{phi / exp_:.2f})")
        out.append(f"    {lo:.3f}-{hi:.3f}  " + "   ".join(parts))
    s = fav_frame(S, 0.25)
    out.append("  Trade: buy the favourite at the first 4th-quarter minute its ask is in the band, "
               "worst quote")
    out.append(f"  {1 + LAG_CANDLES} min later, hold to settlement (per $1 contract; null = the "
               "fill minute's mid less price and fee):")
    for lo, hi in ((0.70, 0.85), (0.85, 0.93), (0.93, 0.97), (0.97, 0.99), (0.93, 0.99)):
        parts = []
        for lab, x in (("all", s), ("Oct-Jan", s[s["split"] == "train"]),
                       ("Feb-Apr", s[s["split"] == "test"])):
            r = fav_trades(x, lo, hi)
            if r is None:
                parts.append(f"{lab} n/a")
                continue
            parts.append(f"{lab} n {r['n']} pnl {100 * r['pnl'][0]:+.2f}c ± {100 * r['pnl'][1]:.2f} "
                         f"(null {100 * r['null']:+.2f}c; pnl - null {100 * r['edge'][0]:+.2f} ± "
                         f"{100 * r['edge'][1]:.2f})")
        out.append(f"    ask {lo:.2f}-{hi:.2f}  " + "   ".join(parts))
    return out


def fav_trades(s, lo, hi):
    """One trade per game: buy the favourite at the first minute of `s` (a
    fav_frame) its ask is in (lo, hi], worst quote (1 + LAG) min later, held
    to settlement. {n, pnl: (mean, ±95), null, edge: pnl - null (mean, ±95)}
    or None under 30 trades."""
    x = s[(s["ask_f"] > lo) & (s["ask_f"] <= hi)]
    x = x[x["fill_f"].between(0, 1, inclusive="neither") & x["fill_mid_f"].notna()]
    t = x.sort_values("ts").groupby("game_id").head(1)
    if len(t) < 30:
        return None
    f = fee(t["fill_f"])
    pnl = t["won"] - t["fill_f"] - f
    null = t["fill_mid_f"] - t["fill_f"] - f
    return dict(n=len(t), pnl=mr.mean_ci(pnl), null=float(null.mean()),
                edge=mr.mean_ci(pnl - null))


# --------------------------------------------------------------- report ----
def report(S, P, drops, espn, kal, n_games):
    L = []
    w = L.append
    tr, te = P[P["split"] == "train"], P[P["split"] == "test"]
    w("In-game Kalshi: strength prior vs market reaction -- 2025-26, reconstructed model "
      "rows (HINDSIGHT model; Kalshi prices are real). ± is 95%, game-clustered.")
    w(f"p* = Phi(delta / (sigma sqrt(tau)) + z0 sqrt(tau)); train {tr['slate_date'].min()} to "
      f"{TRAIN_END} (sigma, w fitted here), test to {te['slate_date'].max()}.")
    w("No native row has a model P yet: nothing here is forward evidence.")
    w("")
    w("== 0. Data")
    w(f"  reconstructed 2025-26 games with a Kalshi event: {n_games}; used {len(P)} "
      f"(train {len(tr)}, test {len(te)}); dropped: "
      + (", ".join(f"{k} {v}" for k, v in sorted(drops.items())) or "none"))
    w(f"  live states (regulation minutes with a candle): {len(S):,}; with a mid "
      f"{int(S['mid'].notna().sum()):,}; median per game {int(P['n_states'].median())}")
    w(f"  first play after the scheduled tip: median {P['tip_delay_min'].median():.1f} min; "
      f"pregame quote {P['pre_quote_min'].median():.1f} min before it (median)")
    w(f"  live bid-ask spread: median {100 * P['live_spread'].median():.1f}c per game")
    agree = P.dropna(subset=["kalshi_close_q_home"])
    dq = (agree["p_close"] - agree["kalshi_close_q_home"]).abs()
    w(f"  pregame mid vs the benchmark's Kalshi close (both markets normalised, at the "
      f"scheduled tip): mean |diff| {100 * dq.mean():.2f} pp, "
      f"corr {np.corrcoef(agree['p_close'], agree['kalshi_close_q_home'])[0, 1]:.4f} "
      f"(n {len(agree)})")
    w(f"  ESPN final agrees with the ledger result: {int(P['espn_result_agrees'].sum())} of "
      f"{len(P)}; stray ESPN wallclock stamps replaced: {int(P['stray_stamps'].sum())} in "
      f"{int((P['stray_stamps'] > 0).sum())} games")
    w("")

    sig, sig_ci, n_sig = sigma_process(espn, tr)
    sd_sp, sd_sp_ci, q_corr, n_sp = spread_check(espn, P)
    wb, wb_ci = blend_w(tr)
    wu, wu_ci = blend_w(tr, -5.0, 5.0)
    for df in (S, P):
        df["z_blend"] = wb * df["z_model"] + (1 - wb) * df["z_mkt"]
    tr, te = P[P["split"] == "train"], P[P["split"] == "test"]
    S_tr, S_te = S[S["split"] == "train"].copy(), S[S["split"] == "test"].copy()
    fits = {lab: sigma_fits(x, "z_mkt") for lab, x in (("train", S_tr), ("test", S_te))}
    so_mkt = fits["train"]["all minutes"][0]
    so_mod = sigma_fits(S_tr, "z_model")["all minutes"]
    w("== 1. Sigma (points per 48 min) and the blend weight")
    w(f"  score process, train (quarter changes, no outcomes or prices): {sig:.2f} ± "
      f"{sig_ci:.2f} ({n_sig} games) -- the sigma every p* below uses (the notebook's choice).")
    w(f"  Yardstick, all games: SD of the regulation margin around the closing spread "
      f"{sd_sp:.2f} ± {sd_sp_ci:.2f} ({n_sp} games); consecutive quarters' margins net of "
      f"the spread correlate {q_corr:+.3f} (negative = leads shrink, which lifts the "
      f"within-game estimate above the game's SD).")
    w("  Fitted to the outcome and to the live mid, MKT prior (same minutes, same p* form;")
    w("  mid < outcome = the mid moves more per point than outcomes justify, over-reaction):")
    for lab, f in fits.items():
        for part, v in f.items():
            w(f"    {lab:<5} {part:<12} outcome {sigma_text(v[:2])}   mid {sigma_text(v[2:4])}"
              f"   mid - outcome {v[4]:+5.2f} ± {v[5]:4.2f}")
    w("  The p* form fixes the prior's weight at sqrt(tau), so a prior that is itself too")
    w("  confident or too timid (section 2, pregame slopes) moves these fits too: a gap")
    w("  that is largest in the 1st half, where the prior weighs most, is not clean")
    w("  evidence about the reaction to the score. Section 1b frees the prior's weight;")
    w("  section 3 is the test that counts.")
    w(f"  outcome fit, MODEL prior, train: {sigma_text(so_mod[:2])}")
    w(f"  blend w(model) on [0, 1], train: {wb:.2f} ± {wb_ci:.2f};  unbounded: "
      f"{wu:+.2f} ± {wu_ci:.2f}  (0 = the pregame mid already holds the model)")
    w("")

    free = {lab: free_fits(x, "z_mkt") for lab, x in (("train", S_tr), ("test", S_te))}
    w("== 1b. The prior's weight set free: p* = Phi(delta / (sigma sqrt(tau)) + k z0 sqrt(tau)),")
    w("  MKT prior (k = 1 is the notebook's form; k > 1 = the pregame view counts for more than")
    w("  a Brownian with its drift says). Fitted to the outcome and to the live mid on the same")
    w("  minutes (a probit on delta / sqrt(tau) and z0 sqrt(tau)); ±95 game-clustered, the")
    w("  difference from the two fits' joint covariance. mid - outcome sigma > 0 = the mid")
    w("  moves less per point than outcomes justify (under-reaction); < 0 over-reaction.")
    for lab, f in free.items():
        for part, v in f.items():
            w(f"    {lab:<5} {part:<12} outcome sigma {v['sig_y'][0]:5.2f} ± {v['sig_y'][1]:4.2f}"
              f" k {v['k_y'][0]:.2f} ± {v['k_y'][1]:.2f}   mid sigma {v['sig_m'][0]:5.2f} ± "
              f"{v['sig_m'][1]:4.2f} k {v['k_m'][0]:.2f} ± {v['k_m'][1]:.2f}   mid - outcome "
              f"sigma {v['d_sig'][0]:+5.2f} ± {v['d_sig'][1]:4.2f} k {v['d_k'][0]:+.2f} ± "
              f"{v['d_k'][1]:.2f}")
    w("")

    y = te["y"].to_numpy(float)
    pb = Phi(te["z_blend"])
    ll_mkt = ll(te["p_close"], y)
    w(f"== 2. Pregame, test ({len(te)} games)")
    for name, p in (("MKT (Kalshi pregame mid)", te["p_close"]), ("MODEL (v6)", te["p_model"]),
                    ("BLEND", pb)):
        w(f"  {name:<26} log loss {ll(p, y).mean():.4f}   Brier {((p - y) ** 2).mean():.4f}")
    for name, p in (("MODEL - MKT", te["p_model"]), ("BLEND - MKT", pb)):
        m, ci = mr.mean_ci(ll(p, y) - ll_mkt)
        mb, cib = mr.mean_ci((p - y) ** 2 - (te["p_close"] - y) ** 2)
        w(f"  {name:<26} log loss {m:+.4f} ± {ci:.4f}   Brier {mb:+.4f} ± {cib:.4f}")
    lq = mr.logit(te["p_close"])
    b, ci = offset_fit((mr.logit(te["p_model"]) - lq)[:, None], y, lq, np.ones(len(te)),
                       te["game_id"])
    w(f"  outcome's weight on the model's disagreement with the pregame mid: "
      f"{b[0]:+.2f} ± {ci[0]:.2f}")
    w("  Calibration of the pregame mid (compare the live mid's slopes in section 3; the")
    w("  test games are Feb-Apr, when outcomes run steeper than prices in hindsight):")
    for lab, x in (("train", tr), ("test", te)):
        X = np.c_[np.ones(len(x)), mr.logit(x["p_close"])]
        b, ci = offset_fit(X, x["y"], np.zeros(len(x)), np.ones(len(x)), x["game_id"])
        w(f"    {lab:<5} slope {b[1]:.2f} ± {ci[1]:.2f}   intercept {b[0]:+.3f} ± {ci[0]:.3f}"
          f"   games {len(x)}")
    w("")

    for z, _ in PRIORS:
        for df in (S_tr, S_te):
            df[f"p_{z}"] = p_star(df["delta"], df["tau"], df[z], sig)
    for df in (S_tr, S_te):
        df["p_z_mkt_out"] = p_star(df["delta"], df["tau"], df["z_mkt"], so_mkt)
        df["p_z_mkt_fk"] = p_free(df, free["train"])
        df["p_z_mkt_fkp"] = p_free(df, free["train"], by_part=True)
    beyond = {lab: score_beyond_mid(x) for lab, x in (("train", S_tr), ("test", S_te))}
    for df in (S_tr, S_te):
        df["p_mid_adj"] = p_mid_adj(df, beyond["train"])
        df["p_mid_adjp"] = p_mid_adj(df, beyond["train"], by_part=True)
        df["p_mid_a"] = p_mid_adj(df, beyond["train"], score_only=True)
        df["p_mid_ap"] = p_mid_adj(df, beyond["train"], by_part=True, score_only=True)
    w(f"== 3. In-game, test ({S_te['game_id'].nunique()} games; negative = the first beats "
      f"the second; each game weighted equally)")
    for z, name in PRIORS:
        w(diff_line(f"p*({name}) - live mid", S_te, f"p_{z}", "mid"))
    w(diff_line("p*(MKT, outcome sigma) - live mid", S_te, "p_z_mkt_out", "mid"))
    w(diff_line("p*(MKT, sigma and k) - live mid", S_te, "p_z_mkt_fk", "mid"))
    w(diff_line("p*(MKT, sigma and k by part) - mid", S_te, "p_z_mkt_fkp", "mid"))
    w(diff_line("p*(MKT, sigma and k) - p*(MKT, out sig)", S_te, "p_z_mkt_fk", "p_z_mkt_out"))
    w("  (sigma and k: the train outcome fit of section 1b, all minutes or each part.)")
    w(diff_line("p*(BLEND) - p*(MKT)", S_te, "p_z_blend", "p_z_mkt"))
    w(diff_line("p*(MODEL) - p*(MKT)", S_te, "p_z_model", "p_z_mkt"))
    w("  p*(MKT) - live mid, by part of the game:")
    for lo, hi, name in TAU_BUCKETS:
        x = S_te[(S_te["tau"] > lo) & (S_te["tau"] <= hi)]
        w("  " + diff_line(name, x, "p_z_mkt", "mid"))
    w("  " + diff_line("quiet minutes only", S_te[S_te["quiet"]], "p_z_mkt", "mid"))
    w("  Outcome's weight on p*'s disagreement with the live mid (0 = adds nothing):")
    for z, name in PRIORS:
        w("  " + weight_line(f"p*({name})", S_te, f"p_{z}"))
    w("  " + weight_line("p*(MKT, outcome sigma)", S_te, "p_z_mkt_out"))
    w("  " + weight_line("p*(MKT, sigma and k)", S_te, "p_z_mkt_fk"))
    w("  " + weight_line("p*(MKT, sigma and k by part)", S_te, "p_z_mkt_fkp"))
    w("  The score and the prior beyond the live mid, no p* form: y ~ logit(mid) [offset] +")
    w("  a delta / sqrt(tau) + b z0 sqrt(tau) (a > 0 = the mid under-weights the score, < 0")
    w("  over-weights it; b likewise for the pregame view):")
    for lab, f in beyond.items():
        for name, (b, ci, n) in f.items():
            w(f"    {lab:<5} {name:<12} a {b[0]:+.4f} ± {ci[0]:.4f}   b {b[1]:+.3f} ± {ci[1]:.3f}"
              f"   games {n}")
    w("  Scored out of sample, the mid corrected by the train fit's (a, b):")
    w("  " + diff_line("mid + score - live mid", S_te, "p_mid_adj", "mid"))
    w("  " + diff_line("mid + score by part - live mid", S_te, "p_mid_adjp", "mid"))
    w("  " + diff_line("mid + a only - live mid", S_te, "p_mid_a", "mid"))
    w("  " + diff_line("mid + a only by part - live mid", S_te, "p_mid_ap", "mid"))
    w("  Calibration of the live mid against the outcome (slope 1 = right; < 1 the mid is")
    w("  too confident, > 1 too timid):")
    for lab, x in (("train", S_tr), ("test", S_te)):
        w("  " + slope_line(f"{lab}, all minutes", x.dropna(subset=["mid"]), "mid"))
        for lo, hi, name in TAU_BUCKETS:
            xx = x[(x["tau"] > lo) & (x["tau"] <= hi)].dropna(subset=["mid"])
            w("  " + slope_line(f"{lab}, {name}", xx, "mid"))
    w("  Timing (test): ESPN wallclock vs Kalshi mid, corr of minute changes:")
    L += timing_lines(S_te)
    w("")

    trades = []
    for lab, x in (("test", S_te), ("train", S_tr)):
        w(f"== 4. Trades, {lab} (one per game per prior; first quiet minute, tau {TAU_MIN}-"
          f"{TAU_MAX}, p* - ask - fee >= {100 * ENTRY_EDGE:.0f}c; filled at the worst quote "
          f"{1 + LAG_CANDLES} min later; per $1 contract)")
        if lab == "train":
            w("  (sigma and w were fitted on these games: MKT and MODEL use no fitted "
              "outcome here; BLEND's w, MKT out-sig, sigma-and-k and MID+score do.)")
        for z, name in PRIORS + (("z_mkt_out", "MKT out-sig"), ("z_mkt_fk", "MKT sig,k"),
                                 ("z_mkt_fkp", "MKT sig,k part"), ("mid_adj", "MID+score"),
                                 ("mid_adjp", "MID+score part"), ("mid_a", "MID+a"),
                                 ("mid_ap", "MID+a part")):
            t = run_trades(x, f"p_{z}", name)
            trades.append(t)
            w(trade_line(name, t))
            if lab == "test" and len(t) >= 20:
                L += edge_rows(t)
        w("  null = the same contract at the fill minute's mid (what a fair mid says it is "
          "worth) less price and fee; fav = the live favourite bought at the same minute.")
        w("")
    T = pd.concat([t for t in trades if len(t)], ignore_index=True)

    w("== 5. The notebook's decision rules (test)")
    pre_ok = {n: ll(p, y).mean() < ll_mkt.mean() for n, p in
              (("MODEL", te["p_model"]), ("BLEND", pb))}
    (bm, bci), _, _ = per_game_diff(S_te, "p_z_blend", "p_z_mkt")
    tt = T[T["split"] == "test"]

    def ci_lo(prior):
        x = tt[tt["prior"] == prior]
        if len(x) < 20:
            return float("nan"), float("nan")
        m, ci = mr.mean_ci(x["pnl"])
        return m - ci, m
    w(f"  pregame : MODEL or BLEND log loss < MKT -> MODEL {'yes' if pre_ok['MODEL'] else 'no'}, "
      f"BLEND {'yes' if pre_ok['BLEND'] else 'no'}")
    w(f"  in-game : BLEND vs MKT < 0 with the interval below 0 -> {bm:+.4f} ± {bci:.4f}: "
      f"{'pass' if bm + bci < 0 else 'fail'}")
    mkt_lo, mkt_m = ci_lo("MKT")
    for prior in ("MODEL", "BLEND"):
        lo, m = ci_lo(prior)
        ok = lo > 0 and m > mkt_m
        w(f"  tradable: {prior} pnl lower bound > 0 and above MKT -> lower bound "
          f"{100 * lo:+.1f}c, mean {100 * m:+.1f}c vs MKT {100 * mkt_m:+.1f}c: "
          f"{'pass' if ok else 'fail'}")
    w(f"  reaction: MKT-prior pnl lower bound > 0 -> {100 * mkt_lo:+.1f}c: "
      f"{'pass' if mkt_lo > 0 else 'fail'}")
    for prior in ("MKT sig,k", "MKT sig,k part", "MID+score", "MID+score part", "MID+a",
                  "MID+a part"):
        lo, m = ci_lo(prior)
        if lo != lo:
            w(f"  reaction ({prior}): too few trades to judge")
            continue
        w(f"  reaction ({prior}): pnl lower bound > 0 -> {100 * lo:+.1f}c "
          f"(mean {100 * m:+.1f}c): {'pass' if lo > 0 else 'fail'}")
    w("")
    w("== 6. Live fluctuations vs the spread and time remaining (owner's question)")
    w("  Each live minute: the mid's move against the move the score and clock justify")
    w("  (p*(MKT, sigma and k), the train outcome fit) and against the bid-ask spread. Per cent")
    w("  of score move: how far the mid follows it in the same minute, the next, and minutes")
    w("  2-6 after (0 if the mid is a fair price by then, whatever p*'s scale: > 0 the mid")
    w("  keeps drifting the score's way, under-reaction; < 0 it gives the move back,")
    w("  over-reaction). Noise SD: the bounce in the mid when the margin does not move")
    w("  (minus the covariance of consecutive moves). ±95 game-clustered.")
    for lab, x in (("test", S_te), ("train", S_tr)):
        w(f"  -- {lab}")
        L += fluctuation_lines(x, "p_z_mkt_fk")
    w("")
    w("== 7. Both sides at tip, each with a resting sell at entry + up (owner's question)")
    w("  Per $1 pair: home YES bought at the last pregame ask, the away side at 1 - the home")
    w("  bid, taker fee on both; a side's sell fills when its bid reaches the target (candle")
    w("  closes, through settlement); an unsold side settles at 0 or 1. Hold both = the")
    w("  pair held to settlement (always pays $1): the pair's cost. If prices are fair the")
    w("  exit rule cannot change the expected result, so profit - hold should be ~0 (less")
    w(f"  the maker fee, {MAKER_FEE} P (1 - P), where it applies). ± is 95% across games.")
    L += straddle_lines(P, kal)
    w("")
    w("== 8. Signals for the eventual winner at the +10c fill (owner's question; exploratory)")
    w("  Each fill of section 7's +10c sells: does the side win more often than its mid at the")
    w("  fill minute says, and does anything seen then add to the price? A resting sell cannot")
    w("  use a signal; a rule that could (hold when it says so) would need it to beat the")
    w("  price, then the spread and fee. Several looks at the same games: a lead, not a rule.")
    ev, paths = trigger_frames(P, kal, pd.concat([S_tr, S_te], ignore_index=True))
    L += trigger_lines(ev, paths)
    w("")
    w("== 9. The quarter-by-quarter price path and the eventual winner (owner's question;")
    w("  exploratory). Winners' prices rise by construction, so the question is whether the")
    w("  path so far predicts the winner beyond the price at the end of Q1, Q2 and Q3: beta 0")
    w("  = the path adds nothing (moves in cents of the home mid; choppiness = SD of its")
    w("  minute moves from tip, times logit(price): beta < 0 = in choppy games the price is")
    w("  too confident). ± is 95%, one row per game.")
    L += shape_lines(quarter_frame(pd.concat([S_tr, S_te], ignore_index=True), P))
    w("")
    w("== 10. Favourite-longshot bias late in games (raw quotes, the extremes included)")
    w("  Prediction markets classically overprice longshots; the mid's under-weighting of")
    w("  a late lead (section 3, a > 0 in the 4th quarter) is that bias seen from the")
    w("  favourite's side. won - price > 0 = the favourite is underpriced.")
    L += longshot_lines(pd.concat([S_tr, S_te], ignore_index=True))
    return "\n".join(L), T


# -------------------------------------------------------------- holdout ----
HOLDOUT_RANGES = (("2025-04-14", "2025-06-25"), ("2026-04-13", "2026-06-25"))
HOLDOUT_TYPES = (3, 5)            # ESPN season types: postseason, play-in


def holdout_games():
    """Every completed 2025 and 2026 play-in and playoff game from ESPN's
    scoreboards (cached), in the shape game_states reads. The model is not
    used (p_home is a placeholder)."""
    path = CACHE / "holdout_games.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"game_id": str})
    rows = []
    for lo, hi in HOLDOUT_RANGES:
        for d in pd.date_range(lo, hi).strftime("%Y-%m-%d"):
            for g in market.scoreboard(d):
                if g["season_type"] in HOLDOUT_TYPES and g["completed"] \
                        and g["home_pts"] is not None and g["away_pts"] is not None:
                    rows.append(dict(game_id=g["game_id"], slate_date=d, tip_utc=g["tip_utc"],
                                     home=g["home"], away=g["away"],
                                     home_won=int(g["home_pts"] > g["away_pts"]),
                                     season_type=g["season_type"]))
    r = pd.DataFrame(rows).drop_duplicates("game_id")
    r = r.assign(p_home=0.5, kalshi_close_q_home=np.nan, close_book="none", close_spread=np.nan)
    path.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(path, index=False)
    return r


def holdout_kalshi(row, start, end):
    """The home market's candles, trying the event ticker away+home, then
    home+away (cached under the game id like kalshi_game)."""
    path = CACHE / "kalshi" / f"{row.game_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    res = {"ticker": None, "candles": []}
    for h_, a_ in ((row.home, row.away), (row.away, row.home)):
        t = kalshi.market_ticker(row.slate_date, row.home, h_, a_)
        cs = kb.candles(t, start, end, 1)
        if cs:
            res = {"ticker": t, "candles": [
                [c.get("end_period_ts"), px(c.get("yes_ask"), "close"), px(c.get("yes_ask"), "high"),
                 px(c.get("yes_bid"), "close"), px(c.get("yes_bid"), "low"),
                 float(c.get("volume") or 0)] for c in cs]}
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    return res


def holdout_report(games, S, P, drops):
    """The three pre-registered tests (PREREG_holdout.md), exactly as fixed."""
    L = []
    w = L.append
    S = S.assign(split=np.where(S["game_id"].map(P.set_index("game_id")["slate_date"]) < "2026-01-01",
                                "2025", "2026"))
    w("Pre-registered holdout (diagnostics/strength_vs_reaction/PREREG_holdout.md): late-game")
    w("Kalshi pricing on the 2025 and 2026 play-in and playoff games. Tests as fixed before")
    w("this data was read; nothing tuned. ± is 95%.")
    w("")
    w(f"Games: {len(games)} completed play-in/playoff games on ESPN; used {len(P)} "
      f"(2025 {int((P['slate_date'] < '2026-01-01').sum())}, 2026 "
      f"{int((P['slate_date'] >= '2026-01-01').sum())}); dropped: "
      + (", ".join(f"{k} {v}" for k, v in sorted(drops.items())) or "none"))
    w(f"ESPN final agrees with the scoreboard result: {int(P['espn_result_agrees'].sum())} of {len(P)}")
    w("")
    n, lam, k = entry_losses(S, 0.93, 0.995)
    plo, phi = poisson_ci(k)
    p = poisson_p_low(k, lam)
    w(f"H-L1 favourite-longshot bias (primary): {n} games with a 4th-quarter favourite at "
      f"93-99.5c; expected losses {lam:.2f}, actual {k}; ratio {k / lam:.2f} "
      f"(exact 95% {plo / lam:.2f}-{phi / lam:.2f}); one-sided p = {p:.3f}: "
      f"{'PASS' if p < 0.05 else 'not passed'}  (regular season: 27 vs 41.0, ratio 0.66)")
    for season in ("2025", "2026"):
        n_, lam_, k_ = entry_losses(S[S["split"] == season], 0.93, 0.995)
        w(f"    {season}: {n_} games, expected {lam_:.2f}, lost {k_}")
    b, ci, ng = score_beyond_mid(S)["4th quarter"]
    z = b[0] / (ci[0] / Z95)
    w(f"H-L2 score beyond the mid, 4th quarter: a {b[0]:+.4f} ± {ci[0]:.4f} ({ng} games), z "
      f"{z:+.2f}: {'PASS' if z > 1.645 else 'not passed'}  (regular season +0.021 / +0.022)")
    r = fav_trades(fav_frame(S, 0.25), 0.93, 0.99)
    if r is None:
        w("H-L3 the trade: fewer than 30 trades; not assessable")
    else:
        z3 = r["edge"][0] / (r["edge"][1] / Z95)
        w(f"H-L3 the trade: {r['n']} trades; pnl {100 * r['pnl'][0]:+.2f}c ± {100 * r['pnl'][1]:.2f}, "
          f"null {100 * r['null']:+.2f}c, pnl - null {100 * r['edge'][0]:+.2f}c ± "
          f"{100 * r['edge'][1]:.2f} (z {z3:+.2f}): {'PASS' if z3 > 1.645 else 'not passed'}  "
          f"(regular season pnl -0.36c, pnl - null +1.48c)")
    w("")
    w("Power was stated in advance: only a large effect can pass here, and a miss is not a")
    w("refutation. The decisive test is the 2026-27 regular season from Kalshi's archive.")
    return "\n".join(L)


def holdout_main(workers):
    games = holdout_games()
    print(f"{len(games)} holdout games", flush=True)
    espn = {}
    with ThreadPoolExecutor(workers) as ex:
        for gid, js in zip(games["game_id"], ex.map(
                lambda g: _safe(espn_game, g), games["game_id"])):
            espn[gid] = js
    kal = {}
    for row in games.itertuples():
        js = espn.get(row.game_id)
        if js and js["plays"]:
            ts = [p[0] for p in js["plays"]]
            kal[row.game_id] = _safe(holdout_kalshi, row, min(ts) - PRE_WINDOW,
                                     max(ts) + POST_WINDOW)
    S, P, drops = assemble(games, espn, kal)
    text = holdout_report(games, S, P, drops)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "HOLDOUT.txt").write_text(text + "\n")
    print(text)


def _safe(f, *args):
    try:
        return f(*args)
    except Exception as e:  # noqa: BLE001 - one game; keep going
        print(f"{f.__name__} {args[0] if not hasattr(args[0], 'game_id') else args[0].game_id}: "
              f"{e!r}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--holdout", action="store_true",
                    help="the pre-registered play-in/playoff holdout (PREREG_holdout.md)")
    a = ap.parse_args()
    if a.holdout:
        holdout_main(a.workers)
        return
    games = load_games(a.limit)
    print(f"{len(games)} games", flush=True)
    espn, kal = fetch(games, a.workers)
    if a.fetch_only:
        return
    S, P, drops = assemble(games, espn, kal)
    text, T = report(S, P, drops, espn, kal, len(games))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.txt").write_text(text + "\n")
    P.drop(columns=["z_mkt", "z_model", "z_blend"]).round(5).to_csv(OUT / "games.csv",
                                                                     index=False)
    T.round(5).to_csv(OUT / "trades.csv", index=False)
    print(text)


if __name__ == "__main__":
    main()
