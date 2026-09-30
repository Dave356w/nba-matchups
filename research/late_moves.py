#!/usr/bin/env python3
"""Flag games where the market moved against the model's lean, for lineup review.

  python research/late_moves.py [--threshold 4]

Reconstructed rows (HINDSIGHT) with both an opening and a closing moneyline.
A game is flagged when the lean side's no-vig probability fell by at least
--threshold points from open to close. The move is known only at the close:
this is a list for manual review of lineup/rest news, never a model input.

Priority: A = lean lost and the market moved 8+ pts; B = lean lost; C = lean
won (a control). Writes research/lineup_review/late_moves.csv.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

import analysis  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--threshold', type=float, default=4.0)
args = ap.parse_args()
r = analysis.with_close(ledger.graded(ledger.load(ledger.RECON_PATH))).copy()
for c in ["p_home","close_q_home","home_won","open_home_ml","open_away_ml","close_home_ml","close_away_ml","home_pts","away_pts","close_spread"]:
    r[c] = pd.to_numeric(r[c], errors="coerce")
r = r[r.open_home_ml.notna() & r.open_away_ml.notna()]
r["open_q"] = [market.devig(h, a) for h, a in zip(r.open_home_ml, r.open_away_ml)]
lh = r.p_home >= .5
y = r.home_won.to_numpy(float)
r["excess"] = market.logloss(r.p_home.to_numpy(float), y) - market.logloss(r.close_q_home.to_numpy(float), y)
out = pd.DataFrame({
    "slate_date": r.slate_date, "season": r.season.astype(str).radd("20").str[-4:].astype(int).map(lambda s: f"{s-1}-{str(s)[2:]}"),
    "close_book": r.close_book.map(market.BOOK_NAMES),
    "matchup": r.away + " @ " + r.home, "tip_utc": r.tip_utc,
    "model_lean": r.lean, "faded": np.where(lh, r.away, r.home),
    "model_p_lean": (100*np.where(lh, r.p_home, 1-r.p_home)).round(1),
    "open_q_lean": (100*np.where(lh, r.open_q, 1-r.open_q)).round(1),
    "close_q_lean": (100*np.where(lh, r.close_q_home, 1-r.close_q_home)).round(1),
    "open_ml_lean": np.where(lh, r.open_home_ml, r.open_away_ml).astype(int),
    "close_ml_lean": np.where(lh, r.close_home_ml, r.close_away_ml).astype(int),
    "close_spread_home": r.close_spread,
    "final": r.away_pts.astype(int).astype(str) + "-" + r.home_pts.astype(int).astype(str),
    "lean_result": np.where(r.lean_won.astype(float) == 1, "W", "L"),
    "lean_b2b": np.where(lh, r.home_b2b, r.away_b2b).astype(int),
    "faded_b2b": np.where(lh, r.away_b2b, r.home_b2b).astype(int),
    "excess_logloss": r.excess.round(3),
    "espn_game": "https://www.espn.com/nba/game/_/gameId/" + r.game_id.astype(str),
})
out["move_vs_lean_pp"] = (out.close_q_lean - out.open_q_lean).round(1)
flag = out[out.move_vs_lean_pp <= -args.threshold].copy()
flag["flipped_side"] = flag.close_q_lean < 50
flag["priority"] = np.select([(flag.lean_result == "L") & (flag.move_vs_lean_pp <= -8),
                              flag.lean_result == "L"], ["A", "B"], "C")
cols = ["priority","slate_date","season","close_book","matchup","tip_utc","model_lean","faded","model_p_lean",
        "open_q_lean","close_q_lean","move_vs_lean_pp","flipped_side","open_ml_lean","close_ml_lean",
        "close_spread_home","final","lean_result","lean_b2b","faded_b2b","excess_logloss","espn_game"]
flag = flag[cols].sort_values(["priority","move_vs_lean_pp","slate_date"])
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lineup_review", "late_moves.csv")
flag.to_csv(path, index=False)
print(len(out), "games with open+close;", len(flag), "flagged")
print(flag.groupby(["season","close_book"]).agg(n=("matchup","size"), lean_won=("lean_result", lambda x: (x=="W").mean()*100),
      model=("model_p_lean","mean"), open=("open_q_lean","mean"), close=("close_q_lean","mean")).round(1))
print(flag.priority.value_counts().sort_index().to_dict())
print(f"wrote {path}")
