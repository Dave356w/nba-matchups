#!/usr/bin/env python3
"""Box-score projector logic: point-in-time backtest on the repository's own data.

  python research/projector_backtest.py [--seasons 2025 2026 2027]
      -> diagnostics/projector_backtest/RESULTS.txt, predictions_<season>.csv

Owner's request (2026-11-05): test the "NBA Model Logic Backtest" projector
(owner-supplied script). Its data pipeline does not run here: nba_api is not
installed and stats.nba.com is unreachable from this environment. This file
keeps the projector logic as supplied and replaces only the data source:

  * team games: the cached Basketball-Reference team game logs
    (nba_composite.load_logs, the same logs the production model reads),
    converted to the LeagueGameLog row format the script expects
    (GAME_ID, GAME_DATE, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME, MATCHUP, PTS,
    FGM, FGA, FG3M, FTM, FTA, OREB, DREB, TOV); team codes stay BBR codes;
  * odds: the repository's closing lines, one book per game, never pooled:
    reconstructed rows (ESPN BET 2024-25, ESPN BET / DraftKings 2025-26) and
    native DraftKings 2026-27 (data/nba_ledger.csv); totals from the same
    book (diagnostics/totals_signal/totals_odds.csv, ledger).

Projector logic, unchanged: per team, season-to-date / home-or-away split /
last-10 means blended 0.50 / 0.40 / 0.10 over games strictly before the game
(each game joins the histories only after it is predicted); both teams need
5 prior games; home score = (0.55 ORtg_home + 0.45 DRtg_away) / 100 x mean
pace, away score = (0.45 ORtg_away + 0.55 DRtg_home) / 100 x mean pace. No
parameter is fitted, so there is nothing to walk forward.

Readings (fixed before the run), each book x season alone, on the games
the projector predicts:
  1. the script's own summary: winner accuracy, margin and total MAE / RMSE
     and bias, accuracy by predicted-margin bucket;
  2. beside it on the same games: the production margin model (reconstructed
     v11 rows for 2024-25 / 2025-26, hindsight; rebuilt v11 for 2026-27) and
     the closing spread; the totals model T* fitted on the seasons before
     (totals.fit) and the closing total;
  3. against the closing lines, as the script does (every game picked by the
     sign of the edge) and at edges of 3 and 6 points: ATS and O/U win rate
     and ROI at the closing prices, against the no-vig null.
Research only: nothing here touches the ledgers, pages or model/*.json.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import nba_composite as nc                        # noqa: E402
import totals                                     # noqa: E402

OUT = Path(ROOT) / "diagnostics" / "projector_backtest"


# ── Configuration (as supplied) ─────────────────────────────────────────────
@dataclass(frozen=True)
class BacktestConfig:
    season: str | None = None
    season_type: str = "Regular Season"
    min_prior_games: int = 5
    weight_season: float = 0.50
    weight_split: float = 0.40
    weight_last10: float = 0.10
    output_dir: Path = field(default_factory=lambda: OUT)

    @property
    def weights(self) -> tuple[float, float, float]:
        return (self.weight_season, self.weight_split, self.weight_last10)


def validate_config(config: BacktestConfig) -> None:
    weights = config.weights
    if any(w < 0 for w in weights):
        raise ValueError(f"Weights must be non-negative: {weights}")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError(f"Weights must sum to 1.0. Current sum={sum(weights):.4f}")
    if config.min_prior_games < 0:
        raise ValueError("min_prior_games must be non-negative")


# ── New data source: BBR team logs -> LeagueGameLog rows ────────────────────
def league_game_log_from_bbr(season_end_year: int) -> pd.DataFrame:
    """One row per team per game in the LeagueGameLog format, from the cached
    Basketball-Reference logs (regular season)."""
    logs = nc.load_logs(season_end_year)
    names = {}
    try:
        import build_site                          # team full names, if available
        names = dict(build_site.TEAM_NAMES)
    except Exception:                              # noqa: BLE001
        pass
    ids = {t: k + 1 for k, t in enumerate(sorted(logs))}
    rows = []
    for tm, lg in logs.items():
        for r in lg.itertuples(index=False):
            d = pd.Timestamp(r.date)
            home, away = (tm, r.opp) if r.home else (r.opp, tm)
            rows.append({
                "GAME_ID": f"{d:%Y%m%d}{home}{away}",
                "GAME_DATE": d.strftime("%Y-%m-%d"),
                "TEAM_ID": ids[tm],
                "TEAM_ABBREVIATION": tm,
                "TEAM_NAME": names.get(tm, tm),
                "MATCHUP": f"{tm} vs. {r.opp}" if r.home else f"{tm} @ {r.opp}",
                "PTS": r.pts, "FGM": r.TFG, "FGA": r.TFGA, "FG3M": r.T3P,
                "FTM": r.TFT, "FTA": r.TFTA, "OREB": r.TORB, "DREB": r.TDRB, "TOV": r.TTOV,
                "MIN": 240,
            })
    return pd.DataFrame(rows)


def normalize_team_to_abbr(value: Any) -> str | None:
    """BBR codes are kept as they are (the odds and ledgers use them too)."""
    if value is None or pd.isna(value):
        return None
    return str(value).strip().upper()


# ── Feature engineering (as supplied) ───────────────────────────────────────
REQUIRED_GAMELOG_COLUMNS = {
    "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "MATCHUP", "PTS",
    "FGM", "FGA", "FG3M", "FTM", "FTA", "OREB", "DREB", "TOV",
}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_team_minutes(value: Any) -> float:
    if value is None or pd.isna(value):
        return 240.0
    text = str(value).strip()
    if not text:
        return 240.0
    if ":" in text:
        try:
            return float(text.split(":", 1)[0])
        except ValueError:
            return 240.0
    try:
        return float(text)
    except ValueError:
        return 240.0


def prepare_team_game_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_GAMELOG_COLUMNS - set(raw_df.columns)
    if missing:
        raise ValueError(f"Raw game log missing required columns: {sorted(missing)}")
    df = raw_df.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["TEAM_ID"] = _numeric(df["TEAM_ID"]).astype("Int64")
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].map(normalize_team_to_abbr)
    for col in ["PTS", "FGM", "FGA", "FG3M", "FTM", "FTA", "OREB", "DREB", "TOV"]:
        df[col] = _numeric(df[col])
    matchup = df["MATCHUP"].astype(str)
    df["is_home"] = matchup.str.contains("vs", case=False, regex=False)
    df["is_away"] = matchup.str.contains("@", regex=False)
    enriched_rows: list[dict[str, Any]] = []
    for game_id, game_df in df.groupby("GAME_ID", sort=False):
        if len(game_df) != 2:
            continue
        home_df = game_df[game_df["is_home"]]
        away_df = game_df[game_df["is_away"]]
        if len(home_df) != 1 or len(away_df) != 1:
            continue
        rows = [home_df.iloc[0], away_df.iloc[0]]
        opps = [away_df.iloc[0], home_df.iloc[0]]
        for row, opp in zip(rows, opps):
            fga, fta = float(row["FGA"] or 0), float(row["FTA"] or 0)
            oreb, tov = float(row["OREB"] or 0), float(row["TOV"] or 0)
            opp_fga, opp_fta = float(opp["FGA"] or 0), float(opp["FTA"] or 0)
            opp_oreb, opp_tov = float(opp["OREB"] or 0), float(opp["TOV"] or 0)
            poss = fga + 0.44 * fta - oreb + tov
            opp_poss = opp_fga + 0.44 * opp_fta - opp_oreb + opp_tov
            game_poss = np.nanmean([poss if poss > 0 else np.nan, opp_poss if opp_poss > 0 else np.nan])
            pts, opp_pts = float(row["PTS"]), float(opp["PTS"])
            fgm, fg3m, ftm = float(row["FGM"] or 0), float(row["FG3M"] or 0), float(row["FTM"] or 0)
            dreb = float(row["DREB"] or 0)
            minutes = parse_team_minutes(row.get("MIN", 240))
            pace = game_poss * (240.0 / minutes) if minutes and game_poss and not pd.isna(game_poss) else game_poss
            enriched_rows.append({
                "game_id": str(game_id), "game_date": row["GAME_DATE"],
                "team_id": int(row["TEAM_ID"]), "team_abbr": row["TEAM_ABBREVIATION"],
                "team_name": row["TEAM_NAME"], "opp_team_id": int(opp["TEAM_ID"]),
                "opp_abbr": opp["TEAM_ABBREVIATION"], "opp_team_name": opp["TEAM_NAME"],
                "is_home": bool(row["is_home"]), "actual_points": pts,
                "actual_points_allowed": opp_pts, "actual_margin_team_perspective": pts - opp_pts,
                "points": pts, "points_allowed": opp_pts,
                "off_rating": 100.0 * pts / game_poss if game_poss and game_poss > 0 else np.nan,
                "def_rating": 100.0 * opp_pts / game_poss if game_poss and game_poss > 0 else np.nan,
                "pace": pace,
                "efg_pct": (fgm + 0.5 * fg3m) / fga if fga > 0 else np.nan,
                "ts_pct": pts / (2.0 * (fga + 0.44 * fta)) if (fga + 0.44 * fta) > 0 else np.nan,
                "ft_pct": ftm / fta if fta > 0 else np.nan,
                "dreb": dreb, "opp_tov": opp_tov, "three_pt_points": fg3m * 3.0,
                "two_pt_points": max(fgm - fg3m, 0.0) * 2.0, "raw_possessions": poss,
                "opp_raw_possessions": opp_poss, "game_possessions": game_poss,
            })
    out = pd.DataFrame(enriched_rows)
    return out.sort_values(["game_date", "game_id", "is_home"],
                           ascending=[True, True, False]).reset_index(drop=True)


MODEL_METRIC_COLUMNS = ["points", "points_allowed", "off_rating", "def_rating", "pace", "efg_pct",
                        "ts_pct", "ft_pct", "dreb", "opp_tov", "three_pt_points", "two_pt_points"]


def weighted_average(parts: Iterable[tuple[float, float | None]]) -> float | None:
    valid = [(w, float(v)) for w, v in parts if v is not None and not pd.isna(v)]
    if not valid:
        return None
    total_weight = sum(w for w, _ in valid)
    return sum(w * v for w, v in valid) / total_weight if total_weight else None


def mean_or_none(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df.columns:
        return None
    value = pd.to_numeric(df[col], errors="coerce").mean()
    return None if pd.isna(value) else float(value)


def metrics_from_history(history, *, is_home: bool, config: BacktestConfig):
    hist_df = pd.DataFrame(history).sort_values(["game_date", "game_id"])
    split_df = hist_df[hist_df["is_home"] == is_home]
    last10_df = hist_df.tail(10)
    ws, whr, wl10 = config.weights
    return {col: weighted_average([(ws, mean_or_none(hist_df, col)),
                                   (whr, mean_or_none(split_df, col)),
                                   (wl10, mean_or_none(last10_df, col))])
            for col in MODEL_METRIC_COLUMNS}


def project_full_game(home, away) -> tuple[float, float]:
    """The supplied full-game efficiency logic, unchanged."""
    avg_pace = ((home.get("pace") or 100.0) + (away.get("pace") or 100.0)) / 2.0
    home_score = (((0.55 * (home.get("off_rating") or 0.0)) + (0.45 * (away.get("def_rating") or 0.0))) / 100.0) * avg_pace
    away_score = (((0.45 * (away.get("off_rating") or 0.0)) + (0.55 * (home.get("def_rating") or 0.0))) / 100.0) * avg_pace
    return float(home_score), float(away_score)


def build_game_pairs(team_games: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for game_id, group in team_games.groupby("game_id", sort=False):
        home, away = group[group["is_home"]], group[~group["is_home"]]
        if len(group) != 2 or len(home) != 1 or len(away) != 1:
            continue
        h, a = home.iloc[0], away.iloc[0]
        pairs.append({"game_id": game_id, "game_date": h["game_date"], "home_id": int(h["team_id"]),
                      "away_id": int(a["team_id"]), "home_abbr": h["team_abbr"],
                      "away_abbr": a["team_abbr"], "actual_home_score": float(h["actual_points"]),
                      "actual_away_score": float(a["actual_points"])})
    return pd.DataFrame(pairs).sort_values(["game_date", "game_id"]).reset_index(drop=True)


def run_point_in_time_backtest(team_games: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    game_pairs = build_game_pairs(team_games)
    rows_by_game_team = {(str(r["game_id"]), int(r["team_id"])): r.to_dict()
                         for _, r in team_games.iterrows()}
    team_history: dict[int, list] = defaultdict(list)
    predictions = []
    for _, game in game_pairs.iterrows():
        hid, aid = int(game["home_id"]), int(game["away_id"])
        hh, ah = team_history[hid], team_history[aid]
        if len(hh) >= config.min_prior_games and len(ah) >= config.min_prior_games:
            hm = metrics_from_history(hh, is_home=True, config=config)
            am = metrics_from_history(ah, is_home=False, config=config)
            hp, ap = project_full_game(hm, am)
            ah_s, aa_s = float(game["actual_home_score"]), float(game["actual_away_score"])
            predictions.append({
                "game_id": game["game_id"], "game_date": game["game_date"],
                "matchup": f"{game['away_abbr']} @ {game['home_abbr']}",
                "away_abbr": game["away_abbr"], "home_abbr": game["home_abbr"],
                "prior_games_home": len(hh), "prior_games_away": len(ah),
                "pred_home_score": hp, "pred_away_score": ap,
                "actual_home_score": ah_s, "actual_away_score": aa_s,
                "pred_margin_home": hp - ap, "actual_margin_home": ah_s - aa_s,
                "pred_total": hp + ap, "actual_total": ah_s + aa_s,
                "winner_correct": bool((hp - ap > 0) == (ah_s - aa_s > 0)),
                "margin_error": (hp - ap) - (ah_s - aa_s),
                "abs_margin_error": abs((hp - ap) - (ah_s - aa_s)),
                "total_error": (hp + ap) - (ah_s + aa_s),
                "abs_total_error": abs((hp + ap) - (ah_s + aa_s)),
                "pred_margin_abs": abs(hp - ap),
            })
        gid = str(game["game_id"])
        team_history[hid].append(rows_by_game_team[(gid, hid)])
        team_history[aid].append(rows_by_game_team[(gid, aid)])
    return pd.DataFrame(predictions)


def rmse(values) -> float:
    return float(np.sqrt(np.mean(np.square(pd.to_numeric(values, errors="coerce")))))


# ── Comparisons on the same games ───────────────────────────────────────────
def odds_and_production(season_end_year: int) -> pd.DataFrame:
    """One row per priced game: book, closing spread / total / prices, the
    production margin model's margin and P (hindsight rows)."""
    if season_end_year == 2027:
        src = pd.read_csv(os.path.join(ROOT, "data", "nba_rebuilt.csv"))
        led = pd.read_csv(os.path.join(ROOT, "data", "nba_ledger.csv")).set_index("game_id")
        src["close_book"] = "dk"
        for c in ("close_spread", "close_home_spread_odds", "close_away_spread_odds"):
            src[c] = src[c].fillna(src["game_id"].map(led[c]))
        src["close_total"] = src["game_id"].map(led["close_ou_line"])
        src["close_over"] = src["game_id"].map(led["close_ou_over"])
        src["close_under"] = src["game_id"].map(led["close_ou_under"])
    else:
        src = pd.read_csv(os.path.join(ROOT, "data", "nba_reconstructed.csv"))
        src = src[src["season"] == season_end_year]
        o = pd.read_csv(os.path.join(ROOT, "diagnostics", "totals_signal", "totals_odds.csv"))
        src = src.merge(o[["game_id", "book", "close_total", "close_over", "close_under"]],
                        on="game_id", how="left")
        same = src["book"].isna() | (src["book"] == src["close_book"])
        src.loc[~same, ["close_total", "close_over", "close_under"]] = np.nan
    src["game_date"] = src["slate_date"].astype(str)
    return src[["game_date", "home", "away", "close_book", "close_spread", "close_home_spread_odds",
                "close_away_spread_odds", "close_total", "close_over", "close_under",
                "margin_home", "p_home"]].rename(columns={"home": "home_abbr", "away": "away_abbr",
                                                           "margin_home": "prod_margin",
                                                           "p_home": "prod_p_home"})


def tstar_totals(season_end_year: int) -> pd.DataFrame:
    """T*, the totals model fitted on the game seasons before the season."""
    seasons = [y for y in [2016, 2017, 2018, 2019] + list(range(2021, 2028)) if y <= season_end_year]
    rows = totals.training_rows({y: nc.load_logs(y) for y in seasons})
    fit = totals.fit(rows[rows["season"].isin([y for y in seasons if y < season_end_year and y >= 2017])])
    cur = rows[rows["season"] == season_end_year].copy()
    cur["prod_total"] = [totals.predict(fit, f)[0] for f in cur.to_dict("records")]
    return cur.rename(columns={"date": "game_date", "home": "home_abbr", "away": "away_abbr"})[
        ["game_date", "home_abbr", "away_abbr", "prod_total"]]


def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


def bet_line(pick_home_or_over, win, push, pay, q):
    keep = ~push
    if keep.sum() < 2:
        return f"n {int(keep.sum()):4d}"
    w, p, qq = win[keep], pay[keep], q[keep]
    roi = np.where(w, p - 1, -1.0)
    return (f"n {int(keep.sum()):4d}  won {w.mean() * 100:5.1f}%  ROI {roi.mean() * 100:+6.1f}% ± "
            f"{1.96 * roi.std(ddof=1) / np.sqrt(len(roi)) * 100:4.1f}  null {(qq * p - 1).mean() * 100:+.1f}%")


def evaluate(P: pd.DataFrame, label: str) -> list[str]:
    L = [f"== {label}  games {len(P)}"]
    L.append(f"  projector (as supplied): winner {P['winner_correct'].mean():.3f}  margin MAE "
             f"{P['abs_margin_error'].mean():.2f} RMSE {rmse(P['margin_error']):.2f} bias "
             f"{P['margin_error'].mean():+.2f}  total MAE {P['abs_total_error'].mean():.2f} RMSE "
             f"{rmse(P['total_error']):.2f} bias {P['total_error'].mean():+.2f}  SD pred margin "
             f"{P['pred_margin_home'].std():.2f}")
    m = P.dropna(subset=["close_spread", "prod_margin"])
    if len(m):
        act = m["actual_margin_home"]
        L.append(f"  same {len(m)} games with a closing spread and the production model:")
        for nm, pred in (("projector", m["pred_margin_home"]), ("production v11", m["prod_margin"]),
                         ("closing spread", -m["close_spread"])):
            L.append(f"    {nm:15s} margin RMSE {rmse(pred - act):.2f}  winner "
                     f"{((pred > 0) == (act > 0)).mean():.3f}  bias {(pred - act).mean():+.2f}  "
                     f"SD {pred.std():.2f}")
        L.append(f"    corr(projector, close) {np.corrcoef(m['pred_margin_home'], -m['close_spread'])[0, 1]:.3f}  "
                 f"corr(production, close) {np.corrcoef(m['prod_margin'], -m['close_spread'])[0, 1]:.3f}")
    t = P.dropna(subset=["close_total", "prod_total"])
    if len(t):
        act = t["actual_total"]
        L.append(f"  same {len(t)} games with a closing total and T*:")
        for nm, pred in (("projector", t["pred_total"]), ("T* (production totals)", t["prod_total"]),
                         ("closing total", t["close_total"])):
            L.append(f"    {nm:22s} total RMSE {rmse(pred - act):.2f}  bias {(pred - act).mean():+.2f}  "
                     f"SD {pred.std():.2f}")
    # against the closing lines: every game by the edge's sign (as the script), and at 3 / 6 pts
    s = P.dropna(subset=["close_spread", "close_home_spread_odds", "close_away_spread_odds"])
    if len(s):
        edge = s["pred_margin_home"] + s["close_spread"]
        res = s["actual_margin_home"] + s["close_spread"]
        dh, da = dec(s["close_home_spread_odds"]), dec(s["close_away_spread_odds"])
        qh = (1 / dh) / (1 / dh + 1 / da)
        L.append("  ATS at the closing spread (projector side by the edge's sign):")
        for thr in (0, 3, 6):
            sel = (edge.abs() >= thr).to_numpy() if thr else np.ones(len(s), bool)
            home = (edge > 0).to_numpy()
            win = np.where(home, res > 0, res < 0)
            L.append(f"    |edge| >= {thr}: " + bet_line(home[sel], win[sel], (res == 0).to_numpy()[sel],
                                                       np.where(home, dh, da)[sel],
                                                       np.where(home, qh, 1 - qh)[sel]))
    o = P.dropna(subset=["close_total", "close_over", "close_under"])
    if len(o):
        edge = o["pred_total"] - o["close_total"]
        res = o["actual_total"] - o["close_total"]
        do, du = dec(o["close_over"]), dec(o["close_under"])
        qo = (1 / do) / (1 / do + 1 / du)
        over = (edge > 0).to_numpy()
        L.append(f"  O/U at the closing total (projector side by the edge's sign; overs "
                 f"{over.mean() * 100:.0f}%):")
        for thr in (0, 3, 6):
            sel = (edge.abs() >= thr).to_numpy() if thr else np.ones(len(o), bool)
            win = np.where(over, res > 0, res < 0)
            L.append(f"    |edge| >= {thr}: " + bet_line(over[sel], win[sel], (res == 0).to_numpy()[sel],
                                                       np.where(over, do, du)[sel],
                                                       np.where(over, qo, 1 - qo)[sel]))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=[2025, 2026, 2027])
    a = ap.parse_args()
    config = BacktestConfig()
    validate_config(config)
    OUT.mkdir(parents=True, exist_ok=True)
    text = ["Box-score projector logic (owner-supplied), point-in-time, on the repository's data",
            "(BBR team logs; closing lines one book per game). Readings fixed before the run.",
            "Production rows are hindsight (reconstructed / rebuilt v11). ROI ± 95%.", ""]
    for y in a.seasons:
        team_games = prepare_team_game_rows(league_game_log_from_bbr(y))
        preds = run_point_in_time_backtest(team_games, config)
        preds["game_date"] = pd.to_datetime(preds["game_date"]).dt.strftime("%Y-%m-%d")
        P = preds.merge(odds_and_production(y), on=["game_date", "home_abbr", "away_abbr"], how="left")
        P = P.merge(tstar_totals(y), on=["game_date", "home_abbr", "away_abbr"], how="left")
        P.to_csv(OUT / f"predictions_{y}.csv", index=False, float_format="%.4g")
        books = [b for b in P["close_book"].dropna().unique()]
        # all games: accuracy only (books are never pooled against the lines)
        odds_cols = ["close_spread", "close_home_spread_odds", "close_away_spread_odds",
                     "close_total", "close_over", "close_under"]
        text += evaluate(P.assign(**{c: np.nan for c in odds_cols}), f"{y - 1}-{str(y)[2:]} all games")
        for b in books:
            text += evaluate(P[P["close_book"] == b], f"{y - 1}-{str(y)[2:]} {b} only")
        text.append("")
    out = "\n".join(text) + "\n"
    (OUT / "RESULTS.txt").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
