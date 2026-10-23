#!/usr/bin/env python3
"""Daily NBA build: grade, score today's slate against the market, render.

  python build_site.py                 # today's ET slate
  python build_site.py --date 2026-11-20
  python build_site.py --render-only   # rebuild pages from committed data

Steps
  1. Grade: finished games from earlier slates get final scores and one
     book's open/close (ESPN core odds; the row's pregame book if it has a
     close, see market.pick_close). Pending games never receive a close.
  2. Score: every regular-season game on today's ESPN scoreboard that has not
     tipped gets the composite delta, P(home win), and the current moneylines
     of the first listed book in market.BOOKS (DraftKings, then ESPN BET);
     written to data/nba_ledger.csv only while before tip. Each game's ESPN
     injury list is snapshotted alongside to data/nba_injuries.csv under the
     same pregame lock (shown on the Picks cards; the model does not read it),
     and Sleeper's (RotoWire's) projected lineups to data/nba_sleeper.csv
     (first and latest pregame snapshot; research data, read by nothing).
     Preseason games go to data/nba_preseason.csv instead (score_preseason),
     graded the same way and never mixed with the native ledger.
  3. Render public/index.html, grades.html, market-calibration.html and
     preseason.html.

The model is nba_composite.py, unchanged. This file only feeds it pregame game
logs (games strictly before the slate date) and records what it said.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import analysis
import cold_start
import kalshi
import ledger
import market
import nba_composite as nc
import player_availability as pav
import report
import sleeper

ET = ZoneInfo("America/New_York")
# Pages show times in Pacific (PDT/PST, labelled PT); slate dates, the build
# schedule and the injury report stay on the NBA's ET calendar.
PT = ZoneInfo("America/Los_Angeles")
OUT_DIR = Path("public")
REPORT_PATH = Path("data") / report.REPORT_NAME
# v2 = v1 from game 10 on, plus the last-season carryover model (cold_start.py)
# for games where min(games played) is 1-9. v1 abstained before game 10.
MODEL_TAG = "fourfactors_hl25_b2b_carry25_v2"
# v3 = v2 plus pregame player availability for games 10+ (player_availability.py,
# model/logit_avail.json): the NBA injury report's Out/Doubtful players,
# weighted by last-season BPM, relative to the rating window. Rows fall back
# to v2 (and keep the v2 tag) when the report or box history is missing.
MODEL_TAG_V3 = "fourfactors_hl25_b2b_carry25_avail_v3"
# v4 = the rating's slope grows through the season: + e*delta*phase
# (phase = days since opening night / 175, capped at 1; research/
# calibration_shape.py) in both the base logit (model/logit.json) and the
# availability logit (model/logit_avail.json). MODEL_TAG_V4 tags v4 rows
# without an injury report (and games 1-9, carryover unchanged);
# MODEL_TAG_V4_AVAIL tags rows scored with the report. Which tag a row gets
# follows the fitted model files, so v3 files keep producing v2/v3 tags.
MODEL_TAG_V4 = "fourfactors_hl25_b2b_carry25_phase_v4"
MODEL_TAG_V4_AVAIL = "fourfactors_hl25_b2b_carry25_phase_avail_v4"
# v5 = v4 plus opponent 3-point luck (luck_def) and roster talent (talent_diff,
# minutes share x last-season BPM over the previous box score) in both games-10+
# logits (research/team_quality.py; nba_composite.V5_FEATURES). A game whose v5
# terms cannot be computed (no box scores / 3PA) is scored by the frozen v4
# base logit (model/logit_v4.json) and keeps the v4 tag.
MODEL_TAG_V5 = "fourfactors_hl25_b2b_carry25_phase_luck_talent_v5"
MODEL_TAG_V5_AVAIL = "fourfactors_hl25_b2b_carry25_phase_luck_talent_avail_v5"
# v6 = v5 plus the own free-throw percentage gap (ft_diff: decayed FTM/FTA,
# percentage points, home - away; nba_composite.V6_FEATURES) in both games-10+
# logits (diagnostics/v5_audit, Task D and the D.6 gate). The four factors read
# free throws only as FTA/FGA. Same v4 fallback as v5.
MODEL_TAG_V6 = "fourfactors_hl25_b2b_carry25_phase_luck_talent_ft_v6"
MODEL_TAG_V6_AVAIL = "fourfactors_hl25_b2b_carry25_phase_luck_talent_ft_avail_v6"
# v7 = v6 plus roster talent (talent_diff, the games-10+ term) in the games 1-9
# logit (model/logit_early.json; owner's decision 2026-10-22,
# research/early_roster.py). Only the early route's math changes, so only its
# rows take the v7 tag; games 10+ keep the v6 tags. An early game whose
# talent cannot be computed is scored by the frozen carryover logit
# (model/logit_early_v6.json) and keeps the v6 tag, and every early row also
# records that logit's P (carry_p_home), on which H1 and H6 stay.
MODEL_TAG_V7_EARLY = "fourfactors_hl25_b2b_carry25_talent_early_v7"
# v8 = one model for every game, opening night included (owner's decision
# 2026-10-22; research/unified.py). The rating carries last season in at
# 0.25 x (1 - games played / 10), so it fades out by game 10
# (nba_composite.fade_v8_games); the base and availability logits keep the v6
# terms and are fitted on every game (availability from game 1); game 0's
# roster is the preseason box scores (player_availability.roster_talent). A
# game whose terms are missing falls back to the frozen carryover logit
# before game 10 (MODEL_TAG, the v2 formula) and to the frozen v4 logit
# after. Games 1-9 still record the carryover logit's P (carry_p_home) for
# H1 and H6. Model files carry "fade_games" (is_v8).
MODEL_TAG_V8 = "fourfactors_hl25_b2b_fade10_phase_luck_talent_ft_v8"
MODEL_TAG_V8_AVAIL = "fourfactors_hl25_b2b_fade10_phase_luck_talent_ft_avail_v8"
# v9 = v8's inputs, routes and seasons, fitted by least squares on the final
# home margin (owner's decision 2026-10-23; research/margin_model.py): the
# model predicts the margin m and P(home win) = Phi(m / sigma)
# (nba_composite.fit_margin / predict). Rows record m (margin_home), the
# model's own spread. The fallbacks (carryover, frozen v4) are unchanged
# logits and keep their tags. Model files carry "link": "margin" (is_v9).
MODEL_TAG_V9 = "fourfactors_hl25_b2b_fade10_phase_luck_talent_ft_margin_v9"
MODEL_TAG_V9_AVAIL = "fourfactors_hl25_b2b_fade10_phase_luck_talent_ft_margin_avail_v9"
# Exhibition rows (data/nba_preseason.csv, score_preseason): the v2 carryover
# logit on last season's games alone.
PRESEASON_TAG = "fourfactors_hl25_b2b_carry25_preseason_g0"
FALLBACK_FILE = "logit_v4.json"
AVAIL_TAGS = (MODEL_TAG_V3, MODEL_TAG_V4_AVAIL, MODEL_TAG_V5_AVAIL,
              MODEL_TAG_V6_AVAIL, MODEL_TAG_V8_AVAIL, MODEL_TAG_V9_AVAIL)
ACTIVE_TAGS = [MODEL_TAG]          # set by main() from the fitted model files
TEAM_NAMES = {
    "ATL": "Hawks", "BOS": "Celtics", "BRK": "Nets", "CHO": "Hornets",
    "CHI": "Bulls", "CLE": "Cavaliers", "DAL": "Mavericks", "DEN": "Nuggets",
    "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers",
    "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat",
    "MIL": "Bucks", "MIN": "Timberwolves", "NOP": "Pelicans", "NYK": "Knicks",
    "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHO": "Suns",
    "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors",
    "UTA": "Jazz", "WAS": "Wizards",
}


def log(msg):
    print(msg, flush=True)


def et_today():
    return datetime.now(ET).strftime("%Y-%m-%d")


def season_for(date):
    """Basketball-Reference season year: 2026-27 season -> 2027."""
    d = pd.Timestamp(date)
    return d.year + 1 if d.month >= 8 else d.year


# ------------------------------------------------------------- model io ----
def load_model():
    try:
        return nc.load_json("weights.json"), nc.load_json("logit.json")
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None


def load_avail():
    """The v3 availability logit (model/logit_avail.json), or None."""
    try:
        return nc.load_json(pav.MODEL_FILE)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_fallback():
    """The frozen v4 base logit (model/logit_v4.json) for games whose v5
    terms are missing, or None."""
    try:
        return nc.load_json(FALLBACK_FILE)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def has_phase(model):
    return model is not None and "d_phase" in (model.get("features") or [])


def is_v5(model):
    """v5 or later (v6 keeps every v5 term)."""
    return model is not None and "talent_diff" in (model.get("features") or [])


def is_v6(model):
    return model is not None and "ft_diff" in (model.get("features") or [])


def is_v8(model):
    """v8: one faded rating and one pair of logits for every game."""
    return model is not None and model.get("fade_games") is not None


def is_v9(model):
    """v9: v8 fitted on the final margin (nba_composite.is_margin)."""
    return is_v8(model) and nc.is_margin(model)


def is_early_talent(early):
    """The v7 early logit (talent_diff in its features)."""
    return early is not None and "talent_diff" in (early.get("features") or [])


def model_tag(model, avail=False):
    """The tag for a row scored by `model` (the availability logit if avail)."""
    if is_v9(model):
        return MODEL_TAG_V9_AVAIL if avail else MODEL_TAG_V9
    if is_v8(model):
        return MODEL_TAG_V8_AVAIL if avail else MODEL_TAG_V8
    if avail:
        return (MODEL_TAG_V6_AVAIL if is_v6(model) else
                MODEL_TAG_V5_AVAIL if is_v5(model) else
                MODEL_TAG_V4_AVAIL if has_phase(model) else MODEL_TAG_V3)
    return (MODEL_TAG_V6 if is_v6(model) else
            MODEL_TAG_V5 if is_v5(model) else
            MODEL_TAG_V4 if has_phase(model) else MODEL_TAG)


def active_tags(model, avail_model, fallback=None, early=None):
    """Tags the current model files produce (page footer, ledger split),
    including the v4 fallback's when the base model is v5 and the v7 early
    tag when the early logit has talent."""
    tags = [model_tag(model)]
    if avail_model is not None:
        tags.append(model_tag(avail_model, avail=True))
    if is_v5(model) and fallback is not None:
        tags.append(model_tag(fallback))
    if is_v8(model):
        tags.append(MODEL_TAG)            # the carryover fallback (games 0-9)
    elif is_early_talent(early):
        tags.append(MODEL_TAG_V7_EARLY)
    return tags


def load_early():
    """The early-season logit (model/logit_early.json), or None."""
    try:
        return nc.load_json(cold_start.MODEL_FILE)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_carry():
    """The carryover logit (delta, b2b_net): the frozen model/logit_early_v6.json,
    else logit_early.json when it has no talent (v2-v6 files), else None."""
    try:
        return nc.load_json(cold_start.FROZEN_FILE)
    except (FileNotFoundError, json.JSONDecodeError):
        early = load_early()
        return None if is_early_talent(early) else early


def fresh_logs(year, today):
    """Game logs for `year`, refetched at most once per ET day."""
    stamp = nc.CACHE / f"refreshed_{year}.txt"
    refresh = not (stamp.exists() and stamp.read_text().strip() == today)
    logs = nc.load_logs(year, refresh=refresh)
    if refresh:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(today)
    return logs


def score_game_v8(logs, home, away, date, weights, model, prior_logs=None,
                  avail_model=None, avail=None, opening=None, talent=None,
                  talent0=None, fallback=None, carry=None, half_life=None):
    """v8 pregame composite and P(home win) for any game, from games strictly
    before `date` (see MODEL_TAG_V8). A team missing from `logs` (opening
    night) has played no games; `prior_logs` (last season) feed the faded
    rating until a team's game 10. talent(team, date) once a team has
    played, talent0[team] (its preseason roster) before; `avail` terms use
    `avail_model` from game 1 on. Missing terms: the carryover logit `carry`
    before game 10 (MODEL_TAG), the frozen v4 `fallback` after, else abstain.
    Games 1-9 also carry carry_p_home (H1 and H6). v9 (margin) models also
    give margin_home, the predicted home margin; the logit fallbacks do not."""
    hl = model.get("half_life", nc.HALF_LIFE) if half_life is None else half_life
    date = pd.Timestamp(date)
    out = dict(gp_home=np.nan, gp_away=np.nan, home_b2b=np.nan,
               away_b2b=np.nan, delta=np.nan, p_home=np.nan, margin_home=np.nan,
               lean=np.nan,
               p_lean=np.nan)
    prior_logs = prior_logs or {}
    empty = pd.DataFrame(columns=["date"] + nc.COLS + nc.X3)
    lh, la = logs.get(home, empty), logs.get(away, empty)
    ih = int((pd.to_datetime(lh["date"]) < date).sum())
    ia = int((pd.to_datetime(la["date"]) < date).sum())
    g = min(ih, ia)
    out.update(gp_home=ih, gp_away=ia)
    rh, ra = nc.rest_days(lh, ih, date), nc.rest_days(la, ia, date)
    ph, pa = prior_logs.get(home), prior_logs.get(away)
    pct = nc.league_3p_before(logs, date) if logs else float("nan")
    if not np.isfinite(pct):
        pct = nc.last_season_3p(prior_logs)
    x = nc.fade_inputs(lh, ih, ph, la, ia, pa, pct, weights, hl)
    opening = nc.season_opening(logs) if opening is None and logs else opening

    def tal(tm, i):
        if i >= 1:
            return talent(tm, date) if talent else float("nan")
        return (talent0 or {}).get(tm, float("nan"))

    vals, d = {}, np.nan
    if x is not None:
        d = x["delta"]
        vals = {**nc.logit_inputs(d, rh, ra, date, opening),
                "luck_def": x["luck_def"], "ft_diff": x["ft_diff"],
                "talent_diff": tal(home, ih) - tal(away, ia), **(avail or {})}

    def ready(m):
        return m is not None and x is not None and all(
            np.isfinite(vals.get(f, np.nan)) for f in m["features"])
    cd, use = None, None
    if avail is not None and g >= 1 and ready(avail_model):
        use, out["model_tag"] = avail_model, model_tag(avail_model, avail=True)
    elif ready(model):
        use, out["model_tag"] = model, model_tag(model)
    if carry is not None and g < nc.MIN_GAMES:
        cd = cold_start.carry_delta(lh, ih, ph, la, ia, pa, weights,
                                    rho=carry.get("rho", cold_start.RHO),
                                    half_life=carry.get("half_life", nc.HALF_LIFE))
    if use is None:
        if cd is not None:
            use, out["model_tag"], d = carry, MODEL_TAG, cd
            vals = {**nc.logit_inputs(cd, rh, ra, date, opening)}
        elif g >= nc.MIN_GAMES and ready(fallback):
            use, out["model_tag"] = fallback, model_tag(fallback)
        else:
            return out
    xs = [[vals[f] for f in use["features"]]]
    p = float(nc.predict(use, xs)[0])
    if nc.is_margin(use):                 # v9: the model's own home margin
        out["margin_home"] = round(float(nc.predict_margin(use, xs)[0]), 2)
    if cd is not None and 1 <= g < nc.MIN_GAMES:
        cv = nc.logit_inputs(cd, rh, ra, date, opening)
        out["carry_p_home"] = round(float(nc.predict(
            carry, [[cv[f] for f in carry["features"]]])[0]), 5)
    lean_home = p >= 0.5
    out.update(home_b2b=int(rh == 0), away_b2b=int(ra == 0),
               delta=round(float(d), 3), p_home=round(p, 5),
               lean=home if lean_home else away,
               p_lean=round(p if lean_home else 1 - p, 5))
    return out


def score_game(logs, home, away, date, weights, model, half_life=None,
               early=None, prior_logs=None, avail_model=None, avail=None,
               opening=None, talent=None, fallback=None, carry=None, talent0=None):
    """Pregame composite and P(home win) from games strictly before `date`.

    From MIN_GAMES games on: the base logit. Below that, when both teams have
    played at least once and `early` (model/logit_early.json) and last
    season's logs are given: the carryover model (cold_start.py). Otherwise
    (game 0, or no early model) it abstains: delta/p NaN.

    v3: from MIN_GAMES on, when `avail_model` (model/logit_avail.json) and this
    game's availability terms `avail` ({'av_min', 'av_bpm'}) are given, the
    availability logit is used and the row is tagged MODEL_TAG_V3.

    v4: models whose features include d_phase get delta * season phase, with
    phase measured from `opening` (default: the earliest date in `logs`, i.e.
    opening night). Every scored row carries model_tag (see model_tag()).

    v5: models whose features include luck_def / talent_diff get opponent
    3-point luck (nba_composite.opp_luck, league 3P% before `date`) and
    talent(team, date) (player_availability.talent_fn), home - away. When a
    term is missing the game is scored by `fallback` (the frozen v4 base
    logit) and tagged v4; without a fallback it abstains.

    v6: models whose features include ft_diff also get the own FT% gap
    (nba_composite.own_ft_pct over each team's games before `date`), home -
    away; same fallback.

    v7: an `early` logit with talent_diff (games 1-9) gets talent(team, date)
    like the games-10+ logits and the MODEL_TAG_V7_EARLY tag; without talent
    the game is scored by `carry` (the carryover logit, default `early`
    itself when it has no talent) and keeps the v6 tag. Every early row also
    carries carry_p_home, the carryover logit's P (H1 and H6 read it).

    v8 models (is_v8) are scored by score_game_v8 (talent0: preseason rosters).
    """
    if is_v8(model):
        return score_game_v8(logs, home, away, date, weights, model, prior_logs,
                             avail_model, avail, opening, talent, talent0,
                             fallback, carry, half_life)
    hl = model.get("half_life", nc.HALF_LIFE) if half_life is None else half_life
    date = pd.Timestamp(date)
    out = dict(gp_home=np.nan, gp_away=np.nan, home_b2b=np.nan,
               away_b2b=np.nan, delta=np.nan, p_home=np.nan, lean=np.nan,
               p_lean=np.nan)
    if home not in logs or away not in logs:
        return out
    lh, la = logs[home], logs[away]
    ih, ia = int((lh["date"] < date).sum()), int((la["date"] < date).sum())
    out.update(gp_home=ih, gp_away=ia)
    rh, ra = nc.rest_days(lh, ih, date), nc.rest_days(la, ia, date)
    if min(ih, ia) >= nc.MIN_GAMES:
        fh = nc.decayed_features(lh, ih, hl)
        fa = nc.decayed_features(la, ia, hl)
        d = nc.composite(fh - fa, weights["sd"], weights["w"])
        use, out["model_tag"] = model, model_tag(model)
        if avail_model is not None and avail is not None:
            use, out["model_tag"] = avail_model, model_tag(avail_model, avail=True)
    elif min(ih, ia) >= 1 and early is not None and prior_logs:
        d = cold_start.carry_delta(lh, ih, prior_logs.get(home), la, ia,
                                   prior_logs.get(away), weights,
                                   rho=early.get("rho", cold_start.RHO),
                                   half_life=early.get("half_life", nc.HALF_LIFE))
        if d is None:
            return out
        if carry is None and not is_early_talent(early):
            carry = early
        use, out["model_tag"] = early, (MODEL_TAG_V7_EARLY if is_early_talent(early)
                                        else model_tag(model))
        fallback = carry          # the carryover logit, not the games-10+ v4 one
    else:
        return out
    vals = {**nc.logit_inputs(d, rh, ra, date, nc.season_opening(logs)
                              if opening is None else opening), **(avail or {})}
    feats = use["features"]
    if "luck_def" in feats:
        pct = nc.league_3p_before(logs, date)
        vals["luck_def"] = nc.opp_luck(lh, ih, pct, weights, hl) \
            - nc.opp_luck(la, ia, pct, weights, hl)
    if "talent_diff" in feats:
        vals["talent_diff"] = (talent(home, date) - talent(away, date)) if talent \
            else float("nan")
    if "ft_diff" in feats:
        vals["ft_diff"] = nc.own_ft_pct(lh, ih, hl) - nc.own_ft_pct(la, ia, hl)
    if not all(np.isfinite(vals.get(f, np.nan)) for f in feats):
        if fallback is None:
            return out
        use = fallback
        out["model_tag"] = model_tag(model) if use is carry else model_tag(fallback)
        feats = use["features"]
    p = float(nc.predict(use, [[vals[f] for f in feats]])[0])
    if carry is not None and min(ih, ia) < nc.MIN_GAMES:
        out["carry_p_home"] = round(float(nc.predict(
            carry, [[vals[f] for f in carry["features"]]])[0]), 5)
    lean_home = p >= 0.5
    out.update(home_b2b=int(rh == 0), away_b2b=int(ra == 0),
               delta=round(d, 3), p_home=round(p, 5),
               lean=home if lean_home else away,
               p_lean=round(p if lean_home else 1 - p, 5))
    return out


# ------------------------------------------------------------- pipeline ----
def espn_close(row):
    """One ESPN book's open/close for a graded row (its pregame book first)."""
    pre_book = row["pre_book"]
    return market.pick_close(market.book_odds(str(row["game_id"])),
                             prefer=pre_book if pd.notna(pre_book) else None)


def kalshi_close(row):
    """Kalshi's last pregame-minute pair for a graded preseason row."""
    return kalshi.close_odds(str(row["slate_date"]), row["home"], row["away"],
                             row["tip_utc"])


def kalshi_grade(row):
    """Kalshi's close and open for a graded regular-season row (from 2025-26),
    beside the sportsbook's: kalshi.grade_odds."""
    if not pd.to_numeric(row["season"], errors="coerce") >= ledger.KALSHI_FROM_SEASON:
        return None
    return kalshi.grade_odds(str(row["slate_date"]), row["home"], row["away"],
                             row["tip_utc"])


def grade(led, today, close=espn_close, kalshi_close_fn=None):
    """Final scores for finished games from earlier slates, with the closing
    pair from `close(row)` (ESPN books; kalshi_close for preseason rows) and,
    with `kalshi_close_fn` (kalshi_grade, regular season), Kalshi's beside it.
    Recent graded rows still missing a close (ledger.missing_close) are
    asked again."""
    todo = pd.concat([ledger.pending(led, today), ledger.missing_close(led, today)]
                     + ([ledger.missing_close(led, today, kalshi=True)]
                        if kalshi_close_fn else []))
    todo = todo[~todo.index.duplicated()]
    if not len(todo):
        return led, 0
    n = 0
    for date, rows in todo.groupby("slate_date"):
        try:
            games = {g["game_id"]: g for g in market.scoreboard(str(date))}
        except Exception as e:  # noqa: BLE001
            log(f"grade {date}: scoreboard failed ({e!r}); retry next run")
            continue
        for _, row in rows.iterrows():
            gid = str(row["game_id"])
            g = games.get(gid)
            if not g or not g["completed"]:
                continue
            try:
                odds = close(row)
            except Exception as e:  # noqa: BLE001
                log(f"grade {gid}: odds failed ({e!r}); graded without close")
                odds = None
            k = None
            if kalshi_close_fn is not None:
                try:
                    k = kalshi_close_fn(row)
                except Exception as e:  # noqa: BLE001
                    log(f"grade {gid}: Kalshi failed ({e!r}); retried next build")
            n += ledger.apply_result(led, gid, g, odds, k)
    return led, n


def report_utc(report_time):
    """The NBA injury report's edition time (naive ET) as a UTC string, or
    NaN when the build read no report."""
    if report_time is None or pd.isna(report_time):
        return np.nan
    return ledger.fmt_utc(pd.Timestamp(report_time).tz_localize(ET).to_pydatetime())


def route(row, model):
    """Which formula scored a score_game row: early (games 1-9, carryover),
    avail (availability logit), base (base logit), v4 (the frozen v4
    fallback of a v5 model), or abstain (no P). v8: avail, base, carry (the
    carryover fallback before game 10), v4 or abstain. Reads the row;
    changes no routing."""
    if not np.isfinite(pd.to_numeric(row.get("p_home"), errors="coerce")):
        return "abstain"
    if is_v8(model):
        tag = row.get("model_tag")
        return ("avail" if tag in AVAIL_TAGS else
                "base" if tag in (MODEL_TAG_V8, MODEL_TAG_V9)
                else "carry" if tag == MODEL_TAG else "v4")
    if min(row.get("gp_home", 0), row.get("gp_away", 0)) < nc.MIN_GAMES:
        return "early"
    tag = row.get("model_tag")
    if tag in AVAIL_TAGS:
        return "avail"
    if is_v5(model) and tag != model_tag(model):
        return "v4"
    return "base"


def score_slate(today, weights, model, now=None):
    games = [g for g in market.scoreboard(today)
             if g["season_type"] == market.REGULAR_SEASON
             and g["home"] in market.BBR_TEAMS and g["away"] in market.BBR_TEAMS]
    if not games:
        return []
    now = now or ledger.utc_now()
    pre = [g for g in games
           if g["state"] == "pre" and ledger.parse_utc(g["tip_utc"]) > now]
    if not pre:
        return []
    logs = fresh_logs(season_for(today), today)
    early, carry, prior = load_early(), load_carry(), None
    teams = {t for g in pre for t in (g["home"], g["away"])}
    if (early is not None or is_v8(model)) and (teams - set(logs) or any(
            int((logs[t]["date"] < pd.Timestamp(today)).sum()) < nc.MIN_GAMES
            for t in teams & set(logs))):
        try:
            prior = nc.load_logs(season_for(today) - 1)   # cached; completed season
        except Exception as e:  # noqa: BLE001
            log(f"last-season logs failed ({e!r}); early games abstain")
    avail_model, live = load_avail(), None
    if avail_model is not None or is_v5(model):     # v5 needs box-score talent
        try:
            live = pav.LiveAvailability(season_for(today), today,
                                        now.astimezone(ET).replace(tzinfo=None))
            log(f"availability: report {live.report_time} "
                f"({0 if live.report is None else len(live.report)} rows); "
                f"BPM covers {100 * live.bpm_minutes:.0f}% of minutes")
        except Exception as e:  # noqa: BLE001 - base / v4 fallback rows
            log(f"availability failed ({e!r}); rows use the base logit "
                "(v5: the v4 fallback)")
    opening = nc.season_opening(logs) or pd.Timestamp(today)
    fallback = load_fallback() if is_v5(model) else None
    talent = getattr(live, "talent", None)
    talent0 = getattr(live, "talent0", None)
    rows = []
    for g in pre:
        r = dict(game_id=g["game_id"], slate_date=today,
                 season=season_for(today), tip_utc=g["tip_utc"],
                 model_tag=model_tag(model), home=g["home"], away=g["away"])
        terms = None
        if live is not None:
            try:
                terms = live.terms(g["game_id"], g["home"], g["away"], today)
            except Exception as e:  # noqa: BLE001
                log(f"availability {g['game_id']}: {e!r}")
        r.update(score_game(logs, g["home"], g["away"], today, weights, model,
                            early=early, prior_logs=prior, avail_model=avail_model,
                            avail=None if terms is None else
                            {k: terms[k] for k in ("av_min", "av_bpm")},
                            opening=opening, talent=talent, fallback=fallback,
                            carry=carry, talent0=talent0))
        # decision-time record (ledger first_report_utc / first_route): the
        # report edition this build read, whether or not it covered the game
        r["report_utc"] = report_utc(getattr(live, "report_time", None))
        r["route"] = route(r, model)
        add_pregame_price(r)
        rows.append(r)
    add_kalshi_pregame(rows, today)
    return rows


def add_kalshi_pregame(rows, today):
    """Kalshi's live pair beside each row's sportsbook price, in place
    (kalshi_pre_*; the pages' market where present, analysis.market_view).
    One request per build; a failure leaves the rows on the book alone."""
    try:
        book = kalshi.open_games()
    except Exception as e:  # noqa: BLE001
        log(f"kalshi pregame failed ({e!r}); rows keep the sportsbook only")
        return
    for r in rows:
        o = kalshi.pregame_odds(book, r["slate_date"], r["home"], r["away"])
        for k in ("home_ml", "away_ml", "q_home"):
            r[f"kalshi_pre_{k}"] = o.get(f"cur_{k}", np.nan)


def add_pregame_price(r):
    """The current moneyline pair of the first book in market.BOOKS, in place
    (pre_book, pre_home_ml, pre_away_ml, pre_q_home; price_note when none),
    and that book's current spread from the same response (pre_spread,
    pre_home_spread_odds, pre_away_spread_odds; blank when it has none)."""
    try:
        odds = market.pick_pregame(market.book_odds(r["game_id"]))
        if not odds:
            r["price_note"] = "no DraftKings or ESPN BET moneyline pair"
    except Exception as e:  # noqa: BLE001
        log(f"odds {r['game_id']}: {e!r}")
        odds = {}
        r["price_note"] = f"odds fetch failed: {type(e).__name__}"
    r["pre_book"] = odds.get("book")
    r["pre_home_ml"] = odds.get("cur_home_ml")
    r["pre_away_ml"] = odds.get("cur_away_ml")
    q = market.devig(r["pre_home_ml"], r["pre_away_ml"])
    r["pre_q_home"] = round(q, 5) if np.isfinite(q) else np.nan
    r["pre_spread"] = odds.get("cur_spread")
    r["pre_home_spread_odds"] = odds.get("cur_home_spread_odds")
    r["pre_away_spread_odds"] = odds.get("cur_away_spread_odds")


def score_preseason(today, weights, now=None):
    """Exhibition tracking (data/nba_preseason.csv): every NBA-vs-NBA
    preseason game on today's scoreboard that has not tipped, scored by the
    early-season carryover logit (model/logit_early_v6.json; before v7
    model/logit_early.json) on LAST SEASON's games only (no rosters: v8's
    game-0 roster is these games' box scores, read once the season opens).

    No new fit: the delta and logit are the shipped carryover ones (the
    offseason discount rho cancels when every row is last season's). b2b_net
    comes from yesterday's scoreboard (any game, any season type); when that
    fetch fails it is 0. The logit was fitted on regular-season games, and
    preseason rotations are not, so this measures how far the model's
    offseason view travels, not the model's regular-season skill. Rows are
    tagged PRESEASON_TAG with route "preseason".

    Prices come from Kalshi (kalshi.py, book "kalshi"), not a sportsbook:
    the YES asks with the taker fee, one request per build; the close is
    the last 1-minute candle before tip, read at grading (kalshi_close).
    """
    games = [g for g in market.scoreboard(today)
             if g["season_type"] == market.PRESEASON
             and g["home"] in market.BBR_TEAMS and g["away"] in market.BBR_TEAMS]
    if not games:
        return []
    now = now or ledger.utc_now()
    pre = [g for g in games
           if g["state"] == "pre" and ledger.parse_utc(g["tip_utc"]) > now]
    early = load_carry()                    # the carryover logit (no box scores)
    if not pre or early is None:
        if pre:
            log("preseason: no carryover logit (model/logit_early_v6.json); skipped")
        return []
    prior = nc.load_logs(season_for(today) - 1)        # cached; completed season
    yday = (pd.Timestamp(today) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        played = {t for g in market.scoreboard(yday) for t in (g["home"], g["away"])}
    except Exception as e:  # noqa: BLE001
        log(f"preseason: yesterday's scoreboard failed ({e!r}); b2b_net = 0")
        played = set()
    try:
        book = kalshi.open_games()
        note = None
    except Exception as e:  # noqa: BLE001
        log(f"preseason: Kalshi fetch failed ({e!r}); rows unpriced")
        book, note = {}, f"Kalshi fetch failed: {type(e).__name__}"
    hl = early.get("half_life", nc.HALF_LIFE)
    rows = []
    for g in pre:
        home, away = g["home"], g["away"]
        r = dict(game_id=g["game_id"], slate_date=today, season=season_for(today),
                 tip_utc=g["tip_utc"], model_tag=PRESEASON_TAG, home=home,
                 away=away, gp_home=0, gp_away=0, route="preseason")
        d = cold_start.carry_delta(None, 0, prior.get(home), None, 0,
                                   prior.get(away), weights,
                                   rho=early.get("rho", cold_start.RHO),
                                   half_life=hl) \
            if home in prior and away in prior else None
        if d is not None:
            hb, ab = int(home in played), int(away in played)
            vals = {"delta": d, "b2b_net": ab - hb}
            p = float(nc.predict(early, [[vals[f] for f in early["features"]]])[0])
            lean_home = p >= 0.5
            r.update(home_b2b=hb, away_b2b=ab, delta=round(d, 3),
                     p_home=round(p, 5), lean=home if lean_home else away,
                     p_lean=round(p if lean_home else 1 - p, 5))
        odds = kalshi.pregame_odds(book, today, home, away)
        if not odds:
            r["price_note"] = note or "no Kalshi market with asks on both sides"
        r["pre_book"] = odds.get("book")
        r["pre_home_ml"] = odds.get("cur_home_ml")
        r["pre_away_ml"] = odds.get("cur_away_ml")
        r["pre_q_home"] = odds.get("cur_q_home", np.nan)       # kalshi.mid_q
        rows.append(r)
    return rows


# ------------------------------------------------------------- rendering ---
CSS = """
:root{
--bg:#f4f6f9;--surface:#ffffff;--surface-2:#f0f3f7;--ink:#13202f;
--muted:#4d5d6e;--faint:#647384;--line:#dce3ea;--line-2:#e9eef3;
--accent:9,120,94;--accent-tx:6,102,79;--accent-on:#ffffff;
--warm:198,84,44;--cool:52,116,168;--lean:176,124,16;
--warm-tx:164,69,36;--cool-tx:46,103,149;--lean-tx:130,92,12;
--good:25,121,62;--bad:184,60,40;
--mono:"JetBrains Mono",ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
--sans:"Archivo",system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
--shadow:0 1px 2px rgba(19,32,47,.04),0 6px 20px -12px rgba(19,32,47,.18);
--r:14px;--r-s:6px;--r-pill:999px;
/* the research pages' names for the same tokens */
--fg:var(--ink);--mut:var(--muted);--card:var(--surface);--card2:var(--surface-2);
--line2:var(--line-2);--pos:rgb(var(--good));--neg:rgb(var(--bad));
--acc:rgb(var(--accent-tx));--accb:rgb(var(--accent));--chip:rgba(var(--accent),.09);
--s1:rgb(var(--cool));--s2:rgb(var(--warm));--grid:var(--line-2);
--pos-bg:rgba(var(--good),.1);--neg-bg:rgba(var(--bad),.1)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#0c141d;--surface:#141e29;--surface-2:#18232f;--ink:#e6eef6;
--muted:#a0b1c2;--faint:#8a9cb0;--line:#243241;--line-2:#1d2936;
--accent:61,192,151;--accent-tx:110,218,184;--accent-on:#0c141d;
--warm:236,122,72;--cool:96,158,208;--lean:244,196,96;
--warm-tx:240,140,98;--cool-tx:120,176,222;--lean-tx:244,196,96;
--good:88,194,125;--bad:239,127,98;
--shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -18px rgba(0,0,0,.8)}
:root:not([data-theme="light"]) .logo.lt{display:none}
:root:not([data-theme="light"]) .logo.dk{display:block}
:root:not([data-theme="light"]) .theme .sun{display:none}
:root:not([data-theme="light"]) .theme .moon{display:block}}
:root[data-theme="dark"]{
--bg:#0c141d;--surface:#141e29;--surface-2:#18232f;--ink:#e6eef6;
--muted:#a0b1c2;--faint:#8a9cb0;--line:#243241;--line-2:#1d2936;
--accent:61,192,151;--accent-tx:110,218,184;--accent-on:#0c141d;
--warm:236,122,72;--cool:96,158,208;--lean:244,196,96;
--warm-tx:240,140,98;--cool-tx:120,176,222;--lean-tx:244,196,96;
--good:88,194,125;--bad:239,127,98;
--shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -18px rgba(0,0,0,.8)}
:root[data-theme="dark"] .logo.lt{display:none}:root[data-theme="dark"] .logo.dk{display:block}
:root[data-theme="dark"] .theme .sun{display:none}:root[data-theme="dark"] .theme .moon{display:block}
*{box-sizing:border-box}[hidden]{display:none!important}
html{background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 var(--sans);
-webkit-font-smoothing:antialiased;padding:0 16px 56px}
.mx-wrap{max-width:1120px;margin:0 auto}
a{color:rgb(var(--accent-tx))}
.skip-link{position:absolute;left:16px;top:-80px;padding:10px 14px;background:var(--surface);
border-radius:var(--r-s);z-index:10}.skip-link:focus{top:10px}
:focus-visible{outline:3px solid rgb(var(--accent));outline-offset:3px}
code{font-family:var(--mono);font-size:.92em}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.stamp{font-family:var(--mono);font-size:12.5px}
/* header (shared with the NFL site) */
.topbar{display:flex;align-items:center;gap:16px;padding:18px 0;margin-bottom:22px;border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:10px;font:800 19px/1 var(--sans);letter-spacing:-.02em;
color:var(--ink);text-decoration:none;white-space:nowrap}
.brand-mark{display:grid;place-items:center;width:36px;height:36px;border-radius:10px;
background:rgb(var(--accent));color:#fff;font:800 11px/1 var(--sans);letter-spacing:.03em}
.brand-version{font:600 11px/1 var(--mono);color:var(--muted);border:1px solid var(--line);padding:4px 6px;border-radius:6px}
.nav{display:flex;gap:4px;margin-left:auto;padding:4px;background:var(--surface);border:1px solid var(--line);border-radius:12px}
.nav a{padding:8px 16px;border-radius:8px;color:var(--muted);text-decoration:none;font:650 14px/1.2 var(--sans);white-space:nowrap}
.nav a:hover{color:var(--ink);background:var(--surface-2)}
.nav a[aria-current="page"]{background:var(--ink);color:var(--surface)}
.theme{appearance:none;display:grid;place-items:center;width:38px;height:38px;border:1px solid var(--line);
background:var(--surface);color:var(--muted);border-radius:10px;cursor:pointer;padding:0}
.theme:hover{color:var(--ink)}.theme svg{width:18px;height:18px}.theme .moon{display:none}
.subnav{display:flex;flex-wrap:wrap;gap:6px;margin:-6px 0 18px}
.subnav a{border:1px solid var(--line);border-radius:var(--r-pill);padding:5px 12px;background:var(--surface);
color:var(--muted);text-decoration:none;font:600 13px/1.3 var(--sans)}
.subnav a:hover{color:var(--ink)}
.subnav a[aria-current="page"]{color:rgb(var(--accent-tx));border-color:rgba(var(--accent),.45);background:rgba(var(--accent),.09)}
.subnav .lbl{font:700 11.5px/2.2 var(--sans);letter-spacing:.07em;text-transform:uppercase;color:var(--faint);margin-right:4px}
/* page heading, tiles, notes (public pages) */
.gr-head{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:12px 24px;margin:6px 0 22px}
.eyebrow{font:700 12px/1.2 var(--sans);letter-spacing:.08em;text-transform:uppercase;color:rgb(var(--accent-tx));margin-bottom:8px}
.gr-h1{font:800 34px/1.08 var(--sans);letter-spacing:-.035em;margin:0 0 8px}
.gr-lead{color:var(--muted);font-size:15px;line-height:1.6;max-width:68ch}
.gr-summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:0 0 18px}
.gr-stat{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:16px 18px;box-shadow:var(--shadow);min-width:0}
.gr-stat .l{font:700 11.5px/1.2 var(--sans);letter-spacing:.07em;text-transform:uppercase;color:var(--faint)}
.gr-stat .v{font:800 28px/1.1 var(--mono);letter-spacing:-.02em;margin:8px 0 6px;font-variant-numeric:tabular-nums}
.gr-stat .v.pos,.gr-stat .v.good{color:rgb(var(--good))}.gr-stat .v.neg,.gr-stat .v.bad{color:rgb(var(--bad))}
.gr-stat .s{font:500 13px/1.45 var(--sans);color:var(--muted)}
.meter{position:relative;height:6px;border-radius:var(--r-pill);background:var(--line-2);margin:10px 0 6px;overflow:visible}
.meter i{position:absolute;left:0;top:0;bottom:0;border-radius:var(--r-pill);background:rgb(var(--accent))}
.meter b{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ink);border-radius:1px}
.gr-note{background:var(--surface-2);border:1px solid var(--line-2);border-radius:10px;
padding:12px 15px;margin:0 0 16px;font-size:14px;line-height:1.6;color:var(--muted)}
.gr-note b{color:var(--ink)}
.flag-note{border-left:3px solid rgb(var(--lean));background:rgba(var(--lean),.09);border-radius:10px;
padding:12px 15px;margin:0 0 16px;font-size:14px;line-height:1.6;color:var(--muted)}
.flag-note b{color:var(--ink)}
h2.sec{font:800 19px/1.25 var(--sans);letter-spacing:-.02em;text-transform:none;margin:36px 0 12px}
.sec-lead{color:var(--muted);font-size:14px;margin:-4px 0 14px;max-width:72ch}
.rec{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:14px 18px;font-size:14px}
.rec .l{font:700 11.5px/1.2 var(--sans);letter-spacing:.07em;text-transform:uppercase;color:var(--faint);margin-bottom:6px}
.gr-tablewrap{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);
box-shadow:var(--shadow);overflow-x:auto;margin-bottom:12px}
table.gr{width:100%;border-collapse:collapse;font-size:14px}
table.gr th{font:700 11.5px/1.2 var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--faint);
text-align:left;padding:11px 14px;border-bottom:1px solid var(--line);white-space:nowrap;background:var(--surface-2);position:static}
table.gr td{padding:11px 14px;border-bottom:1px solid var(--line-2);vertical-align:middle;text-align:left;
font-family:var(--sans);font-size:14px;white-space:normal}
table.gr tr:last-child td{border-bottom:0}table.gr tbody tr:hover{background:var(--surface-2)}
table.gr .n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.small{font-size:13px;line-height:1.55}
.team-code{font:800 13px/1 var(--mono);letter-spacing:.02em}
.empty{background:var(--surface);border:1px dashed var(--line);border-radius:var(--r);padding:22px;color:var(--muted);text-align:center}
/* filters and game cards (public Picks page) */
.toolbar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;margin:26px 0 12px}
.toolbar h2.sec{margin:0}
.seg{display:inline-flex;gap:3px;padding:3px;background:var(--surface);border:1px solid var(--line);border-radius:10px}
.seg button{appearance:none;border:0;background:transparent;color:var(--muted);font:650 13px/1 var(--sans);
padding:8px 12px;border-radius:7px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--surface-2);color:var(--ink);box-shadow:inset 0 0 0 1px var(--line)}
.seg .ct{font-family:var(--mono);color:var(--faint);margin-left:4px}
.ggrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px;align-items:start;margin-bottom:18px}
.gcard{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);overflow:clip}
.gcard.picked-card{border-color:rgba(var(--accent),.35)}
.game-summary{display:block;padding:14px 16px;list-style:none;cursor:pointer}
.game-summary::-webkit-details-marker{display:none}.game-summary::marker{content:""}
.game-summary:hover{background:var(--surface-2)}
.gcard[open]>.game-summary{border-bottom:1px solid var(--line-2)}
.g-meta{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:10px}
.g-meta .when{font:600 12.5px/1.2 var(--sans);color:var(--muted)}
.g-teams{display:flex;flex-direction:column;gap:4px}
.g-team,.g-cols{display:grid;grid-template-columns:34px minmax(0,1fr) 44px 62px 30px;align-items:center;gap:8px;
padding:6px 8px;margin:0 -8px;border-radius:9px}
.g-cols{padding-top:0;padding-bottom:2px;font:700 10px/1 var(--sans);letter-spacing:.07em;text-transform:uppercase;color:var(--faint)}
.g-cols span,.g-team .wp,.g-team .px{text-align:right}
.g-team .wp{font:600 13px/1 var(--mono);color:var(--muted);font-variant-numeric:tabular-nums}
.g-team .px{display:flex;flex-direction:column;gap:3px;font:700 13.5px/1 var(--mono);font-variant-numeric:tabular-nums}
.g-team .px small{font:500 11.5px/1 var(--mono);color:var(--faint)}
.g-team.picked{background:rgba(var(--accent),.09)}
.g-team .nm{font:700 16px/1.2 var(--sans);display:flex;flex-direction:column;align-items:flex-start;gap:3px;min-width:0}
.g-team .nm .tn{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;max-width:100%}
.g-team .nm .sub{display:flex;align-items:center;gap:6px}
.g-team .nm small{font:600 11.5px/1 var(--mono);color:var(--faint)}
.g-team .sc{font:800 18px/1 var(--mono);min-width:28px;text-align:right;font-variant-numeric:tabular-nums}
.lean-pill{font:800 9.5px/1 var(--sans);font-style:normal;letter-spacing:.08em;text-transform:uppercase;
color:var(--accent-on);background:rgb(var(--accent));border-radius:4px;padding:3px 5px;flex:none}
.tchip{display:grid;place-items:center;width:34px;height:34px;flex:none;border-radius:var(--r-pill);
background:var(--surface-2);font:800 11px/1 var(--mono);position:relative}
.tchip .logo{position:absolute;inset:3px;width:28px;height:28px;object-fit:contain}
.tchip .logo.dk{display:none}.tchip.has-logo{background:transparent}.tchip.has-logo span{visibility:hidden}
.pick-lines{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:4px 12px;
margin-top:12px;padding-top:12px;border-top:1px dashed var(--line)}
.pick-lines .kind{font:700 11px/1 var(--sans);letter-spacing:.07em;text-transform:uppercase;color:var(--faint);width:100%}
.pick-lines .pk{font:800 20px/1.2 var(--mono);letter-spacing:-.01em}
.pick-lines .pk.na{font:600 14px/1.4 var(--sans);color:var(--muted)}
.pick-lines .edge{font:600 12.5px/1.3 var(--sans);color:var(--muted)}
.g-mk{font:500 12.5px/1.5 var(--mono);color:var(--muted);margin-top:6px}
.g-mk span{white-space:nowrap}
.g-flags{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.g-note{font:500 12px/1.4 var(--sans);color:var(--faint);margin-top:8px}
.g-detail{padding:14px 16px 18px}
.band{margin:12px 0 6px}
.band-t{font:700 11px/1.3 var(--sans);letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}
.band-l{margin:3px 0 0;font:600 12.5px/1.35 var(--mono);color:var(--mut)}
.band-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:8px 0 4px}
.band-stats>div{min-width:0;padding:8px 10px;background:var(--surface-2);border:1px solid var(--line);border-radius:9px}
.band-stats small{display:block;font:650 10.5px/1.35 var(--sans);letter-spacing:.04em;text-transform:uppercase;color:var(--faint)}
.band-stats strong{display:block;font:750 16px/1.4 var(--mono);font-variant-numeric:tabular-nums}
.band-stats .vs{display:block;font:500 11.5px/1.4 var(--mono);color:var(--mut);overflow-wrap:anywhere}
.g-detail p.band-n{font-size:12px;color:var(--faint);margin:6px 0 0}
.g-detail h3{font:750 12px/1.2 var(--sans);letter-spacing:.07em;text-transform:uppercase;color:var(--faint);margin:16px 0 8px}
.g-detail p{margin:8px 0;font-size:13.5px;color:var(--muted)}
.g-detail .gr-tablewrap{box-shadow:none}
.g-detail table.gr{font-size:13px}.g-detail table.gr td,.g-detail table.gr th{padding:8px 11px}
.inj{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px 20px}
.inj-t{font:700 13px/1.3 var(--sans);margin-bottom:2px}.inj-t small{color:var(--faint);margin-left:6px}
.inj ul{list-style:none;margin:0;padding:0}
.inj li{display:flex;align-items:baseline;gap:8px;padding:6px 0;border-top:1px solid var(--line2);font-size:13px}
.inj li .nm{flex:1;min-width:0}.inj li .nm small{display:block;color:var(--muted);font-size:12px}
.inj li.none{color:var(--muted)}
/* running-units chart (Performance) */
.uchart{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);padding:16px 18px 10px;margin:0 0 12px}
.uchart .head{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap}
.uchart .t{font:700 14px/1.3 var(--sans)}.uchart .v{font:800 18px/1 var(--mono)}
.uchart svg{display:block;width:100%;height:auto;margin-top:8px;overflow:visible}
.uchart .grid-ln{stroke:var(--line-2);stroke-width:1}
.uchart .zero{stroke:var(--faint);stroke-width:1;stroke-dasharray:3 3}
.uchart .ln{fill:none;stroke:rgb(var(--accent));stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.uchart .area{fill:rgba(var(--accent),.08)}
.uchart .ax{fill:var(--faint);font:500 13px var(--mono)}
.uchart .pt{fill:rgb(var(--accent));stroke:var(--surface);stroke-width:2}
.uchart-pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;margin:0 0 12px}
.uchart-pair .uchart{margin:0}.uchart-pair .empty{display:grid;place-items:center;min-height:120px}
/* badges */
.badge{display:inline-flex;align-items:center;gap:5px;font:700 11px/1 var(--sans);letter-spacing:.04em;text-transform:uppercase;
padding:5px 8px;border-radius:var(--r-pill);border:1px solid var(--line);color:var(--muted);white-space:nowrap;
background:var(--surface);vertical-align:middle;margin-right:6px}
.g-flags .badge,.g-meta .badge,td .badge{margin-right:0}
.badge.w,.badge.native{color:rgb(var(--good));border-color:rgba(var(--good),.4);background:rgba(var(--good),.09)}
.badge.l{color:rgb(var(--bad));border-color:rgba(var(--bad),.4);background:rgba(var(--bad),.09)}
.badge.lean{color:rgb(var(--accent-tx));border-color:rgba(var(--accent),.45);background:rgba(var(--accent),.09)}
.badge.warn{color:rgb(var(--warm-tx));border-color:rgba(var(--warm),.45);background:rgba(var(--warm),.09)}
.badge.hyp{color:rgb(var(--cool-tx));border-color:rgba(var(--cool),.45);background:rgba(var(--cool),.09)}
.badge.recon{background:var(--surface-2);color:var(--muted)}
.badge.live{color:rgb(var(--bad));border-color:rgba(var(--bad),.4)}
.badge.live::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
/* research pages: the same tokens on their own components */
h1{font:800 30px/1.1 var(--sans);letter-spacing:-.03em;margin:6px 0 8px}
h2{font:800 15px/1.2 var(--sans);letter-spacing:.08em;text-transform:uppercase;margin:34px 0 8px;scroll-margin-top:8px}
h3{font:750 13px/1.2 var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--faint);margin:18px 0 6px}
.lead,.note{color:var(--mut);max-width:78ch}.note{font-size:13.5px}.lead{font-size:15px;line-height:1.6}
.note code{overflow-wrap:anywhere}
.key{background:var(--card2);border:1px solid var(--line2);border-radius:10px;padding:10px 14px;
font-size:13.5px;max-width:78ch;margin:10px 0;color:var(--mut)}
.key b{color:var(--fg)}.key summary{cursor:pointer;color:var(--acc);margin-top:4px}
.key dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 12px;margin:8px 0 2px}
.key dt{font-weight:600;color:var(--fg)}.key dd{margin:0}
details.key>summary{cursor:pointer;color:var(--acc);font-weight:600}
nav.jump{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 4px;font:500 13px/1.3 var(--sans)}
nav.jump a{border:1px solid var(--line);border-radius:var(--r-pill);padding:3px 11px;background:var(--card);
color:var(--mut);text-decoration:none}
nav.jump a:hover{color:var(--fg)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:14px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:13px 15px;box-shadow:var(--shadow);min-width:0}
.tile .l{font:700 11.5px/1.25 var(--sans);letter-spacing:.07em;text-transform:uppercase;color:var(--faint)}
.tile .v{font:800 22px/1.15 var(--mono);margin:6px 0 4px;font-variant-numeric:tabular-nums}
.tile .s{font:500 12.5px/1.45 var(--mono);color:var(--mut)}
.tile.pos .v{color:var(--pos)}.tile.neg .v{color:var(--neg)}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:var(--r);
background:var(--card);box-shadow:var(--shadow);margin-bottom:12px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line2);text-align:right;white-space:nowrap}
td{font-family:var(--mono);font-size:13.5px}td.l{font-family:var(--sans);font-size:14px}
th{font:700 11.5px/1.2 var(--sans);letter-spacing:.06em;text-transform:uppercase;
color:var(--faint);background:var(--card2);border-bottom:1px solid var(--line);position:sticky;top:0}
td.l,th.l{text-align:left}tr:last-child td{border-bottom:0}
th.k,td.k{background:var(--chip)}
.pos{color:var(--pos)}.neg{color:var(--neg)}.mut{color:var(--mut)}
.chip{display:inline-block;background:var(--chip);border-radius:4px;padding:0 6px;
font:800 13px/1.5 var(--mono);color:var(--acc)}
.basis{display:inline-block;font:700 11px/1.4 var(--sans);letter-spacing:.04em;
text-transform:uppercase;border:1px solid var(--line);border-radius:var(--r-pill);padding:1px 7px;color:var(--mut)}
details.more{margin:12px 0}details.more>summary{cursor:pointer;font-weight:700;font-size:14px;padding:4px 0}
details.more>summary .mut{font-weight:400;font-size:13px}
.charts{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start;margin:12px 0}
.chart{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:12px 14px;
width:100%;max-width:440px;box-shadow:var(--shadow)}
.chart svg{display:block;width:100%;height:auto}
.chart .t{font-weight:700;font-size:14px}
.legend{display:flex;gap:14px;font-size:12.5px;color:var(--mut);margin:2px 0 4px}
.legend svg{display:inline-block;vertical-align:-2px;width:auto}
.charts .note{flex:1;min-width:240px;margin:0}
.ax{fill:var(--mut);font-size:11px;font-family:var(--mono)}.gl{stroke:var(--grid);stroke-width:1}
.diag{stroke:var(--mut);stroke-width:1;stroke-dasharray:4 4}
.eb{stroke-width:2;stroke-linecap:round;opacity:.55}
.s1{fill:var(--s1);stroke:var(--s1)}.s2{fill:var(--s2);stroke:var(--s2)}
.mk{stroke:var(--card);stroke-width:2}
details.fold{background:var(--card);border:1px solid var(--line);border-radius:var(--r);
box-shadow:var(--shadow);margin:0 0 10px;scroll-margin-top:8px;overflow:clip}
details.fold>summary{cursor:pointer;list-style:none;padding:13px 16px;font:700 14.5px/1.3 var(--sans);
display:flex;gap:6px 10px;align-items:center;flex-wrap:wrap}
details.fold>summary:hover{background:var(--surface-2)}
details.fold>summary::-webkit-details-marker{display:none}
details.fold>summary::before{content:"▸";color:var(--faint);transition:transform .15s}
details.fold[open]>summary::before{transform:rotate(90deg)}
details.fold>summary .rt{margin-left:auto;font:600 13px/1.3 var(--mono);color:var(--mut)}
details.fold>summary .badge{margin-right:0}
details.fold>.in{padding:0 16px 12px;border-top:1px solid var(--line2)}
details.fold>.in>.wrap,details.fold>.in>.tiles .tile,details.fold>.in>.gr-tablewrap{box-shadow:none}
details.fold>.in>.gr-tablewrap:first-child{border-width:0;border-radius:0;margin:0 -16px}
.foot{display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px 18px;margin-top:44px;padding-top:16px;
border-top:1px solid var(--line);color:var(--faint);font-size:12.5px;line-height:1.6}
.foot a{color:var(--muted)}.foot code{font-size:12px}.foot .r{text-align:right}
@media (max-width:720px){
.topbar{flex-wrap:wrap;gap:12px;padding:14px 0;margin-bottom:16px}
.nav{order:3;width:100%;margin-left:0}.nav a{flex:1;text-align:center;padding:9px 4px;font-size:13px}
.theme{margin-left:auto}.brand-version{display:none}
.gr-h1{font-size:28px}h1{font-size:26px}
.gr-summary{grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}
.gr-stat{padding:13px}.gr-stat .v{font-size:22px}
.ggrid{grid-template-columns:minmax(0,1fr)}
table{font-size:13px}td{font-size:12.5px}
table.gr{font-size:13px}table.gr th,table.gr td{padding:9px 10px;white-space:nowrap}
.g-detail table.gr td{white-space:normal}
details.fold>summary .rt{margin-left:0;width:100%}
.foot .r{text-align:left}
.uchart .ax{font-size:30px}.uchart .ln{stroke-width:4}.uchart .zero{stroke-width:2}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

# A link to a section inside a closed fold opens that fold (jump chips).
OPEN_JS = ("<script>(function(){function o(){var h=location.hash.slice(1);if(!h)return;"
           "var e=document.getElementById(h);for(;e;e=e.parentElement)"
           "if(e.tagName==='DETAILS')e.open=true;}"
           "addEventListener('hashchange',o);addEventListener('DOMContentLoaded',o);})();</script>")

# Today page: live and final scores from ESPN's public scoreboard, fetched by
# the reader's browser on each page load (display only: never written to
# data/ and never read by the model). Cards match on the ESPN event id
# (data-gid); a failed fetch leaves the page as built.
SCORES_JS = """<script>(function(){
var g=document.querySelector('[data-slate]');if(!g||!window.fetch)return;
var u='https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?limit=100&dates='
+g.getAttribute('data-slate').replace(/-/g,'');
function per(p){return p<=4?'Q'+p:p==5?'OT':(p-4)+'OT';}
fetch(u,{cache:'no-store'}).then(function(r){return r.ok?r.json():null;}).then(function(j){
var n=0;((j&&j.events)||[]).forEach(function(e){
var c=document.querySelector(".gcard[data-gid='"+e.id+"']");if(!c)return;
var st=e.status||{},ty=st.type||{},s={},txt,score=true;
(((e.competitions||[])[0]||{}).competitors||[]).forEach(function(t){s[t.homeAway]=t.score;});
if(ty.state==='in'){txt=ty.name==='STATUS_HALFTIME'?'Half':ty.name==='STATUS_END_PERIOD'?
'End '+per(st.period):ty.name==='STATUS_IN_PROGRESS'?per(st.period)+' '+st.displayClock:ty.shortDetail;}
else if(ty.state==='post'){txt=ty.shortDetail||'Final';score=!!ty.completed;}
else return;
var b=c.querySelector('.g-meta .st');b.textContent=txt;
b.className='badge st'+(ty.state==='in'?' live':'');
if(score&&s.away!=null&&s.home!=null)c.querySelectorAll('.sc').forEach(function(x){
x.textContent=s[x.getAttribute('data-side')];});
c.setAttribute('data-state',ty.state==='in'?'live':'final');n++;});
var a=document.querySelector('[data-scores-asof]');
if(n&&a){a.querySelector('time').textContent=new Date().toLocaleTimeString('en-US',
{timeZone:'America/Los_Angeles',hour:'numeric',minute:'2-digit'})+' PT';a.hidden=false;}
if(n&&window.nbaCount)window.nbaCount();
}).catch(function(){});})();</script>"""

THEME_JS = ("<script>(function(){var k='nba-theme',r=document.documentElement;"
            "try{var s=localStorage.getItem(k);if(s)r.setAttribute('data-theme',s);}catch(e){}"
            "window.toggleTheme=function(){var d=r.getAttribute('data-theme')||"
            "(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');"
            "var n=d==='dark'?'light':'dark';r.setAttribute('data-theme',n);"
            "try{localStorage.setItem(k,n);}catch(e){}};})();</script>")

SITE_NAME = "NBA Composite"
ASSETS = Path(__file__).resolve().parent / "assets" / "fonts"
# The public site (the NFL site's layout): Picks, Results, Performance, and one
# Research tab that holds the research pages, which keep their content.
PUBLIC_PAGES = (("index.html", "Picks"), ("results.html", "Results"),
                ("performance.html", "Performance"))
RESEARCH_PAGES = (("grades.html", "Ledger"), ("market-calibration.html", "Calibration"),
                  ("model.html", "Model"), ("preseason.html", "Preseason"))
PAGES = PUBLIC_PAGES + RESEARCH_PAGES
THEME_BUTTON = (
    "<button class='theme' type='button' onclick='toggleTheme()' "
    "aria-label='Toggle colour theme' title='Toggle theme'><svg class='sun' "
    "viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' "
    "stroke-linecap='round' aria-hidden='true'><circle cx='12' cy='12' r='4'/>"
    "<path d='M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2"
    "M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4'/></svg><svg class='moon' viewBox='0 0 24 24' "
    "fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' "
    "aria-hidden='true'><path d='M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z'/>"
    "</svg></button>")


def model_version(tags=None):
    """'v8' from the active model tags: the highest version among them (the
    brand's version pill)."""
    import re
    vs = [int(m.group(1)) for t in (tags or ACTIVE_TAGS or [""])
          for m in [re.search(r"_v(\d+)$", str(t))] if m]
    return f"v{max(vs)}" if vs else ""


def font_face_css(assets=ASSETS):
    """The sibling sites' faces (Archivo, JetBrains Mono; OFL), inlined so a
    page is one file. Missing files fall back to the system stack."""
    out = []
    for family, fname, weight in (("Archivo", "archivo-subset.woff2", "100 900"),
                                  ("JetBrains Mono", "jetbrains-mono-subset.woff2",
                                   "400 800")):
        p = Path(assets) / fname
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            out.append(f"@font-face{{font-family:'{family}';font-style:normal;"
                       f"font-weight:{weight};font-display:swap;src:url(data:font/"
                       f"woff2;base64,{b64}) format('woff2-variations')}}")
    return "".join(out)


def esc(s):
    return html.escape(str(s))


def page(title, active, body, built):
    """One page in the shared shell: header (brand, Picks / Results /
    Performance / Research, theme), a sub-nav on research pages, footer."""
    cur = " aria-current='page'"
    research = active in {h for h, _ in RESEARCH_PAGES}
    tab = RESEARCH_PAGES[0][0] if research else active
    nav = "".join(f"<a href='{h}'{cur if h == tab else ''}>{t}</a>"
                  for h, t in PUBLIC_PAGES + (("grades.html", "Research"),))
    sub = ("<nav class='subnav' aria-label='Research pages'><span class='lbl'>"
           "Research</span>" + "".join(f"<a href='{h}'{cur if h == active else ''}>{t}</a>"
                                       for h, t in RESEARCH_PAGES) + "</nav>"
           if research else "")
    ver = model_version()
    research_links = " · ".join(f"<a href='{h}'>{t}</a>" for h, t in RESEARCH_PAGES)
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><meta name='description' content='NBA "
            "composite model picks: win probabilities against the market, "
            "graded flat 1u against its null'>"
            f"{THEME_JS}<style>{font_face_css()}{CSS}</style></head><body>"
            "<a class='skip-link' href='#main-content'>Skip to content</a>"
            "<div class='mx-wrap'><header class='topbar'><a class='brand' href='index.html'>"
            f"<span class='brand-mark' aria-hidden='true'>NBA</span>{SITE_NAME}"
            + (f"<span class='brand-version'>{esc(ver)}</span>" if ver else "")
            + f"</a><nav class='nav' aria-label='Main navigation'>{nav}</nav>"
            f"{THEME_BUTTON}</header><main id='main-content'>{sub}{body}</main>"
            "<footer class='foot'><div>Built "
            f"<span class='stamp'>{esc(built)}</span>. Predicted margins and win "
            "probabilities from prior-game four factors, roster talent and the NBA "
            "injury report; the market is a benchmark only and never enters the model. "
            "Not betting advice.</div><div class='r'>Research: "
            f"{research_links} · <a href='{report.REPORT_NAME}'>{report.REPORT_NAME}</a>"
            f"</div></footer></div>{OPEN_JS}</body></html>")


def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def pp(x, d=1, cls=True):
    if x is None or not np.isfinite(x):
        return "—"
    if round(100 * x, d) == 0:
        x = 0.0
    s = f"{100 * x:+.{d}f}"
    if not cls:
        return s
    return f"<span class='{'pos' if x > 0 else 'neg' if x < 0 else ''}'>{s}</span>"


def se_txt(x):
    """A standard error in points, or a dash when undefined (n = 1)."""
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.1f}"


def ml_txt(x):
    try:
        v = int(float(x))
    except (TypeError, ValueError):
        return "—"
    return f"{v:+d}"


def tile(label, value, sub="", cls=""):
    return (f"<div class='tile {cls}'><div class='l'>{label}</div>"
            f"<div class='v'>{value}</div><div class='s'>{sub}</div></div>")


def table(heads, rows, left=(0,), key=()):
    """`key` marks the columns a reader should look at first (shaded)."""
    def c(i):
        return " ".join(x for x, on in (("l", i in left), ("k", i in key)) if on)
    th = "".join(f"<th class='{c(i)}'>{h}</th>" for i, h in enumerate(heads))
    body = "".join(
        "<tr>" + "".join(f"<td class='{c(i)}'>{v}</td>" for i, v in enumerate(r))
        + "</tr>" for r in rows)
    return f"<div class='wrap'><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"


# Team logos hotlinked from ESPN's CDN (Basketball-Reference code -> ESPN code
# where they differ), as on the NFL site: readers' browsers fetch them, the
# build needs no network, and a missing logo falls back to the code.
ESPN_CODES = {"BRK": "bkn", "CHO": "cha", "GSW": "gs", "NOP": "no", "NYK": "ny",
              "PHO": "phx", "SAS": "sa", "UTA": "utah", "WAS": "wsh"}
LOGO_URL = "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nba/{variant}/{code}.png&h=80&w=80"


def logo_html(code):
    """Light- and dark-theme logo over the team code; CSS shows the one
    matching the theme."""
    if code not in TEAM_NAMES:
        return ""
    espn = ESPN_CODES.get(code, str(code).lower())
    return "".join(
        f"<img class='logo {cls}' src='{esc(LOGO_URL.format(variant=v, code=espn))}' "
        "alt='' loading='lazy' decoding='async' referrerpolicy='no-referrer' "
        "onload=\"this.parentNode.classList.add('has-logo')\" onerror='this.remove()'>"
        for cls, v in (("lt", "500"), ("dk", "500-dark")))


def fold(summary, inner, open_=False, id_="", label=""):
    """A collapsible section (the NFL site's fold): summary line always
    visible, the detail one tap away."""
    attrs = (f" id='{id_}'" if id_ else "") + (f" aria-label='{esc(label)}'" if label else "")
    return (f"<details class='fold'{attrs}{' open' if open_ else ''}>"
            f"<summary>{summary}</summary><div class='in'>{inner}</div></details>")


def tip_pt(s):
    """A UTC time (a tip, a snapshot) as Pacific clock time, e.g. '4:00 PM PT'."""
    t = ledger.parse_utc(s)
    return t.astimezone(PT).strftime("%-I:%M %p PT") if t else "—"


def line_txt(x):
    """A spread line for one side: '-3.5', '+6.5', 'PK'."""
    return "PK" if x == 0 else f"{x:+.1f}"


def ats_note(r, a):
    """The card's account of the ATS hypothesis that covers this game
    (analysis.ats_card): the pick at its registered price, or why not."""
    key, side = a["hyp"]["key"], esc(a["side"])
    who = "model line"
    gap = (f"{who} {side} {line_txt(round(a['model_line'], 1))} vs market "
           f"{line_txt(a['line'])}: {a['edge']:.1f} pts")
    if not a["bet"]:
        return (f"<p class='note'><b>{key}</b> (pre-registered, against the "
                f"spread): no bet. At the first snapshot the {gap}; the rule "
                f"needs {a['hyp']['min_edge']:g}.</p>")
    now = pd.to_numeric(r.get("pre_spread"), errors="coerce")
    if np.isfinite(now):
        now = now if r["home"] == a["side"] else -now
    moved = (f" The line is now {line_txt(now)}." if np.isfinite(now)
             and now != a["line"] else "")
    return (f"<p class='note'><b>{key} pick</b> (pre-registered, against the "
            f"spread): <b>{side} {line_txt(a['line'])}</b> at {ml_txt(a['ml'])}"
            f"{book_tag(r.get('first_book'))}, from the first snapshot "
            f"({tip_pt(r.get('first_snapshot_utc'))}): {gap}; "
            f"model cover "
            f"{pct(a['model_p'])} vs market {pct(a['q'])} (no-vig).{moved} "
            "Graded on the <a href='grades.html#hypotheses'>Ledger</a> at this "
            "price, win or lose.</p>")


def _be_ml(p):
    """A break-even probability as American odds (0.6 -> -150, 0.4 -> +150)."""
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def band_panel(part, key, kind):
    """A card's price-context panel (owner's request and decision, 2026-10-23;
    analysis.card_bands / card_band): how sides priced (moneyline) or lined
    (spread) like this card's did, in a window centred on its own break-even
    probability or line: the market on both sides of every graded DraftKings
    game at the close, the model's live picks this season and its rebuilt
    picks (hindsight), never pooled. Descriptive, not a forecast."""
    if not part:
        return ""
    ml = kind == "ml"
    title = ("Moneyline · games priced like this" if ml
             else "Spread · games lined like this")
    row = analysis.card_band(part, key, kind)
    if not row:
        return (f"<div class='band'><div class='band-t'>{title}</div><p>Too few graded "
                f"DraftKings sides at {'prices' if ml else 'lines'} near this one; no "
                "comparison.</p></div>")
    if ml:
        rng = (f"break-even {100 * row['lo']:.1f}–{100 * row['hi']:.1f}% · prices "
               f"{ml_txt(round(_be_ml(row['lo'])))} to {ml_txt(round(_be_ml(row['hi'])))}")
        span = f"±{100 * row['width']:.1f} pts of break-even"
    else:
        rng = f"lines {line_txt(round(row['lo'], 1))} to {line_txt(round(row['hi'], 1))}"
        span = f"±{row['width']:.1f} pts of line"
    rng += f" · this {'price' if ml else 'line'} {span}" + (
        " (widened to reach enough games)" if row["widened"] else "")
    won = "won" if ml else "covered"

    def rec_tile(name, r, extra=""):
        if not r:
            return (f"<div><small>{name}</small><strong>—</strong>"
                    "<span class='vs'>no graded picks yet</span></div>")
        wl = f"{r['w']}–{r['l']}" + (f"–{r['p']}" if r["p"] else "")
        return (f"<div><small>{name}</small><strong>{pct(r['actual'])}</strong>"
                f"<span class='vs'>{wl} · implied {pct(r['implied'])}{extra}</span></div>")
    m = row["market"]
    mk = (f"<div><small>Market</small><strong>{pct(m['actual'])}</strong>"
          f"<span class='vs'>{m['n']} sides · implied {pct(m['implied'])}</span></div>"
          if m else "<div><small>Market</small><strong>—</strong></div>")
    seasons = ", ".join(season_txt(x) for x in part.get("rebuilt_seasons", []))
    n_live, n_reb = part["n_market_games"]
    return (f"<div class='band'><div class='band-t'>{title}</div>"
            f"<div class='band-l'>{esc(rng)}</div>"
            "<div class='band-stats'>" + mk
            + rec_tile("Model · live", row["live"])
            + rec_tile("Model · rebuilt", row["rebuilt"],
                       f" · {seasons + ', ' if seasons else ''}hindsight")
            + f"</div><p class='band-n'>% {won} vs the no-vig implied, at "
            f"{'prices' if ml else 'lines'} in this window. Market: every side in it "
            f"from {n_live + n_reb} graded DraftKings games ({n_live} this season). "
            "Live and rebuilt picks are never pooled. Descriptive, not a forecast.</p></div>")


def spread_line(kind, a):
    """A card's spread line: the side and its line, the edge in points, and
    the model's line beside the market's at the side's price. An early pick
    (H6) names its model: the carryover P it is graded on, not the card's."""
    side = esc(a["side"])
    model = "Model"
    return (f"<div class='pick-lines'><span class='kind'>{kind}</span><span class='pk'>"
            f"{side} {line_txt(a['line'])}</span><span class='edge'>{a['edge']:.1f} "
            f"pts vs the line</span></div><div class='g-mk'><span>{model} {side} "
            f"{line_txt(round(a['model_line'], 1))}</span> · <span>Market "
            f"{side} {line_txt(a['line'])}</span> · <span>at {ml_txt(a['ml'])}"
            "</span></div>")


def spread_lean_note(r, a, ats):
    """The card's account of its spread lean (analysis.spread_lean): the
    model's side against the latest spread, not a graded pick."""
    side = esc(a["side"])
    gp = np.fmin(*(pd.to_numeric(r.get(c), errors="coerce") for c in ("gp_home", "gp_away")))
    graded = ("The graded spread pick is the pre-registered rule below." if ats else
              "The graded spread picks are H5 games 1–9 and games 10+" + (", and neither covers a team's first game." if gp == 0
                               else "."))
    src = ("the model's predicted margin" if a.get("own")
           else f"13.5 × Φ⁻¹(P), from its win probability")
    return (f"<p class='note'><b>Lean · spread</b>: <b>{side} {line_txt(a['line'])}</b>"
            f" at {ml_txt(a['ml'])}{book_tag(r.get('pre_book'))}, the latest "
            f"pregame line: model line {side} {line_txt(round(a['model_line'], 1))} "
            f"({src}) vs market {line_txt(a['line'])}, {a['edge']:.1f} pts; model cover "
            f"{pct(a['model_p'])} vs market {pct(a['q'])} (no-vig). The model's side "
            f"of the spread, refreshed each build until tip: shown, not graded. {graded}</p>")


def book_tag(book):
    """Visible label naming a sportsbook price's book; Kalshi prices (the
    Preseason page's only market) carry none."""
    if pd.isna(book) or book in ("", "kalshi"):
        return ""
    return f" <span class='basis'>{esc(market.BOOK_NAMES.get(book, book))}</span>"


# One colour rule for every summary figure on the ledger and calibration
# pages: colour marks a gap of at least SIG standard errors from the null
# (green above, red below); anything smaller stays plain. Sign alone never
# colours a figure.
SIG = 2.0


def sig_cls(z):
    if z is None or not np.isfinite(z) or abs(z) < SIG:
        return ""
    return "pos" if z > 0 else "neg"


def z_txt(z):
    return "—" if z is None or not np.isfinite(z) else f"{z:+.1f}"


def season_txt(season):
    """Ledger season (the year it ends) -> '2025-26'."""
    try:
        y = int(float(season))
    except (TypeError, ValueError):
        return "season ?"
    return f"{y - 1}-{y % 100:02d}"


def se4(x):
    """A score-difference SE to 4 dp, or a dash when undefined (n = 1)."""
    return "—" if x is None or not np.isfinite(x) else f"{x:.4f}"


def resolved_txt(z, better, worse):
    """Plain words for a gap already expressed in SEs."""
    if z is None or not np.isfinite(z):
        return "not enough games"
    if abs(z) < SIG:
        return f"not resolved (z = {z:+.1f})"
    return f"{better if z > 0 else worse} by {abs(z):.1f} SE"


# ------------------------------------------------------- public pages ------
# Picks (index.html), Results and Performance: the NFL site's layout, on the
# native forward rows only. The research pages (Ledger, Calibration, Model,
# Preseason) keep every table; nothing here replaces them.

def _head(eyebrow, h1, lead, right=""):
    return ("<div class='gr-head'><div>"
            f"<div class='eyebrow'>{eyebrow}</div><h1 class='gr-h1'>{h1}</h1>"
            f"<div class='gr-lead'>{lead}</div></div>{right}</div>")


def _stat(label, value, sub="", cls="", meter=None):
    """One headline tile; `meter` = (rate, break-even) draws the bar."""
    m = ""
    if meter and all(np.isfinite(meter)):
        a, be = meter
        m = (f"<div class='meter' role='img' aria-label='Win rate {pct(a)}, "
             f"break-even {pct(be)}'><i style='width:{100 * a:.1f}%'></i>"
             f"<b style='left:{100 * be:.1f}%'></b></div>")
    return (f"<div class='gr-stat'><div class='l'>{label}</div>"
            f"<div class='v {cls}'>{value}</div><div class='s'>{m}{sub}</div></div>")


def public_scope(native):
    """The public record (Picks, Results, Performance tiles): the latest
    native section, one closing book x season of analysis.market_view
    (sections never pool), its lean bets at the price the record uses (the
    pregame snapshot, else the close), and those rows. None before any
    graded native row with a close."""
    secs = [s for s in _sections(analysis.market_view(native), ledger.empty())
            if s["basis"] == "native" and s["book"]]
    if not secs:
        return None
    sec = max(secs, key=lambda s: (s["season"], len(s["h"])))
    price = "pre" if analysis.roi_summary(sec["h"], "pre") else "close"
    p = analysis.picks(sec["h"], price)
    p = p[p["rule"] == "lean"] if len(p) else p
    if not len(p):
        return None
    return dict(sec=sec, price=price, picks=p.reset_index(drop=True),
                row=analysis.roi_row("lean", p))


def _scope_txt(scope):
    s = scope["sec"]
    where = "pregame price" if scope["price"] == "pre" else "close"
    return f"{esc(market.BOOK_NAMES[s['book']])} {season_txt(s['season'])} · {where}"


def record_tiles(scope):
    """Record, win rate, units, ROI on the lean: the ROI beside its null and
    the market favourite on the same games; colour only at >= 2 SE."""
    if not scope:
        return ("<div class='gr-summary'>"
                + _stat("Record", "0–0", "No graded picks yet · written before "
                        "tip, frozen after")
                + _stat("Win rate", "—", "Break-even depends on each price")
                + _stat("Units", "0.00u", "Flat 1 unit on the model's lean")
                + _stat("ROI", "—", "Read against its null (about −4%, the "
                        "market's hold), not zero")
                + "</div>")
    r = scope["row"]
    c = sig_cls(analysis.z_vs_null(r))
    return ("<div class='gr-summary'>"
            + _stat("Record", f"{r['w']}–{r['l']}", f"W–L · {r['n']} bet{'' if r['n'] == 1 else 's'} · "
                    f"{_scope_txt(scope)}")
            + _stat("Win rate", pct(r["actual"]), f"Break-even {pct(r['breakeven'])} "
                    "at the prices taken", meter=(r["actual"], r["breakeven"]))
            + _stat("Units", f"{r['units']:+.2f}u", "Flat 1 unit per pick on the "
                    "model's lean", c)
            + _stat("ROI", f"{100 * r['roi']:+.1f}%", f"± {se_txt(r['roi_se'])} pp "
                    f"(1 SE) · null {100 * r['roi_null']:+.1f}% · market favourite "
                    f"{fav_txt(r)}", c)
            + "</div>")


def public_ats(native):
    """[(hypothesis, bets)] for the spread rules (H5, H6) on the latest
    native season, graded at their registered price (raw rows, as the
    Ledger's scoreboard reads them)."""
    if native is None or not len(native):
        return []
    season = pd.to_numeric(native["season"], errors="coerce").max()
    g = ledger.graded(native[pd.to_numeric(native["season"], errors="coerce") == season])
    out = []
    for hyp in analysis.active_hypotheses():
        if hyp.get("market") != "ats":
            continue
        d = analysis.hypothesis_picks(g, hyp, hyp["native"]) if len(g) else pd.DataFrame()
        out.append((hyp, d))
    return sorted(out, key=lambda t: not t[0]["early"])    # H6 (games 1-9) first


def spread_tiles(native):
    """One tile per spread rule: W-L-P, cover rate, ROI beside its null."""
    tiles = []
    for hyp, d in public_ats(native):
        games = "games 1–9" if hyp["early"] else "games 10+"
        label = f"Spread · {hyp['key']} · {games}"
        r = analysis.ats_row(hyp["key"], d) if len(d) else None
        if not r:
            tiles.append(_stat(label, "0–0", "No graded bets yet · first-snapshot "
                               "spread, edge ≥ 1 pt"))
            continue
        c = sig_cls(analysis.z_vs_null(r))
        tiles.append(_stat(
            label, f"{r['w']}–{r['l']}" + (f"–{r['push']}" if r["push"] else ""),
            f"Cover {pct(r['actual'])} (break-even {pct(r['breakeven'])}) · ROI "
            f"<span class='{c}'>{100 * r['roi']:+.1f}%</span> ± {se_txt(r['roi_se'])} "
            f"(1 SE) · null {100 * r['roi_null']:+.1f}% · spread favourite "
            f"{fav_txt(r)}", meter=(r["actual"], r["breakeven"])))
    return "<div class='gr-summary'>" + "".join(tiles) + "</div>" if tiles else ""


def units_svg(units, title="Running units", width=900):
    """Cumulative units, one point per graded pick, from a zero start."""
    if not len(units):
        return ""
    cum = np.r_[0.0, np.cumsum(np.asarray(units, float))]
    W, H, L, R, T, B = width, 260, 46, 14, 14, 30
    lo, hi = min(cum.min(), -1.0), max(cum.max(), 1.0)
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    def X(i):
        return L + (W - L - R) * i / max(len(cum) - 1, 1)

    def Y(v):
        return T + (H - T - B) * (hi - v) / (hi - lo)
    step = max(1, int(np.ceil((hi - lo) / 5)))
    g = []
    for v in range(int(np.ceil(lo / step)) * step, int(np.floor(hi)) + 1, step):
        g.append(f"<line class='grid-ln' x1='{L}' y1='{Y(v):.1f}' x2='{W - R}' "
                 f"y2='{Y(v):.1f}'/><text class='ax' x='{L - 8}' y='{Y(v) + 4:.1f}' "
                 f"text-anchor='end'>{v:+d}</text>".replace("+0<", "0<"))
    g.append(f"<line class='zero' x1='{L}' y1='{Y(0):.1f}' x2='{W - R}' y2='{Y(0):.1f}'/>")
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(cum))
    g.append(f"<polygon class='area' points='{X(0):.1f},{Y(0):.1f} {pts} "
             f"{X(len(cum) - 1):.1f},{Y(0):.1f}'/>")
    g.append(f"<polyline class='ln' points='{pts}'/>")
    if len(cum) <= 120:
        g += [f"<circle class='pt' cx='{X(i):.1f}' cy='{Y(v):.1f}' r='3'><title>"
              f"pick {i}: {v:+.2f}u</title></circle>" for i, v in enumerate(cum) if i]
    g.append(f"<text class='ax' x='{L}' y='{H - 6}'>Start</text><text class='ax' "
             f"x='{W - R}' y='{H - 6}' text-anchor='end'>Pick {len(cum) - 1}</text>")
    return (f"<div class='uchart'><div class='head'><span class='t'>{esc(title)}</span>"
            f"<span class='v'>{cum[-1]:+.2f}u</span></div><svg viewBox='0 0 {W} {H}' "
            f"role='img' aria-label='{esc(title)}: {cum[-1]:+.2f} units after "
            f"{len(cum) - 1} picks'>" + "".join(g) + "</svg></div>")


def spread_charts(native):
    """Running units for each spread rule (H6, H5), one chart each, never
    pooled, bets in tip order; an empty panel until a rule has a grade."""
    nat = (native.assign(game_id=native["game_id"].astype(str))[["game_id", "tip_utc"]]
           if native is not None and len(native) else None)
    parts = []
    for hyp, d in public_ats(native):
        games = "games 1–9" if hyp["early"] else "games 10+"
        title = f"Running units · {hyp['key']} spread · {games}"
        if not len(d):
            parts.append(f"<div class='uchart'><div class='head'><span class='t'>{title}"
                         f"</span></div><div class='empty'>Starts with the first graded "
                         f"{hyp['key']} bet.</div></div>")
            continue
        d = d.assign(game_id=d["game_id"].astype(str)).merge(
            nat, on="game_id", how="left").sort_values(["tip_utc", "game_id"])
        parts.append(units_svg(d["units"], title, width=560))
    return f"<div class='uchart-pair'>{''.join(parts)}</div>" if parts else ""


HOW_INDEX = (
    "<p class='note'><b>Lean</b> is the side the model gives ≥ 50%: the pick, "
    "one unit on its moneyline. <b>Model</b> is the composite's P(win); "
    "<b>market</b> is the sportsbook's price with its margin removed "
    "(DraftKings; the book is named on the card, and its moneyline and spread "
    "come from the same book). The record above grades the lean at the "
    "DraftKings price named under it. "
    "<b>Edge</b> is how far the lean beats the market's no-vig probability "
    "once the model's view is pulled most of the way back to the market: "
    f"only {analysis.SHRINK_W:.0%} of its disagreement with the price counts, "
    "the share past games bore out against the opening line (at the close, "
    "none did), so later in the day even that is generous. <b>EV</b> is what "
    "that edge makes of the posted price, shown beside its <b>null</b>: the "
    "EV if the market is exactly right, about minus the hold, not zero. Both "
    "are estimates, not verified edges; Results and Performance track whether "
    "they hold up.</p>"
    "<p class='note'><b>Composite Δ</b> is 100 × the standardised four-factor gap "
    "(ridge weights, home − away, in win-% points). One model scores every "
    "game, opening night included. <b>early season</b>: until a team's game "
    f"{nc.FADE_GAMES} its rating still blends in last season (weight "
    f"{nc.FADE_RHO:g} × (1 − games played ÷ {nc.FADE_GAMES})), and before its "
    "first game its roster is its preseason box scores. <b>injury report</b>: "
    "from each team's game 1 (its second game) the latest NBA injury report is folded in "
    "(Out/Doubtful players, weighted by last season's BPM). The model also "
    "grows the rating's weight through the season, discounts opponents' "
    "3-point shooting (mostly luck), adds roster talent (minutes share × last "
    "season's BPM) and each team's free-throw percentage; see "
    "<a href='model.html'>Model</a>. Prices are refreshed each build until tip "
    "and frozen after.</p>"
    "<p class='note'><b>Spread pick (H5)</b>: three first-snapshot tests, "
    "H5, H2·F and H4, each with separate games 1–9 and games 10+ rows. "
    "H5 uses 13.5 × Φ⁻¹(P), edge ≥ 1 point and main-line prices; the "
    "early segment uses the recorded primary model P. Stage rows are never "
    "pooled. H1, H6 and late-pregame H2/H3 are retired.</p>"
    "<p class='note'><b>Games priced like this</b> (an opened card): how often "
    "sides at prices near this game's moneyline (within "
    f"±{100 * analysis.CARD_WINDOW['ml']:.1f} points of its break-even "
    "probability), and at lines near its spread (within "
    f"±{analysis.CARD_WINDOW['ats']:g} point), won or covered against the no-vig "
    "probability their price implied; the window widens when too few games "
    f"fall in it (fewer than {analysis.CARD_MIN_SIDES} market sides). "
    "<b>Market</b>: both sides of every graded "
    "DraftKings game at its closing price or line. <b>Model · live</b>: this "
    "season's picks at the pregame price, as the record grades them. <b>Model · "
    "rebuilt</b>: the rebuilt seasons' picks at the close, hindsight (each "
    "season left out of its own fit, not a track record). The two model "
    "records are never pooled. These records describe the past; they are not a "
    "forecast for this game and not a filter.</p>"
    "<p class='note'><b>Lean · spread</b>: on every other card, the model's "
    "side against the latest spread (the side its line beats, by how many "
    "points), refreshed each build until tip. It shows the model's view of the "
    "spread; it is not graded. Taken on every game at the closing line it "
    "returned about the market's hold in hindsight (no edge after the vig; "
    "the retired spread value side), which is why only the registered H5 segments are graded.</p>")


def _team_row(r, code, side, p):
    """One team line of a card: logo, name, model WP, the side's moneyline
    over its spread (as recorded before tip), score (filled by live JS)."""
    picked = np.isfinite(p) and r["lean"] == code
    wp = (p if side == "home" else 1 - p) if np.isfinite(p) else np.nan
    pts = pd.to_numeric(r.get(f"{side}_pts"), errors="coerce")
    line = pd.to_numeric(r.get("pre_spread"), errors="coerce")
    line = line if side == "home" else -line
    small = esc(code) if side == "away" else f"@ {esc(code)}"
    return (f"<div class='g-team{' picked' if picked else ''}'>"
            f"<div class='tchip'><span>{esc(code)}</span>{logo_html(code)}</div>"
            f"<span class='nm'><span class='tn'>{esc(TEAM_NAMES.get(code, code))}</span>"
            f"<span class='sub'><small>{small}</small>"
            + ("<i class='lean-pill'>Lean</i>" if picked else "") + "</span></span>"
            f"<span class='wp'>{pct(wp, 0) if np.isfinite(wp) else '—'}</span>"
            f"<span class='px'>{ml_txt(r.get(f'pre_{side}_ml'))}"
            f"<small>{line_txt(line) if np.isfinite(line) else ''}</small></span>"
            f"<span class='sc' data-side='{side}'>{int(pts) if np.isfinite(pts) else ''}"
            "</span></div>")


# ESPN's statuses, most serious first; anything else sorts after them
INJURY_ORDER = {"Out": 0, "Doubtful": 1, "Questionable": 2, "Day-To-Day": 3,
                "Probable": 4}


def injury_block(r, injuries):
    """The game's ESPN injury list as recorded (data/nba_injuries.csv: the
    last pregame snapshot, frozen at tip), one list per team, most serious
    status first. Display only: the model does not read it. "" when no
    injury frame is given."""
    if injuries is None:
        return ""
    g = injuries[injuries["game_id"].astype(str) == str(r["game_id"])]
    if not len(g):
        return "<h3>Injuries</h3><p>No injury list was captured for this game.</p>"
    cols = []
    for code in (r["away"], r["home"]):
        t = g[g["team"] == code]
        listed = t[t["status"] != "NONE"]
        listed = listed.assign(o=listed["status"].map(INJURY_ORDER).fillna(len(INJURY_ORDER)))
        items = [f"<li><span class='nm'>{esc(x['name'])}"
                 + (f"<small>{esc(x['detail'])}</small>" if x["detail"] else "")
                 + f"</span><span class='badge'>{esc(x['status'])}</span></li>"
                 for _, x in listed.sort_values(["o", "name"], kind="stable").iterrows()]
        if not items:
            items = ["<li class='none'>None listed</li>" if len(t)
                     else "<li class='none'>Not captured</li>"]
        cols.append(f"<div><div class='inj-t'>{esc(TEAM_NAMES.get(code, code))}"
                    f"<small>{esc(code)}</small></div><ul>{''.join(items)}</ul></div>")
    snap = g["snapshot_utc"].max()
    return ("<h3>Injuries</h3><div class='inj'>" + "".join(cols) + "</div>"
            f"<p>ESPN's injury list as of {tip_pt(snap)}, refreshed each build "
            "until tip and frozen after. The model does not read it: from each "
            "team's game 1 (its second game) it reads the NBA's official injury report (Out "
            "and Doubtful count as out); a team's game 0 has no injury "
            "terms.</p>")


def game_card(r, injuries=None, bands=None):
    """One game (the NFL site's card): both teams with model WP, the lean
    and its price, the spread pick when H5 bets; tap for the model
    against the market and each team's injury list (`injury_block`). Reads
    only the row as recorded: the sportsbook's moneyline and spread (one
    book); no Kalshi price (owner's decision, 2026-10-21)."""
    def book_name(b):
        return market.BOOK_NAMES.get(b, b) if pd.notna(b) and b else None
    away, home = r["away"], r["home"]
    src = book_name(r.get("pre_book"))
    p = pd.to_numeric(r["p_home"], errors="coerce")
    q_home = pd.to_numeric(r["pre_q_home"], errors="coerce")
    won = pd.to_numeric(r.get("home_won"), errors="coerce")
    has_ml = np.isfinite(pd.to_numeric(r["pre_home_ml"], errors="coerce"))
    final = np.isfinite(won)
    status = (f"Final {int(r['away_pts'])}–{int(r['home_pts'])}" if final else "Pregame")
    flags, lines = [], []
    det = [f"<p><b>{esc(away)} @ {esc(home)}</b> · tip {tip_pt(r['tip_utc'])}</p>"]
    if not np.isfinite(p):
        lines.append("<div class='pick-lines'><span class='kind'>Lean</span>"
                     "<span class='pk na'>No pick: no model probability</span></div>")
        flags.append("<span class='badge'>No probability</span>")
        det.append("<p>No probability for this game: an input the model needs "
                   "was missing at the last build (rows written before v8, which "
                   "projects every game, skipped each team's first game).</p>")
    else:
        lean_home = r["lean"] == home
        ml = r["pre_home_ml"] if lean_home else r["pre_away_ml"]
        q = q_home if lean_home else 1 - q_home
        pl = float(r["p_lean"])
        dec = market.decimal_payout(ml) if has_ml else np.nan
        # the edge and EV count only analysis.SHRINK_W of the model's
        # disagreement with the price (display only; p_lean is as recorded)
        ps = float(analysis.shrink_p(pl, q)) if np.isfinite(q) else np.nan
        ev = ps * dec - 1 if np.isfinite(dec) and np.isfinite(ps) else np.nan
        ev0 = q * dec - 1 if np.isfinite(dec) and np.isfinite(q) else np.nan
        price = f" {ml_txt(ml)}" if has_ml else ""
        gap = (f"{pp(ps - q, cls=False)} pp edge" if np.isfinite(q)
               else "no price yet")
        lines.append(f"<div class='pick-lines'><span class='kind'>Lean · moneyline"
                     f"</span><span class='pk'>{esc(r['lean'])}{price}</span>"
                     f"<span class='edge'>{gap}</span></div>"
                     f"<div class='g-mk'><span>Model {pct(pl)}</span> · <span>Market "
                     f"{pct(q)}</span>"
                     + (f" · <span>EV {pp(ev, cls=False)}% (null "
                        f"{pp(ev0, cls=False)}%)</span>" if np.isfinite(ev0) else "")
                     + "</div>")
        tag = r.get("model_tag")
        early_row = (pd.notna(r["gp_home"])
                     and min(r["gp_home"], r["gp_away"]) < nc.MIN_GAMES)
        if early_row and tag in (MODEL_TAG_V8, MODEL_TAG_V8_AVAIL,
                                 MODEL_TAG_V9, MODEL_TAG_V9_AVAIL):
            flags.append("<span class='badge' title='The rating still blends in "
                         "last season; it fades out by game 10'>early season</span>")
        elif early_row:
            flags.append("<span class='badge'>early · carryover"
                         + (" + roster" if tag == MODEL_TAG_V7_EARLY else "")
                         + "</span>")
        if tag in AVAIL_TAGS:
            flags.append("<span class='badge'>injury report</span>")
        bk = esc(src) if src else "Market"
        # the line each side would get: the model's own margin (v9) or
        # 13.5 x Phi^-1(P) (earlier rows), beside the book's pregame line
        mh = pd.to_numeric(r.get("margin_home"), errors="coerce")
        own = np.isfinite(mh)
        if not own:
            mh = analysis.ATS_SIGMA * NormalDist().inv_cdf(min(max(p, 1e-6), 1 - 1e-6))
        ln = pd.to_numeric(r.get("pre_spread"), errors="coerce")
        trs = [["Model", pct(1 - p), pct(p)],
               [f"{bk} (no-vig)", pct(1 - q_home), pct(q_home)],
               [f"{bk} moneyline", ml_txt(r["pre_away_ml"]), ml_txt(r["pre_home_ml"])],
               ["Model spread" + ("" if own else " (from P)"),
                line_txt(round(float(mh), 1)), line_txt(round(-float(mh), 1))]]
        if np.isfinite(ln):
            trs.append([f"{bk} spread", line_txt(-ln), line_txt(ln)])
        det.append("<h3>Model vs market</h3>" + table(["", esc(away), esc(home)], trs,
                                                       left=(0,)))
        det.append("<div class='tiles'>"
                   + tile("Edge", f"{pp(ps - q, cls=False)} pp" if np.isfinite(q) else "—",
                          f"{esc(r['lean'])}: model {pct(pl)}, market {pct(q)}")
                   + tile("EV", f"{pp(ev, cls=False)}%" if np.isfinite(ev) else "—",
                          f"null {pp(ev0, cls=False) if np.isfinite(ev0) else '—'}% "
                          "if the market is right")
                   + tile("Composite Δ", f"{float(r['delta']):+.1f}",
                          "home − away, win-% points")
                   + "</div>")
        if bands and has_ml:
            det.append(band_panel(bands.get("ml"), market.implied(ml), "ml"))
    # one spread line per card: the registered H5 games 1–9 / 10+ pick when its rule
    # bets, else the model's side against the latest spread (display only)
    ats = analysis.ats_card(r)
    lean_sp = analysis.spread_lean(r) if np.isfinite(p) else None
    spread = []
    if ats and ats["bet"]:
        lines.append(spread_line(f"Spread pick · {ats['hyp']['key']}", ats))
    elif lean_sp:
        lines.append(spread_line("Lean · spread", lean_sp))
        spread.append(spread_lean_note(r, lean_sp, ats))
    if ats:
        spread.append(ats_note(r, ats))
    shown_sp = ats if ats and ats["bet"] else lean_sp
    if bands and shown_sp:
        spread.append(band_panel(bands.get("ats"), float(shown_sp["line"]), "ats"))
    if spread:
        det.append("<h3>Spread</h3>" + "".join(spread))
    det.append(injury_block(r, injuries))
    b2b = [x for x, f in ((away, r["away_b2b"]), (home, r["home_b2b"])) if f == 1]
    if b2b:
        flags.append(f"<span class='badge warn'>B2B {esc('/'.join(b2b))}</span>")
    has_sp = np.isfinite(pd.to_numeric(r.get("pre_spread"), errors="coerce"))
    note = (f"{esc(src)} moneyline{' and spread' if has_sp else ''} · refreshed "
            "each build until tip, frozen after" if src and has_ml else "No price yet")
    summary = ("<summary class='game-summary'><div class='g-meta'>"
               f"<span class='when'>{tip_pt(r['tip_utc'])}</span>"
               f"<span class='badge st'>{status}</span></div><div class='g-teams'>"
               "<div class='g-cols' aria-hidden='true'><span></span><span></span>"
               "<span>Model</span><span>Price</span><span></span></div>"
               + _team_row(r, away, "away", p) + _team_row(r, home, "home", p)
               + "</div>" + "".join(lines)
               + (f"<div class='g-flags'>{''.join(flags)}</div>" if flags else "")
               + f"<div class='g-note'>{note}</div></summary>")
    cls = "gcard picked-card" if np.isfinite(p) else "gcard"
    return (f"<details class='{cls}' data-gid='{esc(r['game_id'])}' "
            f"data-state='{'final' if final else 'upcoming'}'>{summary}"
            f"<div class='g-detail'>{''.join(det)}</div></details>")


FILTER_JS = """<script>(function(){var bar=document.querySelector('.seg[data-filter]');
if(!bar)return;bar.hidden=false;
function count(){var c={all:0,upcoming:0,live:0,final:0};
document.querySelectorAll('.gcard[data-state]').forEach(function(x){c.all++;c[x.getAttribute('data-state')]++;});
bar.querySelectorAll('button').forEach(function(b){var k=b.getAttribute('data-show');
b.querySelector('.ct').textContent=c[k];if(k==='live')b.hidden=!c.live;});}
window.nbaCount=count;count();
bar.addEventListener('click',function(e){var b=e.target.closest('button');if(!b)return;
var f=b.getAttribute('data-show');bar.querySelectorAll('button').forEach(function(x){
x.setAttribute('aria-pressed',x===b?'true':'false');});
document.querySelectorAll('.gcard[data-state]').forEach(function(c){
c.hidden=f!=='all'&&c.getAttribute('data-state')!==f;});});})();</script>"""


def render_index(led, today, built, model_ok, recon=None, injuries=None, now=None):
    """Picks: today's slate as cards, under the season's lean record; each
    card opens to its game's injury list when `injuries` is given. `now`
    (UTC, default the clock) tells a game that tipped without a probability
    from one the next build can still score."""
    scope = public_scope(led)
    # the cards read the raw rows: the sportsbook's moneyline and spread, one
    # book (owner's decision, 2026-10-20); no Kalshi price (2026-10-21)
    day_txt = pd.Timestamp(today).strftime("%A, %B %-d")
    right = ("<div class='rec'><div class='l'>Forecast updated</div>"
             f"<div class='mono'>{esc(built)}</div></div>")
    head = _head(f"{season_txt(season_for(today))} regular season",
                 f"Picks · {esc(day_txt)}",
                 "The model's side in every game (the lean, one unit on its "
                 "moneyline) against the market, and the pre-registered spread "
                 "pick when its rule bets. Prices refresh each build until tip "
                 "and freeze after. Open a game for the model against the market. "
                 f"Slate <span class='stamp'>{esc(today)}</span>; times are "
                 "Pacific (PT).<span data-scores-asof hidden> Scores from ESPN "
                 "as of <time></time>; reload the page to update them.</span>",
                 right)
    tiles = record_tiles(scope)
    how = fold("How these numbers are made", HOW_INDEX)
    if not model_ok:
        return page("NBA picks", "index.html", head + tiles +
                    "<div class='flag-note'>No fitted model in <code>model/</code> "
                    "yet. Run the <b>Fit model</b> workflow.</div>", built)
    day = led[led["slate_date"].astype(str) == today] if len(led) else led
    if not len(day):
        return page("NBA picks", "index.html", head + tiles +
                    "<div class='empty'>No regular-season games scored for this "
                    "date (off day, preseason, or the build ran after tip). "
                    "Exhibition games are tracked on the <a href='preseason.html'>"
                    "Preseason</a> research page.</div>" + how, built)
    no_p = pd.to_numeric(day["p_home"], errors="coerce").isna()
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    tipped = pd.to_datetime(day["tip_utc"], utc=True, errors="coerce") <= now
    n_abs, n_gone = int(no_p.sum()), int((no_p & tipped).sum())
    n_open = n_abs - n_gone
    missing = ("an input the model needs (last season's games, this season's box "
               "scores or a fitted model) was missing at the last build; the next "
               "build retries until tip")
    frozen = "a row is frozen at tip"
    if not n_gone:
        why = missing
    elif not n_open:
        why = (f"it tipped without one and stays without a pick ({frozen})" if n_gone == 1
               else f"they tipped without one and stay without a pick ({frozen})")
    else:
        why = (f"{n_gone} tipped without one and {'stays' if n_gone == 1 else 'stay'} "
               f"without a pick ({frozen}); for the other{'s' if n_open != 1 else ''}, "
               + missing)
    note = (f"<div class='flag-note'>{n_abs} of {len(day)} game"
            f"{'s' if len(day) != 1 else ''} {'have' if n_abs != 1 else 'has'} no "
            f"model probability: {why}.</div>" if n_abs else "")
    seg = ("<div class='seg' data-filter role='group' aria-label='Filter games' hidden>"
           + "".join(f"<button type='button' data-show='{k}' aria-pressed="
                     f"'{'true' if k == 'all' else 'false'}'>{t}<span class='ct'></span>"
                     "</button>" for k, t in (("all", "All"), ("upcoming", "Upcoming"),
                                               ("live", "Live"), ("final", "Final")))
           + "</div>")
    bands = analysis.card_bands(
        ledger.graded(analysis.market_view(led)),
        ledger.graded(analysis.market_view(recon)) if recon is not None and len(recon) else None)
    cards = [game_card(r, injuries, bands) for _, r in day.sort_values("tip_utc").iterrows()]
    return page("NBA picks", "index.html",
                head + tiles + note
                + f"<div class='toolbar'><h2 class='sec'>Today’s games</h2>{seg}</div>"
                + f"<div class='ggrid' data-slate='{esc(today)}'>{''.join(cards)}</div>"
                + FILTER_JS + SCORES_JS + how, built)


def _game_rows(native, scope):
    """The public game-by-game rows: the lean bet at the record's price, the
    result, and the spread pick (H5) when its rule bet the game."""
    sec = scope["sec"]["h"].assign(game_id=lambda d: d["game_id"].astype(str))
    p = scope["picks"].assign(game_id=lambda d: d["game_id"].astype(str))
    ats = {}
    for hyp, d in public_ats(native):
        for _, a in d.iterrows():
            ats[str(a["game_id"])] = (hyp["key"], a)
    rows = p.merge(sec[["game_id", "slate_date", "tip_utc", "away", "home",
                        "away_pts", "home_pts"]], on="game_id", how="left")
    return rows.sort_values(["slate_date", "tip_utc"], ascending=[False, True]), ats


def rebuilt_scope(rebuilt, native=None):
    """This season's rebuilt rows (data/nba_rebuilt.csv; owner's request
    2026-10-23): every graded game scored after the fact by the current
    model, at the pages' market close (DraftKings), the latest season. The
    lean at the close, the model's side of the closing spread (its own
    margin), proper scores and the margin's miss. Hindsight: never pooled
    with the native record. None without such rows."""
    if rebuilt is None or not len(rebuilt):
        return None
    h = analysis.with_close(ledger.graded(rebuilt))
    if not len(h):
        return None
    h = h[h["close_book"].astype(str) == analysis.HYPOTHESIS_BOOK]
    if not len(h):
        return None
    season = pd.to_numeric(h["season"], errors="coerce").max()
    h = h[pd.to_numeric(h["season"], errors="coerce") == season].copy()
    h["game_id"] = h["game_id"].astype(str)
    p = analysis.picks(h, "close")
    p = p[p["rule"] == "lean"] if len(p) else p
    ats = analysis.ats_side_picks(h, "close", own_margin=True)
    live = set()
    if native is not None and len(native):
        n = native[pd.to_numeric(native["p_home"], errors="coerce").notna()]
        live = set(n["game_id"].astype(str))
    # the model's margin: its own (v9) or, on logit rows, 13.5 x Phi^-1(P)
    nd = NormalDist()
    ph = pd.to_numeric(h["p_home"], errors="coerce").clip(1e-6, 1 - 1e-6)
    m = pd.to_numeric(h["margin_home"], errors="coerce")
    h["model_margin"] = m.where(m.notna(), [analysis.ATS_SIGMA * nd.inv_cdf(x) for x in ph])
    h["final_margin"] = (pd.to_numeric(h["home_pts"], errors="coerce")
                         - pd.to_numeric(h["away_pts"], errors="coerce"))
    return dict(h=h, season=season, book=analysis.HYPOTHESIS_BOOK,
                picks=p.reset_index(drop=True) if len(p) else p,
                row=analysis.roi_row("lean", p) if len(p) else None,
                ats=ats, ats_row=analysis.ats_row("spread", ats) if len(ats) else None,
                live=live, tags=sorted(h["model_tag"].dropna().astype(str).unique()))


REBUILT_LEAD = ("Every graded game this season, opening night included, scored "
                "<b>after the fact</b> by the current model ({ver}): features from games "
                "before each date and the NBA injury report published before tip, every "
                "fit on earlier seasons only, priced at the {where}. It shows how the "
                "model you see today would have read the games already played, including "
                "the ones the live model did not price. It is <b>hindsight, not the "
                "record</b>: the live record above was written before tip and is never "
                "pooled with it.")


def rebuilt_tiles(rs):
    """Lean at the close, the model's spread side at the close, the proper
    scores and the margin's miss, on the rebuilt rows; colour at >= 2 SE."""
    h, r, a = rs["h"], rs["row"], rs["ats_row"]
    tiles = []
    if r:
        c = sig_cls(analysis.z_vs_null(r))
        tiles.append(_stat("Lean · moneyline", f"{r['w']}–{r['l']}",
                           f"ROI <span class='{c}'>{100 * r['roi']:+.1f}%</span> ± "
                           f"{se_txt(r['roi_se'])} (1 SE) · null {100 * r['roi_null']:+.1f}% "
                           f"· market favourite {fav_txt(r)}"))
    if a:
        c = sig_cls(analysis.z_vs_null(a))
        tiles.append(_stat("Model's side · spread",
                           f"{a['w']}–{a['l']}" + (f"–{a['push']}" if a["push"] else ""),
                           f"cover {pct(a['actual'])} (break-even {pct(a['breakeven'])}) · "
                           f"ROI <span class='{c}'>{100 * a['roi']:+.1f}%</span> ± "
                           f"{se_txt(a['roi_se'])} (1 SE) · null {100 * a['roi_null']:+.1f}%",
                           meter=(a["actual"], a["breakeven"])))
    sc = analysis.scoring(h)
    if sc:
        zl = (-sc["d_logloss"] / sc["d_logloss_se"]
              if sc["d_logloss_se"] and np.isfinite(sc["d_logloss_se"]) else float("nan"))
        tiles.append(_stat("Log loss · model / market",
                           f"{sc['model']['logloss']:.3f} / {sc['market']['logloss']:.3f}",
                           f"model − market <span class='{sig_cls(zl)}'>{sc['d_logloss']:+.4f}"
                           f"</span> ± {se4(sc['d_logloss_se'])} (1 SE) · {sc['n']} games · "
                           f"{resolved_txt(zl, 'model better', 'market better')}"))
    fm, mm = h["final_margin"].to_numpy(float), h["model_margin"].to_numpy(float)
    cl = -pd.to_numeric(h["close_spread"], errors="coerce").to_numpy(float)
    ok = np.isfinite(fm) & np.isfinite(mm) & np.isfinite(cl)
    if ok.sum():
        d = np.abs(mm[ok] - fm[ok]) - np.abs(cl[ok] - fm[ok])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        z = -d.mean() / se if np.isfinite(se) and se > 0 else float("nan")
        tiles.append(_stat("Spread miss · model / close",
                           f"{np.abs(mm[ok] - fm[ok]).mean():.1f} / "
                           f"{np.abs(cl[ok] - fm[ok]).mean():.1f}",
                           f"mean points off the final margin · model − close "
                           f"<span class='{sig_cls(z)}'>"
                           f"{d.mean():+.1f}</span> ± "
                           f"{f'{se:.1f}' if np.isfinite(se) else '—'} pts (1 SE) · "
                           f"{int(ok.sum())} games"))
    return "<div class='gr-summary'>" + "".join(tiles) + "</div>"


def rebuilt_section(rs, detail=True):
    """The rebuilt section of Results (detail: the game-by-game table) and
    Performance (tiles only)."""
    head = ("<h2 class='sec' id='rebuilt'>This season, rebuilt after the fact "
            "<span class='badge'>hindsight</span></h2>")
    if not rs:
        return head + ("<div class='empty'>Appears once this season's played games "
                       "have been rebuilt (Backfill history, rebuild).</div>")
    ver = model_version(rs["tags"]) or "the current model"
    where = f"{esc(market.BOOK_NAMES[rs['book']])} close"
    out = [head, "<p class='sec-lead'>" + REBUILT_LEAD.format(ver=ver, where=where) + "</p>",
           rebuilt_tiles(rs)]
    if not detail:
        out.append("<p class='note'>Game by game on <a href='results.html#rebuilt'>"
                   "Results</a>.</p>")
        return "".join(out)
    h = rs["h"].sort_values(["slate_date", "tip_utc"], ascending=[False, True])
    lean = {str(g): r for g, r in rs["picks"].set_index("game_id").iterrows()} \
        if len(rs["picks"]) else {}
    ats = {str(r["game_id"]): r for _, r in rs["ats"].iterrows()} if len(rs["ats"]) else {}
    head_cols = ("Game", "Live pick", "Model", "Model spread", "Close", "Final",
                 "Lean", "Spread side")
    for i, (date, g) in enumerate(h.groupby("slate_date", sort=False)):
        trs = []
        for _, r in g.iterrows():
            gid = str(r["game_id"])
            ln, sp = lean.get(gid), ats.get(gid)
            home_line = pd.to_numeric(r["close_spread"], errors="coerce")
            fav = r["home"] if r["p_home"] >= 0.5 else r["away"]
            mline = -r["model_margin"] if fav == r["home"] else r["model_margin"]
            trs.append(
                f"<tr><td><span class='team-code'>{esc(r['away'])} @ {esc(r['home'])}"
                f"</span><br><span class='mut small'>{tip_pt(r['tip_utc'])}</span></td>"
                f"<td class='small'>{'yes' if gid in rs['live'] else 'no'}</td>"
                f"<td class='n'>{esc(fav)} {pct(max(r['p_home'], 1 - r['p_home']))}</td>"
                f"<td class='n'>{esc(fav)} {line_txt(round(float(mline), 1))}</td>"
                f"<td class='n'>{pct(r['close_q_home'])} home"
                + (f"<br><span class='mut small'>home {line_txt(home_line)}</span>"
                   if np.isfinite(home_line) else "") + "</td>"
                f"<td class='n'>{int(r['away_pts'])}–{int(r['home_pts'])}</td>"
                + (f"<td>{esc(ln['side'])} {_result_badge(int(ln['won']))} "
                   f"<span class='mono small'>{ln['units']:+.2f}u</span></td>" if ln is not None
                   else "<td>—</td>")
                + (f"<td>{esc(sp['side'])} {line_txt(sp['line'])} {_result_badge(sp['result'])}"
                   f" <span class='mono small'>{sp['units']:+.2f}u</span></td>" if sp is not None
                   else "<td>—</td>") + "</tr>")
        cols = "".join(f"<th{' class=n' if c in ('Model', 'Model spread', 'Close', 'Final') else ''}>"
                       f"{c}</th>" for c in head_cols)
        tbl = (f"<div class='gr-tablewrap'><table class='gr'><thead><tr>{cols}</tr></thead>"
               f"<tbody>{''.join(trs)}</tbody></table></div>")
        day = pd.Timestamp(date).strftime("%a %b %-d")
        out.append(fold(f"{esc(day)} · {len(g)} game{'s' if len(g) != 1 else ''} rebuilt",
                        tbl, open_=i == 0))
    out.append("<p class='note'>Model: the side the rebuilt model favoured and its win "
               "probability. Model spread: that side's line from the model's own predicted "
               "margin (v9; earlier rows 13.5 × Φ⁻¹(P)). Close: the no-vig home probability "
               "and the home line at the close. Lean: one unit on the favoured side at the "
               "closing moneyline; spread side: one unit on the model's side of the closing "
               "spread. Live pick: whether the live model wrote a probability before tip "
               "(opening night's games had none). ROI is read against its null, not zero; "
               f"± is 1 SE and colour marks only a gap of at least {SIG:g} SE.</p>")
    return "".join(out)


def _result_badge(r):
    return ("<span class='badge w'>W</span>" if r == 1 else
            "<span class='badge l'>L</span>" if r == 0 else "<span class='badge'>P</span>")


def render_results(native, built, today, rebuilt=None):
    """Results: every graded pick this season, by slate, newest first; then
    this season rebuilt after the fact (rebuilt_section), apart."""
    scope = public_scope(native)
    rb = rebuilt_section(rebuilt_scope(rebuilt, native))
    body = [_head(f"{season_txt(season_for(today))} season", "Results",
                  "Every graded pick this season, written before tip. The lean "
                  "is one unit on the model's side at the pregame price; spread "
                  "picks (H5) at their first-snapshot price. Read each ROI "
                  "against its <b>null</b> (about −4%, the market's hold), not zero."),
            record_tiles(scope), "<h2 class='sec'>Spread picks</h2>",
            "<p class='sec-lead'>The pre-registered spread rules, graded at the "
            "first snapshot's spread and price.</p>", spread_tiles(native),
            "<h2 class='sec'>Game by game</h2>"]
    if not scope:
        body.append("<div class='empty'>No graded picks yet. Results appear the "
                    "morning after the first games with a model probability.</div>")
        body.append(rb)
        return page("NBA results", "results.html", "".join(body), built)
    rows, ats = _game_rows(native, scope)
    for i, (date, g) in enumerate(rows.groupby("slate_date", sort=False)):
        w, n = int(g["won"].sum()), len(g)
        trs = []
        for _, r in g.iterrows():
            sp = ats.get(str(r["game_id"]))
            spread = ("—" if not sp else
                      f"{sp[0]} {esc(sp[1]['side'])} {line_txt(sp[1]['line'])} "
                      f"{_result_badge(sp[1]['result'])} {sp[1]['units']:+.2f}u")
            trs.append(
                f"<tr><td><span class='team-code'>{esc(r['away'])} @ {esc(r['home'])}"
                f"</span><br><span class='mut small'>{tip_pt(r['tip_utc'])}</span></td>"
                f"<td><b>{esc(r['side'])}</b> <span class='mono'>{ml_txt(r['ml'])}</span></td>"
                f"<td class='n'>{pct(r['model_p'])}</td><td class='n'>{pct(r['q'])}</td>"
                f"<td class='n'>{int(r['away_pts'])}–{int(r['home_pts'])}</td>"
                f"<td>{_result_badge(int(r['won']))}</td>"
                f"<td class='n'>{r['units']:+.2f}u</td><td class='small'>{spread}</td></tr>")
        head = "".join(f"<th{' class=n' if c in ('Model', 'Market', 'Final', 'Units') else ''}>"
                       f"{c}</th>" for c in ("Game", "Lean", "Model", "Market", "Final",
                                             "Result", "Units", "Spread pick"))
        tbl = (f"<div class='gr-tablewrap'><table class='gr'><thead><tr>{head}</tr>"
               f"</thead><tbody>{''.join(trs)}</tbody></table></div>")
        day = pd.Timestamp(date).strftime("%a %b %-d")
        body.append(fold(f"{esc(day)} · {w}–{n - w}<span class='rt'>"
                         f"{g['units'].sum():+.2f}u</span>", tbl, open_=i == 0))
    body.append(f"<p class='note'>Lean record: {_scope_txt(scope)}. "
                + _unpriced_txt(native, scope) + "The full record, both bases and "
                "every book, is on the <a href='grades.html'>Ledger</a>.</p>")
    body.append(rb)
    return page("NBA results", "results.html", "".join(body), built)


def _unpriced_txt(native, scope):
    """Why the live record has fewer games than were played: graded games
    in the record's season the live model wrote no probability for."""
    g = ledger.graded(native)
    g = g[pd.to_numeric(g["season"], errors="coerce") == scope["sec"]["season"]]
    n = int(pd.to_numeric(g["p_home"], errors="coerce").isna().sum())
    if not n:
        return ""
    return (f"{n} graded game{'s' if n != 1 else ''} this season had no live pick "
            "(the model then in use did not score them; opening night under v6); "
            "they are in the <a href='#rebuilt'>rebuilt</a> section below. ")


def market_scope(native):
    """The market view's games: this season's graded native rows with a
    close from the pages' market (DraftKings, analysis.HYPOTHESIS_BOOK),
    every one of them (no model probability needed), as market_games rows:
    dict(book, season, m), or None before the first such game."""
    if native is None or not len(native):
        return None
    g = ledger.graded(analysis.market_view(native))
    if not len(g):
        return None
    g = g[(g["close_book"].astype(str) == analysis.HYPOTHESIS_BOOK)
          & pd.to_numeric(g["close_q_home"], errors="coerce").notna()]
    if not len(g):
        return None
    season = pd.to_numeric(g["season"], errors="coerce").max()
    g = g[pd.to_numeric(g["season"], errors="coerce") == season]
    m = analysis.market_games(g)
    return dict(book=analysis.HYPOTHESIS_BOOK, season=season, m=m) if len(m) else None


def _gap_cell(x, se, n, scale=100, d=1, unit=""):
    """'+3.1 ± 2.0' coloured at >= 2 SE; no SE below analysis.BAND_MIN_N."""
    if x is None or not np.isfinite(x):
        return "—"
    if round(scale * x, d) == 0:
        x = 0.0
    if n < analysis.BAND_MIN_N or se is None or not np.isfinite(se) or se <= 0:
        return f"{scale * x:+.{d}f}{unit}"
    return (f"<span class='{sig_cls(x / se)}'>{scale * x:+.{d}f}{unit}</span> "
            f"<span class='mut'>± {scale * se:.{d}f}</span>")


def _market_performance(native):
    """Performance: the market alone (owner's request, 2026-10-22). How
    DraftKings' closing prices did this season on every graded native game:
    favourites straight up, at the moneyline and against the spread, home
    sides against the spread, the closing line against the final margin,
    and the same by price band, by spread size and by side. No model
    column is read; never the rebuilt rows."""
    head = "<h2 class='sec'>Market performance</h2>"
    ms = market_scope(native)
    if not ms:
        return head + ("<div class='empty'>Starts with the first graded game with "
                       f"a {esc(market.BOOK_NAMES[analysis.HYPOTHESIS_BOOK])} "
                       "close.</div>")
    m = ms["m"]
    where = f"{esc(market.BOOK_NAMES[ms['book']])} {season_txt(ms['season'])} close"
    sm = analysis.market_summary(m)
    su, ats, hats, line = sm["su"], sm["ats"], sm["home_ats"], sm["line"]

    def z(x, se):
        return x / se if se is not None and np.isfinite(se) and se > 0 else float("nan")
    out = [head, f"<p class='note'>How the {where} did on its own, on every graded "
           f"game this season ({su['n']} games), whatever the model said.</p>",
           "<div class='gr-summary'>",
           _stat("Favourites straight up", f"{su['w']}–{su['l']}",
                 f"won {pct(su['actual'])} vs {pct(su['implied'])} implied · "
                 f"{pp(su['diff'], cls=False)} ± {100 * su['se']:.1f} (1 SE) · "
                 f"{resolved_txt(z(su['diff'], su['se']), 'won more than priced', 'won less than priced')}",
                 sig_cls(z(su["diff"], su["se"]))),
           _stat("Favourites · moneyline ROI", f"{100 * su['fav_roi']:+.1f}%",
                 f"± {se_txt(su['fav_se'])} (1 SE) · null {100 * su['fav_null']:+.1f}% · "
                 f"underdogs {100 * su['dog_roi']:+.1f}% ± {se_txt(su['dog_se'])} "
                 f"(null {100 * su['dog_null']:+.1f}%)",
                 sig_cls(z(su["fav_roi"] - su["fav_null"], su["fav_se"])))]
    if ats:
        out.append(_stat("Favourites against the spread",
                         f"{ats['w']}–{ats['l']}" + (f"–{ats['push']}" if ats["push"] else ""),
                         f"covered {pct(ats['rate'])} ± {se_txt(ats['se'])} (1 SE) · "
                         f"ROI {100 * ats['roi']:+.1f}% (null {100 * ats['null']:+.1f}%) · "
                         f"{resolved_txt(z(ats['rate'] - 0.5, ats['se']), 'favourites covered more', 'underdogs covered more')}",
                         sig_cls(z(ats["rate"] - 0.5, ats["se"]))))
        out.append(_stat("Home sides against the spread",
                         f"{hats['w']}–{hats['l']}" + (f"–{hats['push']}" if hats["push"] else ""),
                         f"covered {pct(hats['rate'])} ± {se_txt(hats['se'])} (1 SE) · "
                         f"{resolved_txt(z(hats['rate'] - 0.5, hats['se']), 'home covered more', 'road covered more')}",
                         sig_cls(z(hats["rate"] - 0.5, hats["se"]))))
    if line:
        zb = z(line["bias"], line["bias_se"])
        out.append(_stat("Closing line vs final margin", f"{line['mae']:.1f} pts",
                         f"mean miss (RMSE {line['rmse']:.1f}) · home beat the line by "
                         f"{line['bias']:+.1f} ± {line['bias_se']:.1f} pts (1 SE) · "
                         f"{resolved_txt(zb, 'home did better', 'road did better')}",
                         sig_cls(zb)))
    out.append("</div>")

    rows = []
    for label, r in analysis.market_ml_bands(m):
        rows.append([esc(label), r["n"], f"{r['w']}–{r['l']}", pct(r["actual"]),
                     pct(r["implied"]), _gap_cell(r["diff"], r["se"], r["n"]),
                     _gap_cell(r["fav_roi"] - r["fav_null"], r["fav_se"], r["n"]),
                     _gap_cell(r["dog_roi"] - r["dog_null"], r["dog_se"], r["n"])])
    out.append("<h3>Moneyline by the favourite's closing price</h3>"
               + table(["Favourite's price", "Games", "Fav W–L", "Fav won",
                        "Implied", "Won − implied (± 1 SE)",
                        "Fav ROI − null (± 1 SE)", "Dog ROI − null (± 1 SE)"],
                       rows, key=(5,)))
    rows = []
    for label, r in analysis.market_ats_bands(m):
        rows.append([esc(label.replace("Fav ", "")), r["n"],
                     f"{r['w']}–{r['l']}" + (f"–{r['push']}" if r["push"] else ""),
                     _gap_cell(r["rate"] - 0.5, r["se"], r["w"] + r["l"]),
                     _gap_cell(r["margin"], r["margin_se"], r["n"], scale=1, unit=" pts"),
                     _gap_cell(r["roi"] - r["null"], r["roi_se"], r["n"])])
    if rows:
        out.append("<h3>Against the spread by the favourite's closing line</h3>"
                   + table(["Favourite's line", "Games", "Fav W–L–P",
                            "Fav cover − 50% (± 1 SE)", "Fav margin vs line (± 1 SE)",
                            "Fav ATS ROI − null (± 1 SE)"], rows, key=(3,)))
    rows = []
    for label, r, a in analysis.market_by_side(m):
        rows.append([label, r["n"], f"{r['w']}–{r['l']}",
                     _gap_cell(r["diff"], r["se"], r["n"]),
                     (f"{a['w']}–{a['l']}" + (f"–{a['push']}" if a["push"] else "")) if a else "—",
                     _gap_cell(a["rate"] - 0.5, a["se"], a["w"] + a["l"]) if a else "—"])
    out.append("<h3>Home and road favourites</h3>"
               + table(["", "Games", "Straight up", "Won − implied (± 1 SE)",
                        "ATS W–L–P", "Cover − 50% (± 1 SE)"], rows, key=(3, 5)))
    out.append("<p class='note'>The favourite is the side the no-vig close makes more "
               "likely; on the spread, the side giving points. Won − implied is the "
               "favourites' win rate minus the no-vig probability the close gave them: "
               "0 means the price was right. ROI − null is 1 unit at the closing price "
               "against what that bet returns if the price is exactly right (about "
               "minus the hold), so 0 is a fair market, not a losing one. Margin vs "
               "line is how many points the favourite beat (+) or missed (−) the "
               "spread by. Bands are descriptive, not betting filters. ± is 1 SE, "
               f"not shown below {analysis.BAND_MIN_N} games; colour marks only a gap "
               f"of at least {SIG:g} SE. Market history for every season is on the "
               "<a href='market-calibration.html'>Calibration</a> page.</p>")
    return "".join(out)


def _forecast_section(scope):
    """Performance: the model's and the market's forecasts on the record's
    games (native rows, one closing book x season, public_scope): proper
    scores, calibration (slope, predicted vs actual, closing-line value), the
    reliability chart and a week-by-week table. Never the rebuilt rows; the
    Calibration page has every basis. Colour only at >= 2 SE (sig_cls)."""
    out = ["<h2 class='sec'>Forecasts: model vs market</h2>"]
    h = scope["sec"]["h"] if scope else None
    sc = analysis.scoring(h) if h is not None and len(h) else None
    if not sc:
        return "".join(out) + ("<div class='empty'>Starts with the first graded "
                               "game with a closing line.</div>")
    p = h["p_home"].to_numpy(float)
    q = h["close_q_home"].to_numpy(float)
    y = h["home_won"].to_numpy(float)
    where = f"{esc(market.BOOK_NAMES[scope['sec']['book']])} " \
            f"{season_txt(scope['sec']['season'])} close"

    def z_of(d, se):
        return d / se if se is not None and np.isfinite(se) and se > 0 else float("nan")

    zl = -z_of(sc["d_logloss"], sc["d_logloss_se"])
    zb = -z_of(sc["d_brier"], sc["d_brier_se"])
    out += ["<div class='gr-summary'>",
            _stat("Log loss · model / market",
                  f"{sc['model']['logloss']:.3f} / {sc['market']['logloss']:.3f}",
                  f"model − market <span class='{sig_cls(zl)}'>{sc['d_logloss']:+.4f}"
                  f"</span> ± {se4(sc['d_logloss_se'])} (1 SE) · "
                  f"{resolved_txt(zl, 'model better', 'market better')}"),
            _stat("Brier · model / market",
                  f"{sc['model']['brier']:.3f} / {sc['market']['brier']:.3f}",
                  f"model − market <span class='{sig_cls(zb)}'>{sc['d_brier']:+.4f}"
                  f"</span> ± {se4(sc['d_brier_se'])} (1 SE) · "
                  f"{resolved_txt(zb, 'model better', 'market better')}"),
            _stat("Picked the winner",
                  f"{pct(sc['model']['acc'])} / {pct(sc['market']['acc'])}",
                  f"model lean / market favourite · {sc['n']} games · accuracy is "
                  "not calibration"),
            "</div>"]

    def slope_stat(label, fit):
        sl, se = fit[0], fit[1]
        if not np.isfinite(sl):
            return _stat(label, "—", f"needs 30 graded games with both outcomes · "
                                     f"n={len(y)}")
        z = z_of(sl - 1, se)
        return _stat(label, f"{sl:.2f}", f"± {se:.2f} (1 SE) · 1 = calibrated, below "
                     f"1 too confident · {resolved_txt(z, 'too timid', 'too confident')}",
                     sig_cls(z))
    lc = analysis.lean_calibration(h)
    zc = z_of(lc["diff"], lc["se"])
    _, totals = analysis.market_calibration(h)
    fav = totals.get("favourite")
    cal = ["<div class='gr-summary'>",
           slope_stat("Calibration slope · model", analysis.calibration_slope(p, y)),
           slope_stat("Calibration slope · market", analysis.calibration_slope(q, y)),
           _stat("Model lean: won − stated", f"{pp(lc['diff'], cls=False)} pts",
                 f"won {pct(lc['actual'])} vs {pct(lc['predicted'])} stated · ± "
                 f"{100 * lc['se']:.1f} (1 SE) · "
                 f"{resolved_txt(zc, 'won more than stated', 'won less than stated')}",
                 sig_cls(zc))]
    if fav:
        zf = z_of(fav["diff"], fav["se"])
        cal.append(_stat("Market favourite: won − implied", f"{pp(fav['diff'], cls=False)} pts",
                         f"won {pct(fav['actual'])} vs {pct(fav['implied'])} implied · "
                         f"± {100 * fav['se']:.1f} (1 SE) · "
                         f"{resolved_txt(zf, 'won more than priced', 'won less than priced')}",
                         sig_cls(zf)))
    c = analysis.clv(h)
    if c:
        zv = z_of(c["mean"], c["se"])
        cal.append(_stat("Closing-line value · lean", f"{100 * c['mean']:+.2f} pp",
                         f"± {se_txt(c['se'])} (1 SE) · beat the close on "
                         f"{pct(c['beat'], 0)} · n={c['n']} · "
                         f"{resolved_txt(zv, 'market moved to the lean', 'market moved away')}",
                         sig_cls(zv)))
    out += cal + ["</div>"]
    out.append("<div class='charts'>"
               + reliability_svg([("Model", "s1", analysis.reliability(p, y)),
                                  ("Market (no-vig close)", "s2",
                                   analysis.reliability(q, y))],
                                 f"Model vs market calibration · {where}")
               + "<p class='note'>" + CHART_NOTE.replace("; the table below has them all",
                                                          "") + "</p></div>")
    wk = pd.to_datetime(h["slate_date"].astype(str))
    week = (wk - pd.to_timedelta(wk.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    trs = []
    for w, g in sorted(h.groupby(week.to_numpy()), reverse=True):
        sw = analysis.scoring(g)
        zw = -z_of(sw["d_logloss"], sw["d_logloss_se"])
        lw = analysis.lean_calibration(g)
        _, tw = analysis.market_calibration(g)
        fw = tw.get("favourite")
        start = pd.Timestamp(w)
        end = start + pd.Timedelta(days=6)
        span = (f"{start.strftime('%b %-d')}–{end.strftime('%-d')}"
                if end.month == start.month
                else f"{start.strftime('%b %-d')}–{end.strftime('%b %-d')}")
        trs.append(f"<tr><td>{span}</td><td class='n'>{sw['n']}</td>"
                   f"<td class='n'>{sw['model']['logloss']:.3f}</td>"
                   f"<td class='n'>{sw['market']['logloss']:.3f}</td>"
                   f"<td class='n'><span class='{sig_cls(zw)}'>{sw['d_logloss']:+.4f}</span>"
                   f" <span class='mut'>± {se4(sw['d_logloss_se'])}</span></td>"
                   f"<td class='n'>{pct(lw['actual'])} <span class='mut'>vs "
                   f"{pct(lw['predicted'])}</span></td>"
                   + (f"<td class='n'>{pct(fw['actual'])} <span class='mut'>vs "
                      f"{pct(fw['implied'])}</span></td>" if fw else "<td class='n'>—</td>")
                   + "</tr>")
    out.append("<h3>Forecasts by week</h3><div class='gr-tablewrap'><table class='gr'>"
               "<thead><tr><th>Week</th><th class=n>Games</th><th class=n>Model log "
               "loss</th><th class=n>Market log loss</th><th class=n>Model − market "
               "(± 1 SE)</th><th class=n>Lean won vs stated</th>"
               "<th class=n>Favourites won vs implied</th></tr></thead><tbody>"
               + "".join(trs) + "</tbody></table></div>")
    out.append(f"<p class='note'>Every game the record grades ({where}; native rows, "
               "written before tip): the model's P(home win) and the market's no-vig "
               "close on the same games. Lower log loss and Brier are better; model − "
               "market below 0 means the model did better. The calibration slope is "
               "the outcome's slope on each forecast's log-odds: 1 is calibrated, "
               "below 1 too confident. Closing-line value is how far the market moved "
               "toward the lean from the pregame snapshot to the close. Colour marks "
               f"only a gap of at least {SIG:g} SE (from 0, from a slope of 1, or "
               "from the stated probability). Every season and the rebuilt rows are "
               "on the <a href='market-calibration.html'>Calibration</a> page.</p>")
    return "".join(out)


def render_performance(native, built, today, rebuilt=None):
    """Performance: running units and same-game baselines for the lean."""
    scope = public_scope(native)
    body = [_head(f"{season_txt(season_for(today))} season", "Performance",
                  "How this season's leans are doing, beside simple baselines on "
                  "the same games at the same prices, and how the model's and the "
                  "market's probabilities score and calibrate on those games. Each "
                  "ROI is read against its null, the ROI if the market's no-vig "
                  "price were exactly right (about minus the hold), not zero."),
            record_tiles(scope)]
    if not scope:
        body.append("<div class='empty'>The running-units chart starts with the "
                    "first graded pick.</div>")
    else:
        p = scope["picks"]
        sec = scope["sec"]["h"].assign(game_id=lambda d: d["game_id"].astype(str))
        p = p.assign(game_id=p["game_id"].astype(str)).merge(
            sec[["game_id", "slate_date", "tip_utc"]], on="game_id", how="left"
        ).sort_values(["tip_utc", "game_id"])
        body.append(units_svg(p["units"], f"Running units · lean · {_scope_txt(scope)}"))
        r = scope["row"]

        def base(label, u):
            u = np.asarray(u, float)
            n = len(u)
            se = u.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            w = int((u > 0).sum())
            z = (u.mean() - r["roi_null"]) / se if np.isfinite(se) and se > 0 else float("nan")
            return _stat(label, f"{w}–{n - w}",
                         f"{u.sum():+.2f}u · ROI <span class='{sig_cls(z)}'>"
                         f"{100 * u.mean():+.1f}%</span> ± {se_txt(se)} (1 SE)")
        body += ["<h2 class='sec'>Model vs simple baselines</h2><div class='gr-summary'>",
                 _stat("Compared games", f"{r['n']}", f"{_scope_txt(scope)} · same "
                       "games, same prices"),
                 base("Model lean", p["units"]),
                 base("Market favourite", p["fav_units"]),
                 base("Home side", p["home_units"]),
                 "</div><p class='note'>The model's lean, always taking the market "
                 "favourite, and always taking the home side, each 1 unit at the "
                 f"same price. Null (market exactly right): "
                 f"{100 * r['roi_null']:+.1f}% for the lean. Colour marks only a "
                 f"gap of at least {SIG:g} SE from the null.</p>"]
        wk = pd.to_datetime(p["slate_date"].astype(str))
        p = p.assign(week=(wk - pd.to_timedelta(wk.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d"))
        trs = []
        for week, g in sorted(p.groupby("week"), reverse=True):
            rr = analysis.roi_row(week, g)
            c = sig_cls(analysis.z_vs_null(rr))
            start = pd.Timestamp(week)
            trs.append(f"<tr><td>{start.strftime('%b %-d')}–"
                       f"{(start + pd.Timedelta(days=6)).strftime('%b %-d')}</td>"
                       f"<td class='n'>{rr['w']}–{rr['l']}</td><td class='n'>"
                       f"{pct(rr['actual'])}</td><td class='n'>{rr['units']:+.2f}u</td>"
                       f"<td class='n'><span class='{c}'>{100 * rr['roi']:+.1f}%</span> "
                       f"<span class='mut'>± {se_txt(rr['roi_se'])}</span></td>"
                       f"<td class='n'>{100 * rr['roi_null']:+.1f}%</td></tr>")
        body.append("<h2 class='sec'>By week</h2><div class='gr-tablewrap'><table class='gr'>"
                    "<thead><tr><th>Week</th><th class=n>W–L</th><th class=n>Win %</th>"
                    "<th class=n>Units</th><th class=n>ROI (± 1 SE)</th>"
                    "<th class=n>Null</th></tr></thead><tbody>" + "".join(trs)
                    + "</tbody></table></div>")
    body.append(_market_performance(native))
    body.append(_forecast_section(scope))
    body += ["<h2 class='sec'>Spread picks</h2>", spread_tiles(native),
             spread_charts(native),
             rebuilt_section(rebuilt_scope(rebuilt, native), detail=False),
             "<p class='note'>Calibration on every season, the pre-registered "
             "hypotheses, the rebuilt seasons and the model's coefficients are on "
             "the <a href='grades.html'>Research</a> pages.</p>"]
    return page("NBA performance", "performance.html", "".join(body), built)


# ------------------------------------------------------------- model page ---
REPO_URL = "https://github.com/Dave356w/nba-matchups/blob/main/"
FEATURE_TXT = {
    "delta": "Four-factor composite gap, home − away (win-% points)",
    "b2b_net": "Back-to-back, away − home (1 = away on a b2b)",
    "d_phase": "delta × season phase (days since opening night ÷ season length)",
    "luck_def": "Opponents' 3P% regressed to the league (luck), home − away",
    "talent_diff": "Roster talent: minutes share × last-season BPM, home − away",
    "ft_diff": "Own FT% gap (decayed FTM/FTA), home − away",
    "av_min": "Minutes-weighted share of the rotation listed Out/Doubtful",
    "av_bpm": "BPM-weighted value of players listed Out/Doubtful",
}


def _coef_table(name, m):
    if not m:
        return fold(f"{esc(name)} <span class='mut'>not fitted</span>", "")
    rows = [["Intercept (home edge)", f"{m['intercept']:+.4f}", ""]]
    rows += [[f"<code>{esc(f)}</code>", f"{c:+.4f}",
              f"<span class='mut'>{esc(FEATURE_TXT.get(f, ''))}</span>"]
             for f, c in zip(m["features"], m["coef"])]
    yrs = m.get("years") or []
    meta = (f"{m.get('n_games', '?')} games · seasons ending "
            f"{min(yrs)}–{max(yrs)}" if yrs else f"{m.get('n_games', '?')} games")
    return fold(f"{esc(name)} <span class='mut'>{meta}</span>",
                table(["Term", "Logit coef", "Meaning"], rows, left=(0, 2)))


def _season_table(rows, basis, rule):
    out = []
    for row in rows:
        r, s = row[rule], row["scoring"]
        if not r:
            continue
        z = analysis.z_vs_null(r)
        c = sig_cls(z)
        ll = (f"{s['d_logloss']:+.4f} <span class='mut'>± {se4(s['d_logloss_se'])}</span>"
              if s else "—")
        badge = ("<span class='badge native'>Native</span>" if basis == "native"
                 else "<span class='badge recon' title='Reconstructed: hindsight'>"
                 "Recon</span>")
        out.append([f"{badge}{esc(market.BOOK_NAMES[row['book']])} "
                    f"{season_txt(row['season'])}", r["n"], f"{r['w']}–{r['l']}",
                    f"{r['units']:+.2f}",
                    f"<span class='{c}'>{100 * r['roi']:+.1f}%</span> "
                    f"<span class='mut'>± {se_txt(r['roi_se'])}</span>",
                    pct(r["q"]), pp(r["actual"] - r["q"], cls=False),
                    f"{100 * r['roi_null']:+.1f}%", f"{r['fav_units']:+.2f}",
                    fav_txt(r), ll])
    return out


def render_model(native, recon, built):
    """What the model is and how each season scored: the NFL site's Model
    page, on this project's rows (never pooled across basis, book, season)."""
    native, recon = analysis.market_view(native), analysis.market_view(recon)
    _w, base = load_model()
    avail, early = load_avail(), load_early()
    body = ["<h1>Model</h1><p class='lead'>Goal: flat 1u ROI on the model's "
            "side, judged against the market-correct null and the market "
            "favourite on the same games; log loss and Brier against the close "
            f"are the probability check. Full description: <a href='{REPO_URL}"
            f"MODEL.md'>MODEL.md</a>; the quotable readout: <a href='"
            f"{report.REPORT_NAME}'>{report.REPORT_NAME}</a>.</p>"
            "<p class='note'>Active tags: "
            + " / ".join(f"<code>{esc(t)}</code>" for t in ACTIVE_TAGS) + "</p>"]
    if is_v8(base):
        body.append(
            "<h2>Routing</h2><p class='note'>One model for every game, opening night "
            f"included. Each team's rating blends in last season at {nc.FADE_RHO:g} × "
            f"(1 − games played ÷ {nc.FADE_GAMES}), so it fades out by game "
            f"{nc.FADE_GAMES}; before a team's first game its roster is its "
            "preseason box scores.</p>" + table(
                ["Game", "Formula", "Model file"],
                [["The NBA injury report covers it (from each team's game 1, its second game)",
                  "availability logit", "<code>model/logit_avail.json</code>"],
                 ["Otherwise (game 0 included)", "base logit",
                  "<code>model/logit.json</code>"],
                 [f"A term missing, before game {nc.FADE_GAMES}",
                  f"carryover logit (last season × {cold_start.RHO} carried in)",
                  "<code>model/logit_early_v6.json</code>"],
                 [f"A term missing, game {nc.FADE_GAMES}+", "frozen v4 base logit",
                  "<code>model/logit_v4.json</code>"]], left=(0, 1, 2))
            + "<p class='note'>H1 and H6 (games 1–9) stay on the carryover logit "
            "they were registered on: every early row also records its probability.</p>")
        coefs = [_coef_table("Base logit (every game)", base),
                 _coef_table("Availability logit (games 1+, report covers)", avail),
                 _coef_table("Carryover logit (H1 / H6; fallback before game "
                             f"{nc.FADE_GAMES})", load_carry())]
    else:
        body.append("<h2>Routing</h2>" + table(
            ["Games played (fewer of the two)", "Formula", "Model file"],
            [["0", "abstain (no probability)", "—"],
             [f"1–{nc.MIN_GAMES - 1}", f"carryover logit: last season's log × "
              f"{cold_start.RHO} carried in"
              + (" + roster talent (v7)" if is_early_talent(early) else ""),
              "<code>model/logit_early.json</code>"],
             [f"{nc.MIN_GAMES}+, injury report covers the game", "availability logit",
              "<code>model/logit_avail.json</code>"],
             [f"{nc.MIN_GAMES}+, otherwise", "base logit", "<code>model/logit.json</code>"],
             [f"{nc.MIN_GAMES}+, a v5/v6 term missing", "frozen v4 base logit",
              "<code>model/logit_v4.json</code>"]], left=(0, 1, 2)))
        coefs = [_coef_table("Base logit (games 10+)", base),
                 _coef_table("Availability logit (games 10+, report covers)", avail),
                 _coef_table("Early-season logit (games 1–9" + (", carryover + roster talent)"
                             if is_early_talent(early) else ", carryover)"), early)]
    body.append("<h2>Fitted coefficients</h2><p class='note'>P(home) = σ(intercept "
                "+ Σ coef × term). Coefficients are fitted by log loss on "
                "earlier seasons; a change to them or to a term is a new model "
                "tag.</p>")
    body += coefs
    heads = ["Basis · close book · season", "Bets", "W–L", "Units", "ROI ± SE",
             "No-vig q", "Excess pp", "Mkt null", "Fav units", "Fav ROI",
             "LL model − close"]
    for rule, label in analysis.PICK_RULES:
        if rule not in analysis.SHOWN_RULES:
            continue
        rows = (_season_table(analysis.shown(analysis.season_rows(ledger.graded(native))),
                              "native", rule)
                + _season_table(analysis.shown(analysis.season_rows(ledger.graded(recon))),
                                "reconstructed", rule))
        body.append(f"<h2>Flat 1u ROI by season · {esc(label.split(' — ')[0].lower())} "
                    "at the close</h2>")
        body.append(table(heads, rows, left=(0,), key=(4,)) if rows else
                    "<p class='note'>No graded rows with a closing line yet.</p>")
    body.append(
        "<div class='key'><b>Flat 1u.</b> One unit on the picked side every game, at "
        "that side's closing moneyline. <b>No-vig q</b> is the market's probability "
        "for the picked side with the hold removed; <b>excess</b> is the win rate "
        "minus q. <b>Mkt null</b> is the ROI expected if the market were exactly "
        "right — negative by the hold, so beating it is the bar, not zero. "
        "<b>Fav</b> bets the market favourite on the same games. <b>LL model − "
        "close</b>: log-loss gap on the same games (negative = model better). ± is "
        "one standard error; colour marks ≥ 2 SE from the null. Reconstructed rows "
        "are hindsight (that season left out of the fit, graded at the close), never "
        f"forward evidence. {HIDDEN_NOTE}</div>")
    return page("NBA model", "model.html", "".join(body), built)

# ------------------------------------------------ ledger + calibration ---
BASIS_LABELS = {"native": "Native (pregame-locked, forward)",
                "reconstructed": "Reconstructed (leave-one-season-out, hindsight)"}
BASIS_BADGES = {"native": "<span class='badge native'>Native · forward</span>",
                "reconstructed": "<span class='badge recon'>Reconstructed · hindsight</span>"}
# The research pages, like the report, show nothing priced by a book in
# analysis.HIDDEN_BOOKS (ESPN BET, no longer offered; owner's decisions,
# 2026-10-21): no section, season row, market fold or spread. The Ledger
# grades the hypotheses on DraftKings (analysis.HYPOTHESIS_BOOK).
HIDDEN_NOTE = ("Games priced only by ESPN BET (all of 2024-25, and 2025-26 "
               "before 25 Nov), which is no longer offered, are left out of "
               f"these pages and the <a href='{report.REPORT_NAME}'>report</a>; "
               "the rows stay in "
               f"<a href='{REPO_URL}data/nba_reconstructed.csv'>"
               "data/nba_reconstructed.csv</a>.")

READ_KEY = (
    "<details class='key'><summary>How to read these numbers</summary> Judge every ROI against its "
    "<b>null</b>: the ROI if the market's no-vig prices were exactly right. "
    "That is about −4% (the hold: the bookmaker's margin), not zero. "
    "<b>Market</b> is DraftKings: forward (native) games at their pregame "
    "snapshot and close, rebuilt seasons at the close, the hypotheses at "
    "the prices they were registered on (ESPN BET is no longer offered, so "
    "this page shows nothing priced by it). "
    "<b>±</b> is one "
    "standard error (1 SE). Colour marks only gaps of at least "
    f"{SIG:g} SE from the null: <span class='pos'>green</span> above, "
    "<span class='neg'>red</span> below; everything else is left plain. "
    "Shaded columns are the ones to read first."
    "<details><summary>Glossary</summary><dl>"
    "<dt>Lean</dt><dd>The side the model gives ≥ 50% (as recorded on the row).</dd>"
    "<dt>Value side</dt><dd>The side where the model's probability beats the "
    "market's no-vig probability. Graded only through the pre-registered "
    "hypotheses; the blanket value bet is retired (see the note under the "
    "hypotheses).</dd>"
    "<dt>Model %</dt><dd>The picked side's mean model win probability.</dd>"
    "<dt>Market %</dt><dd>The picked side's mean no-vig market probability "
    "(q): the price with the bookmaker's margin removed.</dd>"
    "<dt>Win %</dt><dd>How often the picked side actually won.</dd>"
    "<dt>Break-even</dt><dd>The win rate the posted price needs to pay back; "
    "above q by about the hold.</dd>"
    "<dt>ROI</dt><dd>Profit per 1-unit bet.</dd>"
    "<dt>Null</dt><dd>The ROI expected if the market is right (q × payout − 1). "
    "<b>vs null</b> = ROI − null.</dd>"
    "<dt>z</dt><dd>(ROI − null) ÷ SE. Across many cells, about one in twenty "
    "shows |z| ≥ 2 by chance.</dd>"
    "<dt>EV</dt><dd>Win % − break-even; its null is q − break-even.</dd>"
    "<dt>Fav ROI</dt><dd>1u on the market favourite of the same games at the "
    "same price: the same-row baseline a pick rule has to beat (a tie at 50% "
    "goes to home).</dd>"
    "<dt>Log loss, Brier</dt><dd>Proper scores of the probabilities; lower is "
    "better. Compared with the close on the same games.</dd>"
    "<dt>Native</dt><dd>Written before tip and frozen: the forward test.</dd>"
    "<dt>Reconstructed</dt><dd>Scored afterwards with that season left out of "
    "the fit and graded at the close: hindsight, never forward evidence.</dd>"
    "</dl></details></details>")


def _slug(*parts):
    return "-".join(str(p).lower().replace(" ", "") for p in parts)


def _sections(native, recon):
    """One section per basis x closing book x season: never pooled.

    Each: basis, book, season, label (plain text), title (html), id, jump
    (short chip text) and h (graded rows with a close, analysis.with_close).
    """
    out = []
    for basis, df in (("native", native), ("reconstructed", recon)):
        h = analysis.with_close(ledger.graded(df))
        parts = []
        for book, hb in analysis.book_split(h):
            for season in sorted(hb["season"].dropna().unique(), reverse=True):
                parts.append((book, season, hb[hb["season"] == season]))
        if not parts:
            out.append(dict(basis=basis, book=None, season=None, h=h,
                            label=BASIS_LABELS[basis], id=basis,
                            title=BASIS_BADGES[basis], jump=basis.capitalize()))
            continue
        for book, season, hb in parts:
            bname, stxt = market.BOOK_NAMES[book], season_txt(season)
            out.append(dict(
                basis=basis, book=book, season=season, h=hb,
                label=f"{BASIS_LABELS[basis]} · {bname} close · {stxt}",
                id=_slug(basis, book, stxt),
                title=f"{BASIS_BADGES[basis]}{esc(bname)} close · {stxt}",
                jump=f"{'Native' if basis == 'native' else 'Recon'} · {bname} {stxt}"))
    return out


def _jump(items):
    return ("<nav class='jump' aria-label='Sections'>"
            + "".join(f"<a href='#{i}'>{esc(t)}</a>" for i, t in items) + "</nav>")


def _native_empty(native):
    n = len(native)
    if not n:
        return ("<p class='note'>No native rows yet. The daily build writes "
                "one before each tip, starting with the first regular-season "
                "slate; this section fills in as those games are graded.</p>")
    pending = int(pd.to_numeric(native["home_won"], errors="coerce").isna().sum())
    return (f"<p class='note'>{n} pregame row{'s' if n != 1 else ''} recorded, "
            f"{pending} waiting for a final score and closing line. Graded "
            "rows appear here, with the same verdicts and tables as the "
            "reconstructed seasons below.</p>")


def _roi_cells(r):
    """ROI, null and ROI − null (± 1 SE), coloured by one rule (sig_cls)."""
    z = analysis.z_vs_null(r)
    c = sig_cls(z)
    return [f"<span class='{c}'>{100 * r['roi']:+.1f}%</span>",
            f"{100 * r['roi_null']:+.1f}%",
            f"<span class='{c}'>{100 * (r['roi'] - r['roi_null']):+.1f}</span>"
            f" <span class='mut'>± {se_txt(r['roi_se'])}</span>"]


ROI_HEADS = ["n", "W–L", "ROI", "Null", "vs null (± 1 SE)", "Win %",
             "Break-even", "Model %", "Market %", "Fav ROI"]


def fav_txt(r):
    """The market favourite's flat ROI on the same games at the same price."""
    v = r.get("fav_roi", float("nan")) if r else float("nan")
    return "—" if v is None or not np.isfinite(v) else f"{100 * v:+.1f}%"


def _roi_table(sections):
    out = []
    for rule_label, rows in sections:
        out.append(f"<h3>{esc(rule_label)}</h3>")
        out.append(table(
            ["Picks"] + ROI_HEADS,
            [[esc(r["label"]), r["n"], f"{r['w']}–{r['l']}", *_roi_cells(r),
              pct(r["actual"]), pct(r["breakeven"]), pct(r["model_p"]), pct(r["q"]),
              f"<span class='mut'>{fav_txt(r)}</span>"]
             for r in rows], left=(0,), key=(3, 5)))
    return "".join(out)


def _bet_tile(label, r, badge=""):
    if not r:
        return tile(label, "—", "no bets")
    z = analysis.z_vs_null(r)
    return tile(
        f"{label}{badge}", f"{100 * r['roi']:+.1f}% ROI",
        f"vs null ({100 * r['roi_null']:+.1f}%): "
        f"{100 * (r['roi'] - r['roi_null']):+.1f} ± {se_txt(r['roi_se'])} — "
        f"{resolved_txt(z, 'above', 'below')}<br>"
        f"{r['w']}–{r['l']} · won {pct(r['actual'])}, break-even "
        f"{pct(r['breakeven'])}<br>market fav, same games: {fav_txt(r)}",
        sig_cls(z))


def _score_tile(s):
    """Model vs the close on log loss (negative Δ = model better)."""
    se = s["d_logloss_se"]
    z = -s["d_logloss"] / se if np.isfinite(se) and se > 0 else float("nan")
    words = resolved_txt(z, "Model better", "Market better")
    head, _, by = words.partition(" by ")
    if not by:
        head, by = "Not resolved", words.replace("not resolved ", "")
    else:
        by = f"by {by}"
    return tile("Probabilities vs the close (log loss)", head.capitalize(),
                f"{by} · model − market {s['d_logloss']:+.4f} ± {se4(se)} (1 SE)"
                f"<br>{s['model']['logloss']:.4f} vs {s['market']['logloss']:.4f}"
                f" · n={s['n']}", sig_cls(z))


def _verdicts(sec):
    """The section's three questions, answered in plain words."""
    h, native = sec["h"], sec["basis"] == "native"
    price = "pre" if native and analysis.roi_summary(h, "pre") else "close"
    summ = {label: rows[0] for label, rows in analysis.roi_summary(h, price)}
    lean = summ.get(analysis.PICK_RULES[0][1])
    at = "pregame price" if price == "pre" else "close"
    tiles = [_score_tile(analysis.scoring(h)),
             _bet_tile(f"Lean bets · 1u at the {at}", lean)]
    if native:
        c = analysis.clv(h)
        if c:
            z = c["mean"] / c["se"] if np.isfinite(c["se"]) and c["se"] > 0 else float("nan")
            tiles.append(tile("Closing-line value (lean)", f"{100 * c['mean']:+.2f} pp",
                              f"± {se_txt(c['se'])} (1 SE) · "
                              f"{resolved_txt(z, 'beat the close', 'lost to the close')}"
                              f"<br>beat the close on {pct(c['beat'], 0)} · n={c['n']}",
                              sig_cls(z)))
    return "<div class='tiles'>" + "".join(tiles) + "</div>"


def _hypotheses_by_season(native, recon):
    """H5 games 1–9 / 10+ split by basis x season on DraftKings prices
    (analysis.hypothesis_by_season); nothing is pooled."""
    rows = []
    for hyp, parts in analysis.hypothesis_by_season(
            native, recon, book=analysis.HYPOTHESIS_BOOK):
        for i, p in enumerate(parts):
            badge = BASIS_BADGES[p["basis"]].replace(" · forward", "").replace(
                " · hindsight", "")
            at = f"at the {analysis.PRICE_NAMES[p['price']]}"
            r = p["row"]
            if not r:
                where = ("no graded bets yet" if p["basis"] == "native"
                         else "no bets")
                rows.append([f"<b>{hyp['key']}</b>" if i == 0 else "",
                             f"{badge}<span class='mut'>{where}</span>", at,
                             "0", "—", "—", "—", "—", "—", "—", "—"])
                continue
            bname = market.BOOK_NAMES.get(p["book"], p["book"] if pd.notna(p["book"]) else "?")
            rows.append([f"<b>{hyp['key']}</b>" if i == 0 else "",
                         f"{badge}{esc(bname)} {season_txt(p['season'])}", at,
                         r["n"], f"{r['w']}–{r['l']}", *_roi_cells(r),
                         pct(r["actual"]), pct(r["breakeven"]), fav_txt(r)])
    if not rows:
        return ""
    return ("<h3>H5 games 1–9 / 10+ by season — never pooled</h3><p class='note'>Each "
            "basis and season stands alone, on DraftKings spreads (native: "
            "the first snapshot's; hindsight: the open's). Win % is the "
            "cover rate; fav is 1u on the spread favourite of the same games "
            "at the same price.</p>"
            + table(["", "Basis · book · season", "Price", "n", "W–L", "ROI",
                     "Null", "vs null (± 1 SE)", "Win %", "Break-even",
                     "Fav ROI"], rows, left=(0, 1, 2), key=(5, 7)))


def _dk_scope(recon):
    """'964 rebuilt 2025-26 games from 25 Nov 2025': the reconstructed rows
    the hypotheses are graded on (analysis.HYPOTHESIS_BOOK), in words."""
    g = analysis.on_book(recon, "close", analysis.HYPOTHESIS_BOOK)
    if g is None or not len(g):
        return "no rebuilt games"
    seasons = ", ".join(season_txt(x) for x in sorted(g["season"].dropna().unique()))
    first = pd.Timestamp(str(g["slate_date"].min())).strftime("%-d %b %Y")
    return f"{len(g):,} rebuilt {seasons} games from {first}"


def _backtest_row(label, price, r, move=None):
    if not r:
        return None
    mv = "—"
    if move:          # pp of no-vig probability, or spread points
        se = f"{move['se']:.2f}" if np.isfinite(move["se"]) else "—"
        mv = f"{move['mean']:+.2f} {move['unit']} <span class='mut'>± {se}</span>"
    return [label, price, r["n"], f"{r['w']}–{r['l']}", *_roi_cells(r),
            pct(r["actual"]), pct(r["q"]), mv, fav_txt(r)]


def _hypothesis_backtest(recon):
    """The rules' hindsight on DraftKings alone (analysis.hypothesis_backtest):
    at the open, at the close, and the open bets at the close price."""
    book = analysis.HYPOTHESIS_BOOK
    name = market.BOOK_NAMES[book]
    rows, halves, none = [], [], []
    for b in analysis.hypothesis_backtest(recon, book):
        hyp = b["hyp"]
        if not (b["open"] or b["close"]):
            none.append(b["label"])
            continue
        reg = lambda p: (" <span class='mut'>· registered</span>"  # noqa: E731
                         if hyp["hindsight"] == p else "")
        lab = f"<b>{esc(b['label'])}</b>"
        part = [_backtest_row(lab, f"at the open{reg('open')}", b["open"]),
                _backtest_row("", f"at the close{reg('close')}", b["close"]),
                _backtest_row("", "open bets at the close",
                              b["open_at_close"], b["move"])]
        part = [r for r in part if r]
        if part and not part[0][0]:
            part[0][0] = lab
        rows += part
        for i, (half, r) in enumerate(b["halves"]):
            row = _backtest_row(lab if i == 0 else "",
                                f"{half}, at the {analysis.PRICE_NAMES[hyp['hindsight']]}", r)
            if row:
                halves.append(row[:-2] + [row[-1]])
    if not rows:
        return ""
    heads = ["", "Price", "n", "W–L", "ROI", "Null", "vs null (± 1 SE)", "Win %",
             "Market %", "Line move", "Fav ROI"]
    missing = (f" {' and '.join(none)} bet only games 1–9, which {name} never "
               "priced in the rebuilt rows, so they have no backtest here."
               if none else "")
    return (f"<h3>{esc(name)} backtest — hindsight</h3><p class='note'>Every "
            f"rule on the {_dk_scope(recon)} that {esc(name)} priced, at its "
            "open and its close, and the same opening bets re-priced at the "
            "close. <b>Line move</b> is how far the market moved toward those "
            "bets by tip (no-vig points of probability, or spread points; "
            "± 1 SE); "
            "native bets are placed at the first snapshot, between the two. "
            f"H4 and H2·F are H3 and H2 at the first snapshot, which rebuilt "
            f"rows do not have, so in hindsight each pair is one set of bets."
            f"{missing} <b>Not an independent test:</b> these games were in "
            "the scans that chose the rules; the native rows are the test. "
            "Full tables: <code>diagnostics/dk_hypotheses/RESULTS.txt</code>.</p>"
            + table(heads, rows, left=(0, 1), key=(4, 6))
            + (fold("By half-season <span class='mut'>— descriptive, not a "
                    "filter</span>",
                    table(heads[:-2] + heads[-1:], halves, left=(0, 1), key=(4, 6)))
               if halves else ""))


def _hypothesis_ats_bands(native, recon):
    """H5 games 1–9 / 10+ by the picked side's spread and price, native beside the
    DraftKings hindsight (analysis.hypothesis_ats_bands): descriptive."""
    order = {"line": [b[2] for b in analysis.ATS_LINE_BANDS],
             "price": [b[2] for b in analysis.ATS_PRICE_BANDS]}
    out = []
    for hyp, parts in analysis.hypothesis_ats_bands(
            native, recon, book=analysis.HYPOTHESIS_BOOK):
        nat, hind = parts["native"], parts["hindsight"]
        if not any(nat[k] or hind[k] for k in order):
            continue
        for key, title in (("line", "the picked side's spread"),
                           ("price", "spread price")):
            by_n = {r["label"]: r for r in nat[key]}
            by_h = {r["label"]: r for r in hind[key]}
            rows = []
            for label in order[key]:
                if label not in by_n and label not in by_h:
                    continue
                cells = [esc(label)]
                for r in (by_n.get(label), by_h.get(label)):
                    cells += ([r["n"], pct(r["actual"]), *_roi_cells(r)[::2]]
                              if r else ["0", "—", "—", "—"])
                rows.append(cells)
            out.append(f"<h3>{hyp['key']} by {title}</h3>" + table(
                ["Band", "Native n", "Cover %", "ROI", "vs null (± 1 SE)",
                 "Hindsight n", "Cover %", "ROI", "vs null (± 1 SE)"],
                rows, left=(0,), key=(3, 4, 7, 8)))
    if not out:
        return ""
    h5 = next(x for x in analysis.HYPOTHESES if x["key"] == "H5")
    return fold(
        "H5 games 1–9 / 10+ by spread and price <span class='mut'>— descriptive, not a "
        "filter</span>",
        f"<p class='note'>The same bets as the scoreboard, split by the picked "
        f"side's line (negative = laying points) and its spread price: native "
        f"at the {analysis.PRICE_NAMES[h5['native']]}, hindsight on the "
        f"rebuilt DraftKings games at the {analysis.PRICE_NAMES[h5['hindsight']]}. "
        "The rules are not re-chosen from these cells; with a dozen of them a "
        "|z| near 2 somewhere happens by chance, and below "
        f"{analysis.BAND_MIN_N} bets a band shows no SE.</p>"
        + "".join(out))


def _hypotheses(native, recon):
    book = analysis.HYPOTHESIS_BOOK
    name = market.BOOK_NAMES[book]
    active, retired, archived_bets = [], [], []
    archive = analysis.retired_registration()
    for hyp, hind, nat in analysis.hypothesis_rows(native, recon, book=book):
        def cell(r):
            if not r:
                return ["0", "—", "—"]
            roi, null, vs = _roi_cells(r)
            return [str(r["n"]), f"{roi}<br><span class='mut'>null {null}</span>", vs]
        if hyp.get("retired"):
            r = archive[hyp["key"]]
            retired.append([f"<b>{hyp['key']}</b>", esc(hyp["rule"]),
                            r["registered_hindsight_text"],
                            str(len(r["native_bets"])), "0 · ungraded at retirement"])
            snapshots = {str(x["game_id"]): x for x in r["native_bets"]}
            for pick in r["native_picks"]:
                snap = snapshots[str(pick["game_id"])]
                archived_bets.append([hyp["key"], snap["slate_date"],
                                      f"{snap['away']} @ {snap['home']}", pick["side"],
                                      line_txt(pick["line"]) if "line" in pick else "—",
                                      ml_txt(pick["ml"]), "Ungraded at retirement"])
        else:
            h = cell(hind)
            h[0] += "<br><span class='mut'>at the open</span>"
            n = cell(nat)
            n[0] += "<br><span class='mut'>at the first snapshot</span>"
            active.append([f"<b>{hyp['key']}</b>", esc(hyp["rule"]),
                           *n, *h])
    early = []
    for hyp, r in analysis.early_hindsight(recon):
        early.append([f"<b>{hyp['key']}</b>", esc(hyp["rule"]), *cell(r)])
    return (
        "<h2 id='hypotheses'>Pre-registered hypotheses — the forward test</h2>"
        "<p class='note'>Three tests: H5, H2·F and H4, at the frozen first "
        "snapshot. Each has separate games 1–9 (·E) and games 10+ rows; "
        "never pooled. Early segments registered 2026-10-23 at 19:46:58 UTC, "
        "before CHI @ MIL was final. Established segments and every threshold "
        "are unchanged. Native is the forward test, on DraftKings prices. "
        "At the <b>first snapshot</b>, probabilities stay frozen. Hindsight is information only. Early v9 uses injury-report inputs "
        "and inherits the injury-timing caveat; the first snapshot is later "
        "than the open. Recorded first probabilities and tags stay frozen.</p>"
        + table(["", "Rule", "Native n · first snapshot", "ROI", "vs null (± 1 SE)",
                 f"Hindsight n ({esc(name)} open)", "ROI", "vs null (± 1 SE)"],
                active, left=(0, 1), key=(3, 4))
        + "<h3>Early segments · ESPN BET · hindsight only</h3>"
        "<p class='note'>276 reconstructed early games, v9 primary probabilities. "
        "Same registered rules at the open; no threshold selection. Separate "
        "from DraftKings and native forward evidence.</p>"
        + table(["", "Rule", "Hindsight n", "ROI", "vs null (± 1 SE)"], early,
                left=(0, 1))
        + "<p class='note'>Retired 2026-10-06: blanket value and its ATS twin.</p>"
        + "<h3>Retired</h3><p class='note'>H1/H6 carryover rules and late-pregame "
        "H2/H3 retired at registration. Their registered figures and native "
        "bets are archived; none was graded at retirement. No new bets or "
        "results enter an active record. Original registrations remain in "
        "CLAUDE.md; snapshots in diagnostics/first_snapshot_registration/retired.json.</p>"
        + table(["", "Rule", "Registered hindsight", "Native bets at retirement",
                 "Native graded n"], retired, left=(0, 1, 2))
        + fold("Archived native bets · ungraded", table(
            ["Rule", "Date", "Game", "Side", "Spread", "Price", "Status"],
            archived_bets, left=(0, 1, 2, 3, 6)))
        + _hypothesis_backtest(recon)
        + _hypotheses_by_season(native, recon)
        + _hypothesis_ats_bands(native, recon))


def _lean_line(h, price):
    """The fold's summary figure: flat 1u ROI on the lean beside its null
    and the market favourite on the same games (colour at >= 2 SE only)."""
    r = dict(analysis.roi_summary(h, price)).get(analysis.PICK_RULES[0][1])
    r = r and r[0]
    if not r:
        return ""
    c = sig_cls(analysis.z_vs_null(r))
    return (f"<span class='rt'>{r['w']}–{r['l']} · ROI <span class='{c}'>"
            f"{100 * r['roi']:+.1f}%</span> · null {100 * r['roi_null']:+.1f}% · "
            f"fav {fav_txt(r)}</span>")


def _ledger_body(sec, native):
    """A section's contents: verdicts, ROI tables, then the folded extras."""
    h, is_native = sec["h"], sec["basis"] == "native"
    if not len(h):
        return (_native_empty(native) if is_native else
                "<p class='note'>No graded rows with a closing line yet.</p>")
    how = ("Written before tip and frozen after" if is_native else
           "Scored afterwards with this season left out of the fit")
    body = [f"<p class='note'>{how} · {len(h)} graded games on "
            f"{h['slate_date'].nunique()} slates.</p>", _verdicts(sec)]
    _roi_section(body, h, is_native)
    _band_section(body, h, is_native)
    _ats_section(body, h)
    _recent_section(body, h, is_native)
    return "".join(body)


def render_grades(native, recon, built):
    # sections grade against Kalshi where a row has it; the hypotheses
    # (below, raw rows) stay on the sportsbook prices they were registered on
    secs = analysis.shown(_sections(analysis.market_view(native), analysis.market_view(recon)))
    body = ["<h1>Ledger</h1><p class='lead'>Every game the model scored, graded "
            "against the final score and the market: one unit on the model's "
            "lean each game. Judge each ROI against its <b>null</b> (about −4%, "
            "the market's hold), not zero, and against the market favourite on "
            "the same games. Forward rows and rebuilt history are never pooled.</p>",
            READ_KEY,
            _jump([("hypotheses", "Hypotheses")] + [(s["id"], s["jump"]) for s in secs]),
            _hypotheses(native, recon)]
    nat = [s for s in secs if s["basis"] == "native"]
    rec = [s for s in secs if s["basis"] != "native"]
    body.append("<h2 id='forward'>Forward record "
                "<span class='badge native'>Native · forward</span></h2>")
    for sec in nat:
        if sec["book"] is None:
            body.append(f"<div id='{sec['id']}' aria-label='{esc(sec['label'])}'>"
                        + _ledger_body(sec, native) + "</div>")
            continue
        price = "pre" if analysis.roi_summary(sec["h"], "pre") else "close"
        body.append(fold(f"{esc(market.BOOK_NAMES[sec['book']])} close · "
                         f"{season_txt(sec['season'])}" + _lean_line(sec["h"], price),
                         _ledger_body(sec, native), open_=True,
                         id_=sec["id"], label=sec["label"]))
    if rec:
        body.append("<h2 id='rebuilt'>Rebuilt history "
                    "<span class='badge recon'>Reconstructed · hindsight</span></h2>"
                    "<div class='flag-note'>The current model replayed on past "
                    "seasons, each season left out of its own fit and graded at the "
                    "close. <b>Hindsight, not a track record:</b> the model was "
                    f"designed after these games were played. {HIDDEN_NOTE}</div>")
    for sec in rec:
        if sec["book"] is None:
            body.append(f"<div id='{sec['id']}' aria-label='{esc(sec['label'])}'>"
                        + _ledger_body(sec, native) + "</div>")
            continue
        body.append(fold(f"{esc(market.BOOK_NAMES[sec['book']])} close · "
                         f"{season_txt(sec['season'])}" + _lean_line(sec["h"], "close"),
                         _ledger_body(sec, native), id_=sec["id"], label=sec["label"]))
    return page("NBA ledger", "grades.html", "".join(body), built)


def _roi_section(body, h, native):
    """1u flat-bet ROI for each pick rule: model WP, market WP, actual."""
    body.append("<h3>ROI — one unit on every pick</h3>")
    if native:
        first = _roi_table(analysis.roi_summary(h, price="first"))
        if first:
            body.append("<p class='note'>At the <b>first snapshot</b> — the "
                        "earliest priced snapshot, with the model probability "
                        "written then: what an early bettor had.</p>" + first)
        pre = _roi_table(analysis.roi_summary(h, price="pre"))
        if pre:
            body.append("<p class='note'>At the <b>pregame snapshot price</b> — "
                        "the last snapshot before tip, near the close.</p>" + pre)
    body.append("<p class='note'>At the <b>closing price</b>.</p>"
                + _roi_table(analysis.roi_summary(h, price="close")))


def _band_table(sections):
    out = []
    for rule_label, rows in sections:
        out.append(f"<h3>{esc(rule_label)}</h3>")
        out.append(table(
            ["Picked price"] + ROI_HEADS[:5] + ["z", "Win %", "Break-even",
                                                "EV (pp)", "EV null (pp)",
                                                "Model %", "Market %"],
            [[esc(r["label"]), r["n"], f"{r['w']}–{r['l']}", *_roi_cells(r),
              z_txt(r["z"]), pct(r["actual"]), pct(r["breakeven"]),
              f"<span class='{sig_cls(r['z'])}'>{pp(r['ev'], cls=False)}</span>",
              f"{100 * r['ev_null']:+.1f}", pct(r["model_p"]), pct(r["q"])]
             for r in rows], left=(0,), key=(3, 5)))
    return "".join(out)


def _band_section(body, h, native):
    """1u flat ROI by the picked side's price band; for native rows also the
    current model's rows alone, the forward test of the band hypotheses."""
    price = "pre" if native else "close"
    parts = [("All rows", analysis.roi_by_band(h, price))]
    if native:
        cur = [t for t in ACTIVE_TAGS]
        rows = analysis.roi_by_band(h, price, tags=cur)
        if rows and (~h["model_tag"].astype(str).isin(cur)).any():
            parts.append(("Current model rows only ("
                          + " / ".join(f"<code>{esc(t)}</code>" for t in cur)
                          + ")", rows))
    parts = [(t, s) for t, s in parts if s]
    if not parts:
        return
    body.append("<details class='more'><summary>ROI by price band "
                "<span class='mut'>— one unit on every pick, by the picked "
                "side's moneyline</span></summary>")
    body.append("<p class='note'>Graded at the <b>"
                + ("pregame snapshot price" if native else "closing price")
                + "</b>. With 8 cells per section, a |z| near 2 somewhere "
                "happens by chance about one section in three. Bands are descriptive monitoring "
                "dimensions, not a filter: a band that looks good here is a "
                "hypothesis to test forward, and the hypotheses under test are "
                "the pre-registered ones at the top of this page (the earlier "
                "DraftKings −249 to −130 lean band is retired).</p>")
    for title, sections in parts:
        if len(parts) > 1:
            body.append(f"<p class='note'><b>{title}</b></p>")
        body.append(_band_table(sections))
    body.append("</details>")


def _ats_band_table(rows, head):
    """One ATS band table: cover rate and ROI beside its null per band."""
    return table(
        [head, "n", "W–L–P", "ROI", "Null", "vs null (± 1 SE)", "z", "Cover %",
         "Break-even", "Model cover %", "Market cover %"],
        [[esc(r["label"]), r["n"], f"{r['w']}–{r['l']}–{r['push']}",
          *_roi_cells(r), z_txt(r["z"]), pct(r["actual"]), pct(r["breakeven"]),
          pct(r["model_p"]), pct(r["q"])] for r in rows],
        left=(0,), key=(3, 5))


def _ats_section(body, h):
    """1u flat spread bets at the closing spread, beside the moneyline ROI.
    Only spreads from books the pages show (analysis.HIDDEN_BOOKS); the
    sportsbook behind a Kalshi-priced row's spread is its `spread_book`."""
    book = h["spread_book"] if "spread_book" in h else h["close_book"]
    h = h[~book.astype(str).isin(analysis.HIDDEN_BOOKS)]
    sections = analysis.ats_summary(h)
    if not sections:
        return
    body.append("<details class='more'><summary>Against the spread "
                "<span class='mut'>— one unit on every pick at the closing "
                "spread</span></summary>")
    out = []
    for rule_label, rows in sections:
        out.append(f"<h3>{esc(rule_label)}</h3>")
        out.append(table(
            ["Picks", "n", "W–L–P", "ROI", "Null", "vs null (± 1 SE)", "Cover %",
             "Break-even", "Model cover %", "Market cover %"],
            [[esc(r["label"]), r["n"], f"{r['w']}–{r['l']}–{r['push']}",
              *_roi_cells(r), pct(r["actual"]), pct(r["breakeven"]),
              pct(r["model_p"]), pct(r["q"])]
             for r in rows], left=(0,), key=(3, 5)))
    bands = analysis.ats_lean_by_band(h)
    for key, title in (("line", "By the picked side's spread"),
                       ("price", "By spread price")):
        if bands[key]:
            out.append(f"<h3>{title} <span class='mut'>— lean ATS, at the "
                       "closing spread</span></h3>")
            out.append(_ats_band_table(bands[key], "Spread" if key == "line"
                                       else "Price"))
    body.append("".join(out))
    body.append("<p class='note'>Only rows with a recorded DraftKings closing "
                "spread. The spread and price bands are the picked side's "
                "(negative = laying points); they are descriptive monitoring "
                "dimensions, not a filter, and with a dozen cells a |z| near 2 "
                "somewhere happens by chance; below "
                f"{analysis.BAND_MIN_N} bets a band shows no SE or z. "
                "<b>Model cover %</b> maps "
                "the model's win probability to a margin with σ = "
                f"{analysis.ATS_SIGMA:g} pts; it is a display mapping, not a "
                "new prediction. <b>Cover %</b> is covers ÷ (covers + misses); "
                "pushes refund. The null is minus the spread hold.</p>")
    body.append("</details>")


def _recent_section(body, h, native, n=60):
    recent = h.sort_values(["slate_date", "tip_utc"], ascending=False).head(n)
    rows = []
    for _, r in recent.iterrows():
        won = int(r["lean_won"]) == 1
        lean_u = market.unit_profit(r["lean_ml"], won)
        rows.append([
            esc(r["slate_date"]), f"{esc(r['away'])} @ {esc(r['home'])}",
            f"<span class='chip'>{esc(r['lean'])}</span>", pct(r["lean_p"]),
            pct(r["lean_q"]), ml_txt(r["lean_ml"]),
            f"{int(r['away_pts'])}–{int(r['home_pts'])}",
            f"<span class='{'pos' if won else 'neg'}'>{'W' if won else 'L'} "
            f"{lean_u:+.2f}u</span>",
        ])
    more = (f"latest {n} of {len(h)}" if len(h) > n else f"all {len(h)}")
    body.append(f"<details class='more'{' open' if native else ''}><summary>"
                f"Game by game <span class='mut'>— {more}, newest first</span>"
                "</summary>")
    body.append(table(["Date", "Away @ Home", "Lean", "Model %", "Market %",
                       "Close ML", "Final (away–home)", "Lean · P/L (1u)"],
                      rows, left=(0, 1, 2)))
    if len(h) > n:
        body.append("<p class='note'>The full record is in the CSV under "
                    "<code>data/</code>.</p>")
    body.append("</details>")


# ------------------------------------------------------------ calibration ---
def reliability_svg(series, title):
    """Stated vs actual probability, ±2 SE bars, dashed diagonal = calibrated.

    series: [(name, css class, analysis.reliability points)]. The first
    series is drawn as circles, the second as diamonds, so identity never
    rests on colour alone; each point has a hover title with its numbers.
    """
    W, H, L, R, T, B = 360, 300, 56, 20, 10, 38
    pw, ph = W - L - R, H - T - B

    def X(v):
        return L + pw * min(max(v, 0.0), 1.0)

    def Y(v):
        return T + ph * (1 - min(max(v, 0.0), 1.0))
    g = []
    for k in range(6):
        v = k / 5
        g.append(f"<line class='gl' x1='{X(v):.1f}' y1='{T}' x2='{X(v):.1f}' y2='{T + ph}'/>"
                 f"<line class='gl' x1='{L}' y1='{Y(v):.1f}' x2='{L + pw}' y2='{Y(v):.1f}'/>"
                 f"<text class='ax' x='{X(v):.1f}' y='{T + ph + 14}' text-anchor='middle'>{100 * v:.0f}%</text>"
                 f"<text class='ax' x='{L - 6}' y='{Y(v) + 4:.1f}' text-anchor='end'>{100 * v:.0f}%</text>")
    g.append(f"<line class='diag' x1='{X(0)}' y1='{Y(0)}' x2='{X(1)}' y2='{Y(1)}'/>")
    g.append(f"<text class='ax' x='{L + pw / 2}' y='{H - 4}' text-anchor='middle'>"
             "Stated probability</text>"
             f"<text class='ax' transform='translate(12 {T + ph / 2}) rotate(-90)' "
             "text-anchor='middle'>Actual win rate</text>")
    legend = []
    for i, (name, cls, pts) in enumerate(series):
        dx = (i - (len(series) - 1) / 2) * 6.0     # nudge overlapping series apart
        for p in pts:
            x, y = X(p["stated"]) + dx, Y(p["actual"])
            lo, hi = Y(p["actual"] - 2 * p["se"]), Y(p["actual"] + 2 * p["se"])
            tip = (f"{name}, {p['lo']:.1f}–{p['hi']:.1f}: stated {pct(p['stated'])}, "
                   f"actual {pct(p['actual'])} (n={p['n']}, ±2 SE {200 * p['se']:.1f} pp)")
            mark = (f"<circle class='{cls} mk' cx='{x:.1f}' cy='{y:.1f}' r='4'/>"
                    if i == 0 else
                    f"<rect class='{cls} mk' x='{x - 4.2:.1f}' y='{y - 4.2:.1f}' "
                    f"width='8.4' height='8.4' transform='rotate(45 {x:.1f} {y:.1f})'/>")
            g.append(f"<g><title>{esc(tip)}</title>"
                     f"<line class='{cls} eb' x1='{x:.1f}' y1='{hi:.1f}' x2='{x:.1f}' y2='{lo:.1f}'/>"
                     f"{mark}<circle cx='{x:.1f}' cy='{y:.1f}' r='10' fill='transparent'/></g>")
        key = ("<circle class='{c} mk' cx='7' cy='7' r='4.5'/>" if i == 0 else
               "<rect class='{c} mk' x='2.8' y='2.8' width='8.4' height='8.4' "
               "transform='rotate(45 7 7)'/>").format(c=cls)
        legend.append(f"<span><svg width='14' height='14' viewBox='0 0 14 14' "
                      f"aria-hidden='true'>{key}</svg> {esc(name)}</span>")
    return (f"<div class='chart'><div class='t'>{esc(title)}</div>"
            f"<div class='legend'>{''.join(legend)}</div>"
            f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='{esc(title)}: "
            "stated probability against actual win rate'>" + "".join(g)
            + "</svg></div>")


CHART_NOTE_SHORT = ("Dashed diagonal = calibrated; bars are ±2 SE. Hover a "
                    "point for its numbers.")
CHART_NOTE = ("Each point is one probability bin: across the x-axis what was "
              "stated, up the y-axis how often it happened. On the dashed "
              "diagonal = calibrated. The bar is ±2 SE, where a correct "
              "forecast would usually land; a bar that misses the diagonal is "
              "a real miss, not noise. Hover a point for its numbers; the "
              "table below has them all.")


def _calib_cell(a):
    if not a:
        return "<span class='mut'>—</span>"
    z = a["diff"] / a["se"] if a["se"] > 0 else float("nan")
    return (f"{pct(a['actual'])} <span class='mut'>vs {pct(a['implied'])}</span> "
            f"<span class='{sig_cls(z)}'>{pp(a['diff'], cls=False)}</span> "
            f"<span class='mut'>± {100 * a['se']:.1f} · n={a['n']}</span>")


def _market_games(native, recon):
    """Distinct graded games (native first) for the model-free market view."""
    allg = pd.concat([d for d in (native, recon) if len(d)], ignore_index=True) \
        if (len(native) or len(recon)) else ledger.empty()
    g = ledger.graded(allg)
    if len(g):
        g = g.assign(_nat=(g["basis"] == "native")).sort_values(
            "_nat", ascending=False).drop_duplicates("game_id")
        g = g.assign(p_home=g["p_home"].fillna(0.5))   # market view needs no model
    return analysis.with_close(g)


def render_calibration(native, recon, built):
    native, recon = analysis.market_view(native), analysis.market_view(recon)
    parts = analysis.shown(analysis.book_split(_market_games(native, recon)))
    secs = analysis.shown(_sections(native, recon))
    jumps = [(_slug("market", b), f"Market · {market.BOOK_NAMES[b]}") for b, _ in parts]
    jumps += [(s["id"], "Model · " + s["jump"]) for s in secs]
    body = ["<h1>Calibration</h1><p class='lead'>When a forecast says 70%, "
            "does it happen 70% of the time? First the <b>market</b> (no-vig "
            "close against results), then the <b>model</b>, with the market's "
            "forecast on the same games beside it.</p>",
            READ_KEY.replace("Judge every ROI against its <b>null</b>: the ROI "
                             "if the market's no-vig prices were exactly right. "
                             "That is about −4% (the hold: the bookmaker's "
                             "margin), not zero. ",
                             "Here the null is the diagonal: stated = actual. "),
            _jump(jumps), "<h2 id='market'>Market: no-vig close vs actual</h2>"]
    if not parts:
        body.append("<p class='note'>No graded games with a closing line yet.</p>")
    for book, hm in parts:
        out = []
        _market_section(out, hm, first=book == parts[0][0])
        body.append(fold(f"<span class='basis'>{esc(market.BOOK_NAMES[book])}</span> "
                         f"<span class='mut'>{len(hm)} games</span>", "".join(out),
                         open_=book == parts[0][0], id_=_slug("market", book)))
    body.append("<h2 id='model'>Model vs market, same games</h2>")
    _model_sections(body, secs)
    body.append("<p class='note'>Bet grading by price band (ROI, EV and their "
                f"nulls) is on the <a href='grades.html'>Ledger</a>. {HIDDEN_NOTE}</p>")
    return page("NBA calibration", "market-calibration.html", "".join(body), built)


def _market_section(body, hm, first=True):
    rows_, totals = analysis.market_calibration(hm)
    if not rows_:
        body.append("<p class='note'>No graded games with a closing line yet.</p>")
        return
    nn = int((hm["basis"] == "native").sum())
    seasons = ", ".join(season_txt(s) for s in sorted(hm["season"].dropna().unique()))
    tiles = []
    for key, lab in (("favourite", "Favourites won"), ("home", "Home sides won")):
        t = totals.get(key)
        if t:
            z = t["diff"] / t["se"] if t["se"] > 0 else float("nan")
            tiles.append(tile(lab, pct(t["actual"]),
                              f"vs {pct(t['implied'])} implied: {pp(t['diff'], cls=False)} "
                              f"± {100 * t['se']:.1f} (1 SE) — "
                              f"{resolved_txt(z, 'above', 'below')} · n={t['n']}",
                              sig_cls(z)))
    body.append("<div class='tiles'>" + "".join(tiles) + "</div>")
    q = np.r_[hm["close_q_home"].to_numpy(float), 1 - hm["close_q_home"].to_numpy(float)]
    y = np.r_[hm["home_won"].to_numpy(float), 1 - hm["home_won"].to_numpy(float)]
    body.append("<div class='charts'>"
                + reliability_svg([("Market (no-vig close)", "s1",
                                    analysis.reliability(q, y))],
                                  "Market calibration, both sides of every game")
                + f"<p class='note'>{CHART_NOTE if first else CHART_NOTE_SHORT}</p></div>")
    body.append("<details class='more'><summary>By price rung and side "
                "<span class='mut'>— actual vs implied, gap ± 1 SE</span></summary>")
    body.append(table(["Price rung", "Home side", "Away side", "Both sides"],
                      [[esc(r["rung"]), _calib_cell(r["home"]),
                        _calib_cell(r["away"]), _calib_cell(r["all"])]
                       for r in rows_]))
    body.append("</details>")
    body.append(f"<p class='note'>{len(hm)} games ({nn} native, "
                f"{len(hm) - nn} reconstructed; seasons {seasons}). Two "
                "observations per game, one per side; ± is one SE under correct "
                "prices, so a gap under ~2± is indistinguishable from fair. No "
                "both-sides total: the sides sum to 1 and one wins, so it is 50% "
                "by construction. Favourites asks once per game.</p>")


def _model_sections(body, secs):
    """Model vs market, one basis x closing book x season at a time."""
    first = True
    for sec in secs:
        h = sec["h"]
        out = []
        _model_section(out, h, first)
        if len(h):
            first = False
            s = analysis.scoring(h)
            se = s["d_logloss_se"]
            z = -s["d_logloss"] / se if np.isfinite(se) and se > 0 else float("nan")
            rt = (f"<span class='rt'>log loss model − market <span class='{sig_cls(z)}'>"
                  f"{s['d_logloss']:+.4f}</span> ± {se4(se)} · n={s['n']}</span>")
        else:
            rt = ""
        body.append(fold(f"{sec['title']}{rt}", "".join(out),
                         open_=sec.get("basis") == "native" or len(secs) == 1,
                         id_=sec["id"], label=f"Model — {sec['label']}"))


def _model_section(body, h, first):
    """One section's model-vs-market scores, chart and bins."""
    if not len(h):
        body.append("<p class='note'>No graded model rows with a close yet.</p>")
        return
    s = analysis.scoring(h)
    zb = (-s["d_brier"] / s["d_brier_se"]
          if np.isfinite(s["d_brier_se"]) and s["d_brier_se"] > 0 else float("nan"))
    body.append("<div class='tiles'>" + "".join([
        _score_tile(s),
        tile("Brier (model / market)",
             f"{s['model']['brier']:.4f} / {s['market']['brier']:.4f}",
             f"model − market {s['d_brier']:+.4f} ± {se4(s['d_brier_se'])} "
             f"(1 SE) · {resolved_txt(zb, 'model better', 'market better')}",
             sig_cls(zb)),
        tile("Picked the winner (model lean / market favourite)",
             f"{pct(s['model']['acc'])} / {pct(s['market']['acc'])}",
             "same games · accuracy is not calibration"),
    ]) + "</div>")
    p = h["p_home"].to_numpy(float)
    q = h["close_q_home"].to_numpy(float)
    y = h["home_won"].to_numpy(float)
    body.append("<div class='charts'>"
                + reliability_svg([("Model", "s1", analysis.reliability(p, y)),
                                   ("Market (no-vig close)", "s2",
                                    analysis.reliability(q, y))],
                                  "Model vs market calibration, P(home win)")
                + f"<p class='note'>{CHART_NOTE if first else CHART_NOTE_SHORT}"
                " Both forecasts are for the same games; each is binned "
                "on its own probability. Negative model − market on the "
                "scores means the model did better; an interval spanning "
                "0 has not separated them.</p></div>")
    cal = analysis.model_calibration(h)
    body.append("<details class='more'><summary>Model bins in numbers "
                "<span class='mut'>— with the market's mean on the same "
                "games</span></summary>")
    body.append(table(
        ["Model P(home)", "n", "Model mean", "Market mean (same games)",
         "Actual", "Actual − model (± 1 SE)"],
        [[f"{c['lo']:.1f}–{c['hi']:.1f}", c["n"], pct(c["model"]),
          pct(c["market"]), pct(c["actual"]),
          f"<span class='{sig_cls((c['actual'] - c['model']) / c['se'] if c['se'] > 0 else float('nan'))}'>"
          f"{pp(c['actual'] - c['model'], cls=False)}</span> "
          f"<span class='mut'>± {100 * c['se']:.1f}</span>"]
         for c in cal], key=(4, 5)))
    body.append("</details>")


# ------------------------------------------------------------- preseason ---
def render_preseason(pre, built):
    """Exhibition rows (data/nba_preseason.csv) on their own page: model vs
    the close on the same games, one closing book x season at a time, then
    the flat-bet record at the pregame price and every row. Nothing here is
    pooled with the native or reconstructed ledgers."""
    body = ["<h1>Preseason</h1><p class='lead'>Exhibition games, scored by "
            "the early-season carryover model on <b>last season's games "
            "only</b> and priced from Kalshi. Starters rest, so these rows "
            "show how far the offseason view travels, not regular-season "
            "skill.</p><div class='flag-note'>Tracked apart from everything "
            "else: never in the Ledger, Calibration, hypotheses or any fit "
            f"(tag <code>{esc(PRESEASON_TAG)}</code>).</div>",
            READ_KEY.replace("Native</dt><dd>Written before tip and frozen: "
                             "the forward test.",
                             "Preseason</dt><dd>Written before tip and "
                             "frozen, like native rows, but exhibitions.")]
    if not len(pre):
        body.append("<p class='note'>No preseason games recorded yet. The "
                    "daily build writes them from the first preseason slate "
                    "it sees.</p>")
        return page("NBA preseason", "preseason.html", "".join(body), built)
    g = ledger.graded(pre)
    h = analysis.with_close(g)
    scored = pd.to_numeric(pre["p_home"], errors="coerce").notna()
    priced = pd.to_numeric(pre["pre_q_home"], errors="coerce").notna()
    body.append(f"<p class='note'>{len(pre)} games recorded: {int(scored.sum())} "
                f"with a model P, {int(priced.sum())} with a pregame price, "
                f"{len(g)} graded, {len(h)} graded with a model P and a "
                "close.</p>")
    secs = []
    for book, hb in analysis.book_split(h):
        bname = market.BOOK_NAMES[book]
        for season in sorted(hb["season"].dropna().unique()):
            stxt = season_txt(season)
            secs.append(dict(h=hb[hb["season"] == season], id=_slug("pre", book, stxt),
                             label=f"Preseason · {bname} close · {stxt}",
                             title=f"<span class='badge recon'>Preseason · exhibition</span>"
                                   f"{esc(bname)} close · {stxt}"))
    if not secs:
        body.append("<p class='note'>No graded games with a model P and a "
                    "close yet.</p>")
    _model_sections(body, secs)
    for sec in secs:
        roi = analysis.roi_summary(sec["h"], price="pre")
        if roi:
            body.append(f"<h3>Flat 1u bets at the pregame price — "
                        f"{esc(sec['label'])}</h3>")
            body.append(_roi_table(roi))
    rows = []
    for _, r in pre.sort_values(["slate_date", "tip_utc"], ascending=False).iterrows():
        won = pd.to_numeric(r["home_won"], errors="coerce")
        final = ("pending" if not np.isfinite(won) else
                 f"{int(r['away_pts'])}–{int(r['home_pts'])}")
        rows.append([esc(r["slate_date"]), f"{esc(r['away'])} @ {esc(r['home'])}",
                     pct(pd.to_numeric(r["p_home"], errors="coerce")),
                     pct(pd.to_numeric(r["pre_q_home"], errors="coerce"))
                     + book_tag(r["pre_book"]),
                     pct(pd.to_numeric(r["close_q_home"], errors="coerce"))
                     + book_tag(r["close_book"]),
                     final])
    body.append("<details class='more' open><summary>Game by game <span "
                "class='mut'>— newest first</span></summary>")
    body.append(table(["Date", "Away @ Home", "Model P(home)",
                       "Pregame q(home)", "Close q(home)", "Final (away–home)"],
                      rows, left=(0, 1), key=(2,)))
    body.append("</details><p class='note'>q is the no-vig market probability. "
                "Prices are refreshed each build until tip and frozen after; "
                "prices are Kalshi's YES asks with the taker fee included "
                "(as American odds: what a bet costs); q is the bid/ask "
                "midpoints normalised (the market's view), and the close is "
                "the last 1-minute Kalshi candle before tip. b2b_net is read "
                "from the previous day's scoreboard.</p>")
    return page("NBA preseason", "preseason.html", "".join(body), built)


def snapshot_injuries(rows):
    """Pregame injury lists for today's accepted rows -> data/nba_injuries.csv.

    Shown on the Picks cards (`injury_block`); the model does not use them.
    The snapshot time is taken after the fetch, and a game at or after tip
    is never touched.
    """
    if not rows:
        return
    snaps = {}
    for r in rows:
        try:
            got = market.game_injuries(r["game_id"], r["home"], r["away"])
        except Exception as e:  # noqa: BLE001
            log(f"injuries {r['game_id']}: {e!r}")
            got = None
        snaps[str(r["game_id"])] = (r["tip_utc"], got)
    inj, acc, rej = ledger.upsert_injuries(ledger.load_injuries(), snaps)
    ledger.save_injuries(inj)
    log(f"injury snapshots: {len(acc)} written, {len(rej)} skipped {rej}")


def snapshot_sleeper(rows, today):
    """Sleeper / RotoWire pregame projections for today's accepted rows ->
    data/nba_sleeper.csv (owner's decision 2026-10-22): each listed player's
    projected minutes and line and his injury status, the game's first
    snapshot kept and its latest replaced until tip, frozen after. Research
    data: the model and the pages do not read it, and a Sleeper failure
    never stops the build.
    """
    if not rows:
        return
    try:
        proj = sleeper.live_projections(today)
    except Exception as e:  # noqa: BLE001 - research data, never blocks the build
        log(f"sleeper projections unavailable ({e!r})")
        proj = None
    snaps = {str(r["game_id"]): (r["tip_utc"], None if proj is None
                                 else sleeper.snapshot_rows(proj, r["home"], r["away"]))
             for r in rows}
    df, acc, rej = ledger.upsert_sleeper(ledger.load_sleeper(), snaps)
    ledger.save_sleeper(df)
    log(f"sleeper snapshots: {len(acc)} written, {len(rej)} skipped {rej}")


def write_pages(native, recon, today, model_ok, pre=None):
    OUT_DIR.mkdir(exist_ok=True)
    built = datetime.now(PT).strftime("%a %b %-d, %Y · %-I:%M %p PT")
    (OUT_DIR / "index.html").write_text(render_index(native, today, built, model_ok,
                                                     recon, ledger.load_injuries()))
    rebuilt = ledger.load(ledger.REBUILT_PATH)
    (OUT_DIR / "results.html").write_text(render_results(native, built, today, rebuilt))
    (OUT_DIR / "performance.html").write_text(render_performance(native, built, today,
                                                                 rebuilt))
    (OUT_DIR / "grades.html").write_text(render_grades(native, recon, built))
    (OUT_DIR / "market-calibration.html").write_text(
        render_calibration(native, recon, built))
    (OUT_DIR / "preseason.html").write_text(
        render_preseason(ledger.empty() if pre is None else pre, built))
    (OUT_DIR / "model.html").write_text(render_model(native, recon, built))
    # The quotable plain-text readout, as the sibling projects keep it:
    # published beside the pages; main() also commits it under data/.
    text = report.report_text(native, recon, pre, ACTIVE_TAGS)
    (OUT_DIR / report.REPORT_NAME).write_text(text, encoding="utf-8")
    (OUT_DIR / ".nojekyll").write_text("")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None, help="ET slate date (default today)")
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args(argv)
    today = a.date or et_today()
    native = ledger.load(ledger.NATIVE_PATH)
    recon = ledger.load(ledger.RECON_PATH)
    pre = ledger.load(ledger.PRESEASON_PATH)
    weights, model = load_model()
    model_ok = weights is not None and model is not None
    global ACTIVE_TAGS
    if model_ok:
        ACTIVE_TAGS = active_tags(model, load_avail(), load_fallback(), load_early())

    if not a.render_only:
        native, n = grade(native, today, kalshi_close_fn=kalshi_grade)
        log(f"graded {n} game(s)")
        if model_ok:
            try:
                rows = score_slate(today, weights, model)
            except Exception as e:  # noqa: BLE001 - keep grading + pages alive
                log(f"slate scoring failed ({e!r}); pages render from the ledger")
                rows = []
            native, acc, rej = ledger.upsert_pregame(native, rows)
            log(f"pregame rows: {len(acc)} written, {len(rej)} rejected {rej}")
            snapshot_injuries([r for r in rows if str(r["game_id"]) in acc])
            snapshot_sleeper([r for r in rows if str(r["game_id"]) in acc], today)
        else:
            log("no fitted model in model/ -- skipping slate scoring")
        ledger.save(native, ledger.NATIVE_PATH)
        # exhibitions: their own file, graded and scored the same way
        pre, n = grade(pre, today, close=kalshi_close)
        rows = []
        if model_ok:
            try:
                rows = score_preseason(today, weights)
            except Exception as e:  # noqa: BLE001
                log(f"preseason scoring failed ({e!r})")
        pre, acc, rej = ledger.upsert_pregame(pre, rows, basis="preseason")
        if n or acc:
            log(f"preseason: graded {n}, {len(acc)} rows written, "
                f"{len(rej)} rejected {rej}")
        if len(pre):
            ledger.save(pre, ledger.PRESEASON_PATH)
    text = write_pages(native, recon, today, model_ok, pre)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    log(f"wrote {OUT_DIR}/" + ", ".join(h for h, _ in PAGES)
        + f", {report.REPORT_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
