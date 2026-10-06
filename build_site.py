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
     same pregame lock (research input; the model does not read it).
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
from datetime import datetime, timezone
from pathlib import Path
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

ET = ZoneInfo("America/New_York")
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
# Exhibition rows (data/nba_preseason.csv, score_preseason): the v2 carryover
# logit on last season's games alone (regular-season game 0 abstains).
PRESEASON_TAG = "fourfactors_hl25_b2b_carry25_preseason_g0"
FALLBACK_FILE = "logit_v4.json"
AVAIL_TAGS = (MODEL_TAG_V3, MODEL_TAG_V4_AVAIL, MODEL_TAG_V5_AVAIL,
              MODEL_TAG_V6_AVAIL)
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


def model_tag(model, avail=False):
    """The tag for a row scored by `model` (the availability logit if avail)."""
    if avail:
        return (MODEL_TAG_V6_AVAIL if is_v6(model) else
                MODEL_TAG_V5_AVAIL if is_v5(model) else
                MODEL_TAG_V4_AVAIL if has_phase(model) else MODEL_TAG_V3)
    return (MODEL_TAG_V6 if is_v6(model) else
            MODEL_TAG_V5 if is_v5(model) else
            MODEL_TAG_V4 if has_phase(model) else MODEL_TAG)


def active_tags(model, avail_model, fallback=None):
    """Tags the current model files produce (page footer, ledger split),
    including the v4 fallback's when the base model is v5."""
    tags = [model_tag(model)]
    if avail_model is not None:
        tags.append(model_tag(avail_model, avail=True))
    if is_v5(model) and fallback is not None:
        tags.append(model_tag(fallback))
    return tags


def load_early():
    """The early-season logit (model/logit_early.json), or None."""
    try:
        return nc.load_json(cold_start.MODEL_FILE)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fresh_logs(year, today):
    """Game logs for `year`, refetched at most once per ET day."""
    stamp = nc.CACHE / f"refreshed_{year}.txt"
    refresh = not (stamp.exists() and stamp.read_text().strip() == today)
    logs = nc.load_logs(year, refresh=refresh)
    if refresh:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(today)
    return logs


def score_game(logs, home, away, date, weights, model, half_life=None,
               early=None, prior_logs=None, avail_model=None, avail=None,
               opening=None, talent=None, fallback=None):
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
    """
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
        use, out["model_tag"] = early, model_tag(model)
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
        use, out["model_tag"] = fallback, model_tag(fallback)
        feats = use["features"]
    p = float(nc.predict(use, [[vals[f] for f in feats]])[0])
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


def grade(led, today, close=espn_close):
    """Final scores for finished games from earlier slates, with the closing
    pair from `close(row)` (ESPN books; kalshi_close for preseason rows)."""
    todo = ledger.pending(led, today)
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
            n += ledger.apply_result(led, gid, g, odds)
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
    fallback of a v5 model), or abstain (no P). Reads the row; changes no
    routing."""
    if not np.isfinite(pd.to_numeric(row.get("p_home"), errors="coerce")):
        return "abstain"
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
    early, prior = load_early(), None
    if early is not None and any(
            int((lg["date"] < pd.Timestamp(today)).sum()) < nc.MIN_GAMES
            for lg in logs.values()):
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
    opening = nc.season_opening(logs)
    fallback = load_fallback() if is_v5(model) else None
    talent = getattr(live, "talent", None)
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
                            opening=opening, talent=talent, fallback=fallback))
        # decision-time record (ledger first_report_utc / first_route): the
        # report edition this build read, whether or not it covered the game
        r["report_utc"] = report_utc(getattr(live, "report_time", None))
        r["route"] = route(r, model)
        add_pregame_price(r)
        rows.append(r)
    return rows


def add_pregame_price(r):
    """The current moneyline pair of the first book in market.BOOKS, in place
    (pre_book, pre_home_ml, pre_away_ml, pre_q_home; price_note when none)."""
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


def score_preseason(today, weights, now=None):
    """Exhibition tracking (data/nba_preseason.csv): every NBA-vs-NBA
    preseason game on today's scoreboard that has not tipped, scored by the
    early-season carryover logit (model/logit_early.json) on LAST SEASON's
    games only, i.e. the game-0 case the regular-season card abstains on.

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
    early = load_early()
    if not pre or early is None:
        if pre:
            log("preseason: no early model (model/logit_early.json); skipped")
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
:root{--bg:#f3f5f7;--fg:#161b20;--mut:#4f5a65;--faint:#636e7b;--card:#fff;
--card2:#eef1f4;--line:#dde3e8;--line2:#eceff2;--pos:#1d7a3c;--neg:#b4442a;
--acc:#825c0c;--accb:#b07c10;--chip:#f6efe0;--s1:#3474a8;--s2:#c6542c;
--grid:#eceff2;--pos-bg:#e9f4ec;--neg-bg:#f9ece8;
--mono:"JetBrains Mono",ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
--sans:"Archivo",system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
--shadow:0 1px 2px rgba(16,18,29,.05),0 10px 26px -20px rgba(16,18,29,.28);--r:6px}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1418;
--fg:#e7ecef;--mut:#96a2ad;--faint:#7f8b97;--card:#171d24;--card2:#131920;
--line:#242e38;--line2:#1c242c;--pos:#58c27d;--neg:#ef7f62;--acc:#f4c460;
--accb:#f4c460;--chip:#2a2618;--s1:#609ed0;--s2:#ec7a48;--grid:#1c242c;
--pos-bg:#14261b;--neg-bg:#2b1a15;
--shadow:0 1px 2px rgba(0,0,0,.45),0 14px 32px -22px rgba(0,0,0,.8)}}
:root[data-theme="dark"]{--bg:#0f1418;--fg:#e7ecef;--mut:#96a2ad;--faint:#7f8b97;
--card:#171d24;--card2:#131920;--line:#242e38;--line2:#1c242c;--pos:#58c27d;
--neg:#ef7f62;--acc:#f4c460;--accb:#f4c460;--chip:#2a2618;--s1:#609ed0;
--s2:#ec7a48;--grid:#1c242c;--pos-bg:#14261b;--neg-bg:#2b1a15;
--shadow:0 1px 2px rgba(0,0,0,.45),0 14px 32px -22px rgba(0,0,0,.8)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--acc)}
main{max-width:1100px;margin:0 auto;padding:20px 16px 60px}
.topbar{display:flex;align-items:baseline;justify-content:space-between;gap:12px;
border-bottom:2px solid var(--fg);padding-bottom:10px;margin-bottom:12px}
.brand{font:800 18px/1 var(--sans);letter-spacing:.13em;text-transform:uppercase;
color:var(--fg);text-decoration:none}
.theme{appearance:none;border:1px solid var(--line);background:var(--card);color:var(--mut);
font:600 14px/1 var(--sans);padding:7px 11px;border-radius:var(--r);cursor:pointer}
.theme:hover{color:var(--fg)}
nav{display:flex;gap:6px 16px;flex-wrap:wrap;margin:0 0 14px;font:600 14px/1.3 var(--sans)}
nav a{color:var(--mut);text-decoration:none;padding-bottom:3px;border-bottom:2px solid transparent}
nav a:hover{color:var(--fg)}
nav a.on{color:var(--fg);border-bottom-color:var(--accb)}
nav.jump{gap:6px;margin:10px 0 4px;font:500 13px/1.3 var(--sans)}
nav.jump a{border:1px solid var(--line);border-radius:999px;padding:2px 10px;background:var(--card)}
h1{font:800 26px/1.15 var(--sans);margin:8px 0 6px}
h2{font:800 15px/1.2 var(--sans);letter-spacing:.08em;text-transform:uppercase;
margin:34px 0 8px;scroll-margin-top:8px}
h2 .badge{letter-spacing:.04em}
h3{font:750 13px/1.2 var(--sans);letter-spacing:.06em;text-transform:uppercase;
color:var(--faint);margin:18px 0 6px}
.lead,.note{color:var(--mut);max-width:78ch}.note{font-size:13.5px}
.note code{overflow-wrap:anywhere}code{font-family:var(--mono);font-size:.92em}
.stamp{font-family:var(--mono);font-size:13px}
.key{background:var(--card2);border:1px solid var(--line2);border-radius:var(--r);padding:9px 13px;
font-size:13.5px;max-width:78ch;margin:10px 0;color:var(--mut)}
.key b{color:var(--fg)}
.key summary{cursor:pointer;color:var(--acc);margin-top:4px}
.key dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 12px;margin:8px 0 2px}
.key dt{font-weight:600;color:var(--fg)}.key dd{margin:0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:14px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:var(--r);
padding:11px 13px;box-shadow:var(--shadow)}
.tile .l{font:650 12px/1.25 var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
.tile .v{font:800 22px/1.15 var(--mono);margin:5px 0 3px;font-variant-numeric:tabular-nums}
.tile .s{font:500 12.5px/1.4 var(--mono);color:var(--mut)}
.tile.pos .v{color:var(--pos)}.tile.neg .v{color:var(--neg)}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:var(--r);
background:var(--card);box-shadow:var(--shadow);margin-bottom:12px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
th,td{padding:7px 9px;border-bottom:1px solid var(--line2);text-align:right;white-space:nowrap}
td{font-family:var(--mono);font-size:13.5px}td.l{font-family:var(--sans);font-size:14px}
th{font:650 11.5px/1.2 var(--sans);letter-spacing:.06em;text-transform:uppercase;
color:var(--faint);background:var(--card2);border-bottom:1px solid var(--line);position:sticky;top:0}
td.l,th.l{text-align:left}tr:last-child td{border-bottom:0}
th.k,td.k{background:var(--chip)}
.pos{color:var(--pos)}.neg{color:var(--neg)}.mut{color:var(--mut)}
.chip{display:inline-block;background:var(--chip);border-radius:4px;padding:0 6px;
font:800 13px/1.5 var(--mono);color:var(--acc)}
.basis{display:inline-block;font:700 11px/1.4 var(--sans);letter-spacing:.04em;
text-transform:uppercase;border:1px solid var(--line);border-radius:3px;padding:1px 5px;color:var(--mut)}
.badge{display:inline-block;font:700 11.5px/1.3 var(--sans);letter-spacing:.04em;
text-transform:uppercase;border-radius:3px;padding:2px 7px;vertical-align:middle;
margin-right:8px;border:1px solid var(--line)}
.badge.native{background:var(--pos-bg);color:var(--pos);border-color:transparent}
.badge.recon{background:var(--card2);color:var(--mut)}
details.more{margin:12px 0}details.more>summary{cursor:pointer;font-weight:700;font-size:14px;padding:4px 0}
details.more>summary .mut{font-weight:400;font-size:13px}
.charts{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start;margin:12px 0}
.chart{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:10px 12px;
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
.foot{margin-top:28px;color:var(--faint);font-size:12.5px;border-top:1px solid var(--line);padding-top:10px}
.foot code{font-size:12px}
@media (max-width:640px){table{font-size:13px}td{font-size:12.5px}.brand{font-size:15px}}
"""

THEME_JS = ("<script>(function(){var k='nba-theme',r=document.documentElement;"
            "try{var s=localStorage.getItem(k);if(s)r.setAttribute('data-theme',s);}catch(e){}"
            "window.toggleTheme=function(){var d=r.getAttribute('data-theme')||"
            "(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');"
            "var n=d==='dark'?'light':'dark';r.setAttribute('data-theme',n);"
            "try{localStorage.setItem(k,n);}catch(e){}};})();</script>")

SITE_NAME = "NBA Four-Factors Composite"
ASSETS = Path(__file__).resolve().parent / "assets" / "fonts"
PAGES = (("index.html", "Today"), ("grades.html", "Ledger"),
         ("market-calibration.html", "Calibration"), ("model.html", "Model"),
         ("preseason.html", "Preseason"), ("ledger_report.txt", "Report (text)"))


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
    cur = " class='on' aria-current='page'"
    nav = "".join(f"<a href='{h}'{cur if h == active else ''}>{t}</a>"
                  for h, t in PAGES)
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><meta name='description' content='NBA "
            "four-factors composite win probabilities, pregame-locked ledger and "
            f"market calibration'>{THEME_JS}<style>{font_face_css()}{CSS}</style>"
            "</head><body><main><div class='topbar'><a class='brand' "
            f"href='index.html'>{SITE_NAME}</a><button class='theme' type='button' "
            "onclick='toggleTheme()' aria-label='Toggle colour theme'>Theme</button>"
            f"</div><nav>{nav}</nav>{body}<div class='foot'>Built "
            f"<span class='stamp'>{esc(built)}</span> · model "
            + " / ".join(f"<code>{esc(t)}</code>" for t in ACTIVE_TAGS)
            + ". Win probabilities from prior-game four factors, roster talent "
            "and the NBA injury report; the market is a benchmark only and never "
            "enters the model. Not betting advice.</div></main></body></html>")


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


def tip_et(s):
    t = ledger.parse_utc(s)
    return t.astimezone(ET).strftime("%-I:%M %p") if t else "—"


def book_tag(book):
    """Visible label for any price not from DraftKings (the default book)."""
    if pd.isna(book) or book in ("", "dk"):
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


def _roi_headline(label, r, basis_txt):
    """An NFL/MLB-style headline tile: flat 1u ROI, its null and the market
    favourite on the same games. Colour only at >= 2 SE from the null."""
    if not r:
        return tile(label, "—", basis_txt)
    z = analysis.z_vs_null(r)
    return tile(label, f"{100 * r['roi']:+.1f}%",
                f"{r['units']:+.2f}u · {r['w']}–{r['l']} · ± {se_txt(r['roi_se'])} SE"
                f"<br>null {100 * r['roi_null']:+.1f}% · market fav "
                f"{fav_txt(r)}<br>{basis_txt}", sig_cls(z))


def headline_tiles(native, recon):
    """Today's headline row: the goal metric (flat 1u ROI on the lean) on
    the native forward rows and on the latest reconstructed season, never
    pooled; plus how many pregame rows are locked."""
    secs = _sections(native, ledger.empty() if recon is None else recon)
    nat = next((s for s in secs if s["basis"] == "native" and s["book"]), None)
    rec = next((s for s in secs if s["basis"] == "reconstructed" and s["book"]), None)
    tiles = []
    if nat:
        price = "pre" if analysis.roi_summary(nat["h"], "pre") else "close"
        r = dict(analysis.roi_summary(nat["h"], price)).get(analysis.PICK_RULES[0][1])
        where = "pregame price" if price == "pre" else "close"
        tiles.append(_roi_headline(
            "Flat 1u ROI · native lean", r and r[0],
            f"{market.BOOK_NAMES[nat['book']]} {season_txt(nat['season'])} · {where}"))
    else:
        tiles.append(tile("Flat 1u ROI · native lean", "—",
                          "no graded bets yet · pregame-locked rows"))
    if rec:
        r = dict(analysis.roi_summary(rec["h"], "close")).get(analysis.PICK_RULES[0][1])
        tiles.append(_roi_headline(
            "Flat 1u ROI · rebuilt history", r and r[0],
            f"{market.BOOK_NAMES[rec['book']]} {season_txt(rec['season'])} · close · "
            "hindsight"))
        sc = analysis.scoring(rec["h"])
        if sc:
            se = sc["d_logloss_se"]
            z = -sc["d_logloss"] / se if np.isfinite(se) and se > 0 else float("nan")
            tiles.append(tile("Log loss vs the close · rebuilt", f"{sc['d_logloss']:+.4f}",
                              f"model − close ± {se4(se)} (1 SE)<br>negative = model "
                              f"better · n={sc['n']}<br>"
                              f"{resolved_txt(z, 'model better', 'close better')}",
                              sig_cls(z)))
    n = len(native)
    graded = len(ledger.graded(native)) if n else 0
    tiles.append(tile("Native ledger", f"{n}", f"{graded} graded · {n - graded} "
                      "pending<br>written before tip, frozen after"))
    return "<div class='tiles'>" + "".join(tiles) + "</div>"


def render_index(led, today, built, model_ok, recon=None):
    tiles = headline_tiles(led, recon)
    head = ("<h1>NBA composite vs market</h1><p class='lead'>Four-factors "
            "composite gap (home − away, in win-% points), the model's "
            "P(win), and the sportsbook price (DraftKings when listed) with "
            "its vig removed. "
            "<b>Model − market</b> is the lean side's model probability minus "
            "the no-vig market probability; <b>model EV</b> is what the model "
            "claims the posted price is worth. Both are model estimates, not "
            "verified edges — the calibration page tracks whether they hold "
            f"up. Slate <span class='stamp'>{esc(today)}</span> (ET).</p>"
            + tiles)
    if not model_ok:
        return page("NBA composite", "index.html", head +
                    "<p class='note'>No fitted model in <code>model/</code> "
                    "yet. Run the <b>Fit model</b> workflow.</p>", built)
    day = led[led["slate_date"].astype(str) == today] if len(led) else led
    if not len(day):
        return page("NBA composite", "index.html", head +
                    "<p class='note'>No regular-season games scored for this "
                    "date (off day, preseason, or the build ran after tip). "
                    "Exhibition games are tracked separately on the "
                    "<a href='preseason.html'>Preseason</a> page.</p>",
                    built)
    rows = []
    for _, r in day.sort_values("tip_utc").iterrows():
        match = f"{esc(r['away'])} @ {esc(r['home'])}"
        p = pd.to_numeric(r["p_home"], errors="coerce")
        if not np.isfinite(p):
            why = (f"abstain · {r['gp_away']:.0f}/{r['gp_home']:.0f} games "
                   "played" if pd.notna(r["gp_home"])
                   else "abstain · no game log")
            rows.append([tip_et(r["tip_utc"]), match, "—", "—", "—",
                         f"{ml_txt(r['pre_away_ml'])} / {ml_txt(r['pre_home_ml'])}"
                         + book_tag(r["pre_book"]),
                         f"<span class='mut'>{why}</span>", "", ""])
            continue
        lean_home = r["lean"] == r["home"]
        q_home = pd.to_numeric(r["pre_q_home"], errors="coerce")
        ml = r["pre_home_ml"] if lean_home else r["pre_away_ml"]
        q = q_home if lean_home else 1 - q_home
        pl = float(r["p_lean"])
        be = market.implied(ml)
        ev = pl * market.decimal_payout(ml) - 1 if np.isfinite(be) else np.nan
        early_tag = (" <span class='basis'>early · carryover</span>"
                     if pd.notna(r["gp_home"]) and
                     min(r["gp_home"], r["gp_away"]) < nc.MIN_GAMES else
                     " <span class='basis'>injury report</span>"
                     if r.get("model_tag") in AVAIL_TAGS else "")
        b2b = "/".join(x for x, f in ((r["away"], r["away_b2b"]),
                                      (r["home"], r["home_b2b"])) if f == 1) or "—"
        rows.append([
            tip_et(r["tip_utc"]), match, f"{float(r['delta']):+.1f}", esc(b2b),
            f"{pct(1 - p)} / {pct(p)}",
            f"{ml_txt(r['pre_away_ml'])} / {ml_txt(r['pre_home_ml'])}"
            + book_tag(r["pre_book"])
            + (f"<br><span class='mut'>{pct(1 - q_home)} / {pct(q_home)}</span>"
               if np.isfinite(q_home) else ""),
            f"<span class='chip'>{esc(r['lean'])}</span> {pct(pl)}{early_tag}",
            pp(pl - q) if np.isfinite(q) else "—",
            pp(ev) if np.isfinite(ev) else "—",
        ])
    heads = ["Tip ET", "Away @ Home", "Composite Δ", "B2B",
             "Model away / home", "Market away / home<br>no-vig",
             "Lean", "Model − market (pp)", "Model EV (%)"]
    note = ("<p class='note'>Composite Δ is 100 × standardised four-factor gap · "
            "ridge weights, in win-% points; P(home) = σ(a + b·Δ + c·b2b_net). "
            "<b>early · carryover</b>: a team has fewer than "
            f"{nc.MIN_GAMES} games, so last season's log (×{cold_start.RHO}) is "
            "carried in with its own early-season logit; in backtests it "
            "trailed the close by about as much as mid-season games do. "
            "<b>injury report</b>: from game 10 the latest NBA injury report "
            "is folded in (Out/Doubtful players, weighted by last season's "
            "BPM, against how often they played in the rating window). "
            + ("<b>v4</b>: a given composite gap counts for more as the "
               "season goes on (× days since opening night, about double by "
               "April), fitted on earlier seasons. "
               if {MODEL_TAG_V4, MODEL_TAG_V5, MODEL_TAG_V6} & set(ACTIVE_TAGS)
               else "") +
            ("<b>v5</b>: the rating discounts the 3-point shooting of a team's "
             "opponents (mostly luck) and adds roster talent: minutes share × "
             "last season's BPM over the players in each team's last game, so "
             "trades and returns count at once. "
             if {MODEL_TAG_V5, MODEL_TAG_V6} & set(ACTIVE_TAGS) else "") +
            ("<b>v6</b>: each team's own free-throw percentage (same decay), "
             "which the four factors read only as attempts. "
             if MODEL_TAG_V6 in ACTIVE_TAGS else "") +
            "Prices are the moneyline at the snapshot time in the ledger "
            "(DraftKings unless tagged ESPN BET), "
            "refreshed each build until tip and frozen after. A game abstains "
            "only while a team has no games yet (or the early model is not "
            "fitted).</p>")
    return page("NBA composite", "index.html",
                head + table(heads, rows, left=(1, 6)) + note, built)



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
        return f"<p class='note'>{esc(name)}: not fitted.</p>"
    rows = [["Intercept (home edge)", f"{m['intercept']:+.4f}", ""]]
    rows += [[f"<code>{esc(f)}</code>", f"{c:+.4f}",
              f"<span class='mut'>{esc(FEATURE_TXT.get(f, ''))}</span>"]
             for f, c in zip(m["features"], m["coef"])]
    yrs = m.get("years") or []
    meta = (f"{m.get('n_games', '?')} games · seasons ending "
            f"{min(yrs)}–{max(yrs)}" if yrs else f"{m.get('n_games', '?')} games")
    return (f"<h3>{esc(name)}</h3><p class='note'>{meta}</p>"
            + table(["Term", "Logit coef", "Meaning"], rows, left=(0, 2)))


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
    _w, base = load_model()
    avail, early = load_avail(), load_early()
    body = ["<h1>Model</h1><p class='lead'>Goal: flat 1u ROI on the model's "
            "side, judged against the market-correct null and the market "
            "favourite on the same games; log loss and Brier against the close "
            "are the probability check. Active tags: "
            + " / ".join(f"<code>{esc(t)}</code>" for t in ACTIVE_TAGS)
            + f". Full description: <a href='{REPO_URL}MODEL.md'>MODEL.md</a>; "
            f"the quotable readout: <a href='{report.REPORT_NAME}'>"
            f"{report.REPORT_NAME}</a>.</p>"]
    body.append("<h2>Routing</h2>" + table(
        ["Games played (fewer of the two)", "Formula", "Model file"],
        [["0", "abstain (no probability)", "—"],
         [f"1–{nc.MIN_GAMES - 1}", f"carryover logit: last season's log × "
          f"{cold_start.RHO} carried in", "<code>model/logit_early.json</code>"],
         [f"{nc.MIN_GAMES}+, injury report covers the game", "availability logit",
          "<code>model/logit_avail.json</code>"],
         [f"{nc.MIN_GAMES}+, otherwise", "base logit", "<code>model/logit.json</code>"],
         [f"{nc.MIN_GAMES}+, a v5/v6 term missing", "frozen v4 base logit",
          "<code>model/logit_v4.json</code>"]], left=(0, 1, 2)))
    body.append("<h2>Fitted coefficients</h2><p class='note'>P(home) = σ(intercept "
                "+ Σ coef × term). Coefficients are fitted by log loss on "
                "earlier seasons; a change to them or to a term is a new model "
                "tag.</p>")
    body += [_coef_table("Base logit (games 10+)", base),
             _coef_table("Availability logit (games 10+, report covers)", avail),
             _coef_table("Early-season carryover logit (games 1–9)", early)]
    heads = ["Basis · close book · season", "Bets", "W–L", "Units", "ROI ± SE",
             "No-vig q", "Excess pp", "Mkt null", "Fav units", "Fav ROI",
             "LL model − close"]
    for rule, label in analysis.PICK_RULES:
        rows = (_season_table(analysis.season_rows(ledger.graded(native)), "native", rule)
                + _season_table(analysis.season_rows(ledger.graded(recon)),
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
        "are hindsight (that season left out of the fit, graded at the close; the "
        "value side is chosen against the close itself), never forward evidence.</div>")
    return page("NBA model", "model.html", "".join(body), built)

# ------------------------------------------------ ledger + calibration ---
BASIS_LABELS = {"native": "Native (pregame-locked, forward)",
                "reconstructed": "Reconstructed (leave-one-season-out, hindsight)"}
BASIS_BADGES = {"native": "<span class='badge native'>Native · forward</span>",
                "reconstructed": "<span class='badge recon'>Reconstructed · hindsight</span>"}

READ_KEY = (
    "<div class='key'><b>How to read.</b> Judge every ROI against its "
    "<b>null</b>: the ROI if the market's no-vig prices were exactly right. "
    "That is about −4% (the bookmaker's hold), not zero. <b>±</b> is one "
    "standard error (1 SE). Colour marks only gaps of at least "
    f"{SIG:g} SE from the null: <span class='pos'>green</span> above, "
    "<span class='neg'>red</span> below; everything else is left plain. "
    "Shaded columns are the ones to read first."
    "<details><summary>Glossary</summary><dl>"
    "<dt>Lean</dt><dd>The side the model gives ≥ 50% (as recorded on the row).</dd>"
    "<dt>Value side</dt><dd>The side where the model's probability beats the "
    "market's no-vig probability.</dd>"
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
    "</dl></details></div>")


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
    val = summ.get(analysis.PICK_RULES[1][1])
    at = "pregame price" if price == "pre" else "close"
    hind = ("" if native else
            " <span class='basis' title='The value side is chosen against the "
            "close itself, which no bettor knew in advance.'>hindsight price</span>")
    tiles = [_score_tile(analysis.scoring(h)),
             _bet_tile(f"Lean bets · 1u at the {at}", lean),
             _bet_tile(f"Value bets · 1u at the {at}", val, hind)]
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


def _hypotheses(native, recon):
    rows = []
    for hyp, hind, nat in analysis.hypothesis_rows(native, recon):
        def cell(r):
            if not r:
                return ["0", "—", "—", "—"]
            return [str(r["n"]), *_roi_cells(r)]
        h, n = cell(hind), cell(nat)
        h[0] = (f"{h[0]} <span class='mut'>at the "
                f"{analysis.PRICE_NAMES[hyp['hindsight']]}</span>")
        n[0] = (f"{n[0]} <span class='mut'>at the "
                f"{analysis.PRICE_NAMES[hyp['native']]}</span>")
        rows.append([f"<b>{hyp['key']}</b>", esc(hyp["rule"]).replace(' · ', '<br>', 1), *n, *h])
    return (
        "<h2 id='hypotheses'>Pre-registered hypotheses — the forward test</h2>"
        "<p class='note'>H1–H3 fixed on 2026-09-30, H4 and amendment A1 "
        "(H2·F: H2's rule at the first snapshot) on 2026-10-01, all "
        "before any native rows; the thresholds are frozen and every result "
        "is reported here, win or lose. <b>Native</b> bets are graded at a "
        "price that could have been bet: the <b>pregame</b> snapshot is the "
        "last one before tip (near the close); the <b>first snapshot</b> is "
        "the earliest priced one, with the model probability written then "
        "(H4 is H3's rule at the early price, where the hindsight edge "
        "appeared). <b>Hindsight</b> is the same rule on the "
        "reconstructed rows at the price the scan found it (pooled over both "
        "seasons and books, as registered), recomputed on the current "
        "reconstructed model; one season of native rows is not a verdict.</p>"
        + table(["", "Rule", "Native n", "ROI", "Null", "vs null (± 1 SE)",
                 "Hindsight n", "ROI", "Null", "vs null (± 1 SE)"],
                rows, left=(0, 1), key=(3, 5)))


def render_grades(native, recon, built):
    secs = _sections(native, recon)
    body = ["<h1>Ledger</h1><p class='lead'>Every game the model scored, "
            "graded against the final score and the betting market. Each "
            "section answers three questions: are the model's probabilities "
            "better than the market's, and do its leans and its value picks "
            "beat the bookmaker's hold? Native and reconstructed rows, and "
            "each book and season, are shown separately — never pooled.</p>",
            READ_KEY,
            _jump([("hypotheses", "Hypotheses")] + [(s["id"], s["jump"]) for s in secs]),
            _hypotheses(native, recon)]
    for sec in secs:
        h, is_native = sec["h"], sec["basis"] == "native"
        body.append(f"<h2 id='{sec['id']}' aria-label='{esc(sec['label'])}'>"
                    f"{sec['title']}</h2>")
        if not len(h):
            body.append(_native_empty(native) if is_native else
                        "<p class='note'>No graded rows with a closing line yet.</p>")
            continue
        how = ("Written before tip and frozen after" if is_native else
               "Scored afterwards with this season left out of the fit")
        body.append(f"<p class='note'>{how} · {len(h)} graded games on "
                    f"{h['slate_date'].nunique()} slates.</p>")
        body.append(_verdicts(sec))
        _roi_section(body, h, is_native)
        _band_section(body, h, is_native)
        _ats_section(body, h)
        _recent_section(body, h, is_native)
    return page("NBA ledger", "grades.html", "".join(body), built)


def _roi_section(body, h, native):
    """1u flat-bet ROI for each pick rule: model WP, market WP, actual."""
    body.append("<h3 style='font-size:16px'>ROI — one unit on every pick</h3>")
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
    body.append("<p class='note'>At the <b>closing price</b>"
                + ("" if native else ": the value side is picked against the "
                   "close itself, which a bettor would not have known — "
                   "hindsight on price as well as on the model")
                + ". Edge bins are descriptive, not a filter to bet.</p>"
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
                + "</b>. With 16 cells per section, a |z| near 2 somewhere is "
                "expected by chance. Bands are descriptive monitoring "
                "dimensions, not a filter: a band that looks good here is a "
                "hypothesis to test forward, and the hypotheses under test are "
                "the pre-registered ones at the top of this page (the earlier "
                "DraftKings −249 to −130 lean band is retired).</p>")
    for title, sections in parts:
        if len(parts) > 1:
            body.append(f"<p class='note'><b>{title}</b></p>")
        body.append(_band_table(sections))
    body.append("</details>")


def _ats_section(body, h):
    """1u flat spread bets at the closing spread, beside the moneyline ROI."""
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
    body.append("".join(out))
    body.append("<p class='note'>Only rows with a recorded closing spread "
                "(same book as the moneyline close). <b>Model cover %</b> maps "
                "the model's win probability to a margin with σ = "
                f"{analysis.ATS_SIGMA:g} pts; it is a display mapping, not a "
                "new prediction. <b>Cover %</b> is covers ÷ (covers + misses); "
                "pushes refund. The null is minus the spread hold.</p>")
    body.append("</details>")


def _recent_section(body, h, native, n=60):
    recent = h.sort_values(["slate_date", "tip_utc"], ascending=False).head(n)
    val = analysis.picks(recent)
    val = val[val["rule"] == "value"].set_index("game_id") if len(val) else val
    rows = []
    for _, r in recent.iterrows():
        won = int(r["lean_won"]) == 1
        lean_u = market.unit_profit(r["lean_ml"], won)
        v = val.loc[r["game_id"]] if len(val) and r["game_id"] in val.index else None
        vtxt = ("—" if v is None else
                f"{esc(v['side'])} {ml_txt(v['ml'])} "
                f"<span class='{'pos' if v['units'] > 0 else 'neg'}'>"
                f"{v['units']:+.2f}u</span>")
        rows.append([
            esc(r["slate_date"]), f"{esc(r['away'])} @ {esc(r['home'])}",
            f"<span class='chip'>{esc(r['lean'])}</span>", pct(r["lean_p"]),
            pct(r["lean_q"]), ml_txt(r["lean_ml"]),
            f"{int(r['away_pts'])}–{int(r['home_pts'])}",
            f"<span class='{'pos' if won else 'neg'}'>{'W' if won else 'L'} "
            f"{lean_u:+.2f}u</span>",
            vtxt,
        ])
    more = (f"latest {n} of {len(h)}" if len(h) > n else f"all {len(h)}")
    body.append(f"<details class='more'{' open' if native else ''}><summary>"
                f"Game by game <span class='mut'>— {more}, newest first</span>"
                "</summary>")
    body.append(table(["Date", "Away @ Home", "Lean", "Model %", "Market %",
                       "Close ML", "Final (away–home)", "Lean · P/L (1u)",
                       "Value pick · P/L (1u)"], rows, left=(0, 1, 2, 8)))
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
    parts = analysis.book_split(_market_games(native, recon))
    secs = _sections(native, recon)
    jumps = [(_slug("market", b), f"Market · {market.BOOK_NAMES[b]}") for b, _ in parts]
    jumps += [(s["id"], "Model · " + s["jump"]) for s in secs]
    body = ["<h1>Calibration</h1><p class='lead'>When a forecast says 70%, "
            "does it happen 70% of the time? First the <b>market</b> itself "
            "(the no-vig close against results, one section per sportsbook), "
            "then the <b>model</b>, with the market's own forecast on the same "
            "games beside it and both scored against the results.</p>",
            READ_KEY.replace("Judge every ROI against its <b>null</b>: the ROI "
                             "if the market's no-vig prices were exactly right. "
                             "That is about −4% (the bookmaker's hold), not "
                             "zero. ",
                             "Here the null is the diagonal: stated = actual. "),
            _jump(jumps)]
    if not parts:
        body.append("<h2>Market: no-vig close vs actual</h2>"
                    "<p class='note'>No graded games with a closing line yet.</p>")
    for book, hm in parts:
        body.append(f"<h2 id='{_slug('market', book)}'>Market: no-vig close vs "
                    f"actual — <span class='basis'>{esc(market.BOOK_NAMES[book])}"
                    "</span></h2>")
        _market_section(body, hm, first=book == parts[0][0])
    _model_sections(body, secs)
    body.append("<p class='note'>Bet grading by price band (ROI, EV and their "
                "nulls) is on the <a href='grades.html'>Ledger</a>.</p>")
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
        body.append(f"<h2 id='{sec['id']}' aria-label='Model — {esc(sec['label'])}'>"
                    f"Model — {sec['title']}</h2>")
        if not len(h):
            body.append("<p class='note'>No graded model rows with a close yet.</p>")
            continue
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
        first = False
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
    body = ["<h1>Preseason</h1><p class='lead'>Exhibition games, tracked "
            "apart from everything else. The regular-season card abstains "
            "until both teams have played; here each game is scored by the "
            "early-season carryover model on <b>last season's games only</b> "
            f"(tag <code>{esc(PRESEASON_TAG)}</code>). Starters rest and "
            "rotations are experiments, so these rows measure how far the "
            "model's offseason view travels, not its regular-season skill. "
            "They never enter the Ledger, Calibration, hypotheses or any "
            "fit.</p>",
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

    Recorded for research (the model does not use them). The snapshot time is
    taken after the fetch, and a game at or after tip is never touched.
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


def write_pages(native, recon, today, model_ok, pre=None):
    OUT_DIR.mkdir(exist_ok=True)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (OUT_DIR / "index.html").write_text(render_index(native, today, built, model_ok,
                                                     recon))
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
    return text
    (OUT_DIR / ".nojekyll").write_text("")


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
        ACTIVE_TAGS = active_tags(model, load_avail(), load_fallback())

    if not a.render_only:
        native, n = grade(native, today)
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
    log(f"wrote {OUT_DIR}/index.html, grades.html, market-calibration.html, "
        f"model.html, preseason.html, {report.REPORT_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
