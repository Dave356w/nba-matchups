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
  3. Render public/index.html, grades.html and market-calibration.html.

The model is nba_composite.py, unchanged. This file only feeds it pregame game
logs (games strictly before the slate date) and records what it said.
"""
from __future__ import annotations

import argparse
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
import ledger
import market
import nba_composite as nc
import player_availability as pav

ET = ZoneInfo("America/New_York")
OUT_DIR = Path("public")
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
FALLBACK_FILE = "logit_v4.json"
AVAIL_TAGS = (MODEL_TAG_V3, MODEL_TAG_V4_AVAIL, MODEL_TAG_V5_AVAIL)
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
    return model is not None and "talent_diff" in (model.get("features") or [])


def model_tag(model, avail=False):
    """The tag for a row scored by `model` (the availability logit if avail)."""
    if avail:
        return (MODEL_TAG_V5_AVAIL if is_v5(model) else
                MODEL_TAG_V4_AVAIL if has_phase(model) else MODEL_TAG_V3)
    return (MODEL_TAG_V5 if is_v5(model) else
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
def grade(led, today):
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
        for gid, pre_book in zip(rows["game_id"].astype(str), rows["pre_book"]):
            g = games.get(gid)
            if not g or not g["completed"]:
                continue
            try:
                odds = market.pick_close(market.book_odds(gid),
                                         prefer=pre_book if pd.notna(pre_book) else None)
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
        try:
            odds = market.pick_pregame(market.book_odds(g["game_id"]))
            if not odds:
                r["price_note"] = "no DraftKings or ESPN BET moneyline pair"
        except Exception as e:  # noqa: BLE001
            log(f"odds {g['game_id']}: {e!r}")
            odds = {}
            r["price_note"] = f"odds fetch failed: {type(e).__name__}"
        r["pre_book"] = odds.get("book")
        r["pre_home_ml"] = odds.get("cur_home_ml")
        r["pre_away_ml"] = odds.get("cur_away_ml")
        q = market.devig(r["pre_home_ml"], r["pre_away_ml"])
        r["pre_q_home"] = round(q, 5) if np.isfinite(q) else np.nan
        rows.append(r)
    return rows


# ------------------------------------------------------------- rendering ---
CSS = """
:root{--bg:#f7f7f5;--fg:#1c1c1e;--mut:#6b6b70;--card:#fff;--line:#e3e3e0;
--pos:#1f7a4d;--neg:#b3362f;--acc:#1d4ed8;--chip:#eef2ff;--s1:#2a78d6;--s2:#eb6834;
--grid:#ecebe8;--pos-bg:#e8f4ee;--neg-bg:#fbeceb}
@media (prefers-color-scheme:dark){:root{--bg:#121214;--fg:#ececef;--mut:#9a9aa2;
--card:#1b1b1f;--line:#2c2c31;--pos:#4ec38a;--neg:#f07b72;--acc:#8ab4ff;--chip:#23263a;
--s1:#3987e5;--s2:#d95926;--grid:#26262b;--pos-bg:#16271f;--neg-bg:#2c1a19}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 60px}
nav{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px}nav a{color:var(--acc);text-decoration:none}
nav a.on{font-weight:600;color:var(--fg)}
nav.jump{gap:6px;margin:10px 0 4px;font-size:13px}
nav.jump a{border:1px solid var(--line);border-radius:999px;padding:1px 10px;background:var(--card)}
h1{font-size:26px;margin:8px 0 4px}h2{font-size:19px;margin:36px 0 6px;scroll-margin-top:8px}
h3{font-size:15px;margin:16px 0 4px}
.lead,.note{color:var(--mut);max-width:78ch}.note{font-size:13.5px}
.note code{overflow-wrap:anywhere}
.stamp{font-variant-numeric:tabular-nums}
.key{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px;
font-size:13.5px;max-width:78ch;margin:10px 0}
.key summary{cursor:pointer;color:var(--acc);margin-top:4px}
.key dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 12px;margin:8px 0 2px}
.key dt{font-weight:600}.key dd{margin:0;color:var(--mut)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:14px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.tile .l{color:var(--mut);font-size:12.5px}.tile .v{font-size:22px;font-weight:600}
.tile .s{color:var(--mut);font-size:12.5px}
.tile.pos{background:var(--pos-bg)}.tile.neg{background:var(--neg-bg)}
.tile.pos .v{color:var(--pos)}.tile.neg .v{color:var(--neg)}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-size:12px;color:var(--mut);font-weight:600;background:var(--card);position:sticky;top:0}
td.l,th.l{text-align:left}tr:last-child td{border-bottom:0}
th.k,td.k{background:var(--chip)}
.pos{color:var(--pos)}.neg{color:var(--neg)}.mut{color:var(--mut)}
.chip{display:inline-block;background:var(--chip);border-radius:6px;padding:0 6px;font-weight:600}
.basis{display:inline-block;font-size:12px;border:1px solid var(--line);border-radius:6px;padding:0 6px;color:var(--mut)}
.badge{display:inline-block;font-size:12px;font-weight:600;border-radius:6px;padding:1px 7px;
vertical-align:middle;margin-right:6px;border:1px solid var(--line)}
.badge.native{background:var(--pos-bg);color:var(--pos);border-color:transparent}
.badge.recon{background:var(--chip);color:var(--mut)}
details.more{margin:12px 0}details.more>summary{cursor:pointer;font-weight:600;font-size:15px;padding:4px 0}
details.more>summary .mut{font-weight:400;font-size:13px}
.charts{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start;margin:12px 0}
.chart{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;
width:100%;max-width:440px}
.chart svg{display:block;width:100%;height:auto}
.chart .t{font-weight:600;font-size:14px}
.legend{display:flex;gap:14px;font-size:12.5px;color:var(--mut);margin:2px 0 4px}
.legend svg{display:inline-block;vertical-align:-2px;width:auto}
.charts .note{flex:1;min-width:240px;margin:0}
.ax{fill:var(--mut);font-size:11px}.gl{stroke:var(--grid);stroke-width:1}
.diag{stroke:var(--mut);stroke-width:1;stroke-dasharray:4 4}
.eb{stroke-width:2;stroke-linecap:round;opacity:.55}
.s1{fill:var(--s1);stroke:var(--s1)}.s2{fill:var(--s2);stroke:var(--s2)}
.mk{stroke:var(--card);stroke-width:2}
"""


def esc(s):
    return html.escape(str(s))


def page(title, active, body, built):
    links = [("index.html", "Today"), ("grades.html", "Ledger"),
             ("market-calibration.html", "Calibration")]
    nav = "".join(f"<a href='{h}' class='{'on' if h == active else ''}'>{t}</a>"
                  for h, t in links)
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>"
            f"<nav>{nav}</nav>{body}<p class='note'>Built "
            f"<span class='stamp'>{esc(built)}</span> · model "
            + " / ".join(f"<code>{esc(t)}</code>" for t in ACTIVE_TAGS)
            + "</p></main></body></html>")


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


def render_index(led, today, built, model_ok):
    head = ("<h1>NBA composite vs market</h1><p class='lead'>Four-factors "
            "composite gap (home − away, in win-% points), the model's "
            "P(win), and the sportsbook price (DraftKings when listed) with "
            "its vig removed. "
            "<b>Model − market</b> is the lean side's model probability minus "
            "the no-vig market probability; <b>model EV</b> is what the model "
            "claims the posted price is worth. Both are model estimates, not "
            "verified edges — the calibration page tracks whether they hold "
            f"up. Slate <span class='stamp'>{esc(today)}</span> (ET).</p>")
    if not model_ok:
        return page("NBA composite", "index.html", head +
                    "<p class='note'>No fitted model in <code>model/</code> "
                    "yet. Run the <b>Fit model</b> workflow.</p>", built)
    day = led[led["slate_date"].astype(str) == today] if len(led) else led
    if not len(day):
        return page("NBA composite", "index.html", head +
                    "<p class='note'>No regular-season games scored for this "
                    "date (off day, preseason, or the build ran after tip).</p>",
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
               if MODEL_TAG_V4 in ACTIVE_TAGS or MODEL_TAG_V5 in ACTIVE_TAGS
               else "") +
            ("<b>v5</b>: the rating discounts the 3-point shooting of a team's "
             "opponents (mostly luck) and adds roster talent: minutes share × "
             "last season's BPM over the players in each team's last game, so "
             "trades and returns count at once. "
             if MODEL_TAG_V5 in ACTIVE_TAGS else "") +
            "Prices are the moneyline at the snapshot time in the ledger "
            "(DraftKings unless tagged ESPN BET), "
            "refreshed each build until tip and frozen after. A game abstains "
            "only while a team has no games yet (or the early model is not "
            "fitted).</p>")
    return page("NBA composite", "index.html",
                head + table(heads, rows, left=(1, 6)) + note, built)


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
             "Break-even", "Model %", "Market %"]


def _roi_table(sections):
    out = []
    for rule_label, rows in sections:
        out.append(f"<h3>{esc(rule_label)}</h3>")
        out.append(table(
            ["Picks"] + ROI_HEADS,
            [[esc(r["label"]), r["n"], f"{r['w']}–{r['l']}", *_roi_cells(r),
              pct(r["actual"]), pct(r["breakeven"]), pct(r["model_p"]), pct(r["q"])]
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
        f"{pct(r['breakeven'])}", sig_cls(z))


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
        "<p class='note'>H1–H3 fixed on 2026-09-30 and H4 on 2026-10-01, "
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


def write_pages(native, recon, today, model_ok):
    OUT_DIR.mkdir(exist_ok=True)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (OUT_DIR / "index.html").write_text(render_index(native, today, built, model_ok))
    (OUT_DIR / "grades.html").write_text(render_grades(native, recon, built))
    (OUT_DIR / "market-calibration.html").write_text(
        render_calibration(native, recon, built))
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
    write_pages(native, recon, today, model_ok)
    log(f"wrote {OUT_DIR}/index.html, grades.html, market-calibration.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
