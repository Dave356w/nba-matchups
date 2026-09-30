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
AVAIL_TAGS = (MODEL_TAG_V3, MODEL_TAG_V4_AVAIL)
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


def has_phase(model):
    return model is not None and "d_phase" in (model.get("features") or [])


def model_tag(model, avail=False):
    """The tag for a row scored by `model` (the availability logit if avail)."""
    if avail:
        return MODEL_TAG_V4_AVAIL if has_phase(model) else MODEL_TAG_V3
    return MODEL_TAG_V4 if has_phase(model) else MODEL_TAG


def active_tags(model, avail_model):
    """Tags the current model files produce (page footer, ledger split)."""
    tags = [model_tag(model)]
    if avail_model is not None:
        tags.append(model_tag(avail_model, avail=True))
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
               opening=None):
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
    phase = nc.season_phase(date, nc.season_opening(logs) if opening is None
                            else opening)
    vals = {"delta": d, "b2b_net": int(ra == 0) - int(rh == 0),
            "d_phase": d * phase, **(avail or {})}
    p = float(nc.predict(use, [[vals[f] for f in use["features"]]])[0])
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
    if avail_model is not None:
        try:
            live = pav.LiveAvailability(season_for(today), today,
                                        now.astimezone(ET).replace(tzinfo=None))
            log(f"availability: report {live.report_time} "
                f"({0 if live.report is None else len(live.report)} rows); "
                f"BPM covers {100 * live.bpm_minutes:.0f}% of minutes")
        except Exception as e:  # noqa: BLE001 - v3 falls back to v2
            log(f"availability failed ({e!r}); rows fall back to v2")
    opening = nc.season_opening(logs)
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
                            opening=opening))
        try:
            odds = market.pick_pregame(market.book_odds(g["game_id"]))
        except Exception as e:  # noqa: BLE001
            log(f"odds {g['game_id']}: {e!r}")
            odds = {}
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
--pos:#1f7a4d;--neg:#b3362f;--acc:#1d4ed8;--chip:#eef2ff}
@media (prefers-color-scheme:dark){:root{--bg:#121214;--fg:#ececef;--mut:#9a9aa2;
--card:#1b1b1f;--line:#2c2c31;--pos:#4ec38a;--neg:#f07b72;--acc:#8ab4ff;--chip:#23263a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 60px}
nav{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px}nav a{color:var(--acc);text-decoration:none}
nav a.on{font-weight:600;color:var(--fg)}
h1{font-size:26px;margin:8px 0 4px}h2{font-size:19px;margin:32px 0 6px}
.lead,.note{color:var(--mut);max-width:78ch}.note{font-size:13.5px}
.stamp{font-variant-numeric:tabular-nums}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:14px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.tile .l{color:var(--mut);font-size:12.5px}.tile .v{font-size:22px;font-weight:600}
.tile .s{color:var(--mut);font-size:12.5px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-size:12px;color:var(--mut);font-weight:600;background:var(--card);position:sticky;top:0}
td.l,th.l{text-align:left}tr:last-child td{border-bottom:0}
.pos{color:var(--pos)}.neg{color:var(--neg)}.mut{color:var(--mut)}
.chip{display:inline-block;background:var(--chip);border-radius:6px;padding:0 6px;font-weight:600}
.basis{display:inline-block;font-size:12px;border:1px solid var(--line);border-radius:6px;padding:0 6px;color:var(--mut)}
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


def tile(label, value, sub=""):
    return (f"<div class='tile'><div class='l'>{label}</div>"
            f"<div class='v'>{value}</div><div class='s'>{sub}</div></div>")


def table(heads, rows, left=(0,)):
    th = "".join(f"<th class='{'l' if i in left else ''}'>{h}</th>"
                 for i, h in enumerate(heads))
    body = "".join(
        "<tr>" + "".join(f"<td class='{'l' if i in left else ''}'>{c}</td>"
                         for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<div class='wrap'><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"


def tip_et(s):
    t = ledger.parse_utc(s)
    return t.astimezone(ET).strftime("%-I:%M %p") if t else "—"


def book_tag(book):
    """Visible label for any price not from DraftKings (the default book)."""
    if pd.isna(book) or book in ("", "dk"):
        return ""
    return f" <span class='basis'>{esc(market.BOOK_NAMES.get(book, book))}</span>"


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
               if MODEL_TAG_V4 in ACTIVE_TAGS else "") +
            "Prices are the moneyline at the snapshot time in the ledger "
            "(DraftKings unless tagged ESPN BET), "
            "refreshed each build until tip and frozen after. A game abstains "
            "only while a team has no games yet (or the early model is not "
            "fitted).</p>")
    return page("NBA composite", "index.html",
                head + table(heads, rows, left=(1, 6)) + note, built)


def _basis_split(native, recon):
    """[(label, rows)]: one section per basis x closing book, never pooled."""
    out = []
    for label, df in (("Native (pregame-locked, forward)", native),
                      ("Reconstructed (leave-one-season-out, hindsight)", recon)):
        h = analysis.with_close(ledger.graded(df))
        parts = analysis.book_split(h)
        if not parts:
            out.append((label, h))
        for book, hb in parts:
            out.append((f"{label} · {market.BOOK_NAMES[book]} close", hb))
    return out


def render_grades(native, recon, built):
    body = ["<h1>Ledger</h1><p class='lead'>The model's lean (the side it "
            "gives ≥ 50%) graded against final scores. Native rows were "
            "recorded before tip; reconstructed rows were scored after the "
            "fact with the season held out of the fit and are shown "
            "separately — never pooled.</p>"]
    for label, h in _basis_split(native, recon):
        body.append(f"<h2>{esc(label)}</h2>")
        if not len(h):
            body.append("<p class='note'>No graded rows with a closing line yet.</p>")
            continue
        _, pooled = analysis.lean_by_price(h)
        tiles = [
            tile("Lean record", f"{pooled['w']}–{pooled['l']}",
                 f"{pct(pooled['win'])} · n={pooled['n']} · "
                 f"{h['slate_date'].nunique()} slates"),
            tile("vs no-vig close", f"{100 * pooled['excess']:+.1f} pp",
                 f"± {100 * pooled['se']:.1f} (1 SE)"),
            tile("Flat units @ close", f"{pooled['units']:+.2f}u",
                 f"ROI {100 * pooled['roi']:+.1f}%"),
        ]
        if label.startswith("Native"):
            c = analysis.clv(h)
            if c:
                tiles.append(tile("Closing-line value", f"{100 * c['mean']:+.2f} pp",
                                  f"± {100 * c['se']:.2f} · beat close "
                                  f"{pct(c['beat'], 0)} · n={c['n']}"))
        body.append("<div class='tiles'>" + "".join(tiles) + "</div>")
        _roi_section(body, h, label.startswith("Native"))
        recent = h.sort_values(["slate_date", "tip_utc"], ascending=False).head(60)
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
                f"<span class='{'pos' if won else 'neg'}'>{'W' if won else 'L'}</span>",
                f"<span class='{'pos' if lean_u > 0 else 'neg'}'>{lean_u:+.2f}u</span>",
                vtxt,
            ])
        body.append(table(["Date", "Away @ Home", "Lean", "Model WP", "Market WP (no-vig)",
                           "Close ML", "Final", "Result", "Lean P/L (1u)",
                           "Value pick · P/L (1u)"], rows, left=(0, 1, 2, 9)))
        if len(h) > 60:
            body.append(f"<p class='note'>Latest 60 of {len(h)} rows; the full "
                        "record is in the CSV under <code>data/</code>.</p>")
    return page("NBA ledger", "grades.html", "".join(body), built)


def _roi_table(sections):
    out = []
    for rule_label, rows in sections:
        out.append(f"<h3 style='font-size:15px;margin:14px 0 4px'>{esc(rule_label)}</h3>")
        out.append(table(
            ["Picks", "n", "W–L", "Model WP", "Market WP (no-vig)", "Break-even",
             "Actual", "Units (1u flat)", "ROI", "± SE", "ROI null"],
            [[esc(r["label"]), r["n"], f"{r['w']}–{r['l']}", pct(r["model_p"]),
              pct(r["q"]), pct(r["breakeven"]), pct(r["actual"]),
              f"{r['units']:+.2f}u",
              f"<span class='{'pos' if r['roi'] > r['roi_null'] else 'neg'}'>"
              f"{100 * r['roi']:+.1f}%</span>",
              se_txt(r['roi_se']), f"{100 * r['roi_null']:+.1f}%"]
             for r in rows], left=(0,)))
    return "".join(out)


def _roi_section(body, h, native):
    """1u flat-bet ROI for each pick rule: model WP, market WP, actual."""
    body.append("<h3 style='font-size:16px;margin:18px 0 4px'>ROI — one unit "
                "on every pick</h3>")
    if native:
        pre = _roi_table(analysis.roi_summary(h, price="pre"))
        if pre:
            body.append("<p class='note'>At the <b>pregame snapshot price</b> — "
                        "what could have been bet when the row was written.</p>" + pre)
    body.append("<p class='note'>At the <b>closing price</b>"
                + ("" if native else ": the value side is picked against the "
                   "close itself, which a bettor would not have known — "
                   "hindsight on price as well as on the model")
                + ".</p>" + _roi_table(analysis.roi_summary(h, price="close")))
    body.append("<p class='note'><b>Model WP</b> and <b>market WP</b> are the "
                "picked side's mean model probability and no-vig market "
                "probability; <b>break-even</b> is the win rate the posted "
                "price needs. <b>ROI null</b> is the ROI expected if the market "
                "is right (about minus the hold, ~−4%): an ROI is judged against "
                "it, and against its ± SE, not against zero. Edge bins are "
                "descriptive, not a filter to bet.</p>")
    _band_section(body, h, native)
    _ats_section(body, h)


def _band_table(sections):
    out = []
    for rule_label, rows in sections:
        out.append(f"<h3 style='font-size:15px;margin:14px 0 4px'>{esc(rule_label)}</h3>")
        out.append(table(
            ["Picked price", "n", "W–L", "Model WP", "Market WP (no-vig)",
             "Break-even", "Actual", "EV (pp)", "EV null (pp)", "Units (1u flat)",
             "ROI", "± SE", "ROI null", "z vs null"],
            [[esc(r["label"]), r["n"], f"{r['w']}–{r['l']}", pct(r["model_p"]),
              pct(r["q"]), pct(r["breakeven"]), pct(r["actual"]),
              pp(r["ev"], cls=False), f"{100 * r['ev_null']:+.1f}",
              f"{r['units']:+.2f}u",
              f"<span class='{'pos' if r['roi'] > r['roi_null'] else 'neg'}'>"
              f"{100 * r['roi']:+.1f}%</span>",
              se_txt(r['roi_se']), f"{100 * r['roi_null']:+.1f}%",
              "—" if not np.isfinite(r["z"]) else f"{r['z']:+.1f}"]
             for r in rows], left=(0,)))
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
    body.append("<h3 style='font-size:16px;margin:18px 0 4px'>ROI by price "
                "band — one unit on every pick</h3>")
    body.append("<p class='note'>Graded at the <b>"
                + ("pregame snapshot price" if native else "closing price")
                + "</b>, banded by the picked side's moneyline. <b>EV</b> = "
                "actual − break-even; its <b>null</b> is q − break-even (about "
                "minus the hold), not zero. <b>z vs null</b> = (ROI − ROI null) "
                "÷ SE; with ~16 cells per table, one |z| near 2 is expected by "
                "chance. Bands are descriptive monitoring dimensions, not a "
                "filter. The hindsight rows left one hypothesis to test "
                "forward: DraftKings leans at −249 to −130.</p>")
    for title, sections in parts:
        if len(parts) > 1:
            body.append(f"<p class='note'><b>{title}</b></p>")
        body.append(_band_table(sections))


def _ats_section(body, h):
    """1u flat spread bets at the closing spread, beside the moneyline ROI."""
    sections = analysis.ats_summary(h)
    if not sections:
        return
    body.append("<h3 style='font-size:16px;margin:18px 0 4px'>Against the "
                "spread — one unit on every pick at the closing spread</h3>")
    out = []
    for rule_label, rows in sections:
        out.append(f"<h3 style='font-size:15px;margin:14px 0 4px'>{esc(rule_label)}</h3>")
        out.append(table(
            ["Picks", "n", "W–L–P", "Model cover P", "Market cover P (no-vig)",
             "Break-even", "Actual", "Units (1u flat)", "ROI", "± SE", "ROI null"],
            [[esc(r["label"]), r["n"], f"{r['w']}–{r['l']}–{r['push']}",
              pct(r["model_p"]), pct(r["q"]), pct(r["breakeven"]),
              pct(r["actual"]), f"{r['units']:+.2f}u",
              f"<span class='{'pos' if r['roi'] > r['roi_null'] else 'neg'}'>"
              f"{100 * r['roi']:+.1f}%</span>",
              se_txt(r['roi_se']), f"{100 * r['roi_null']:+.1f}%"]
             for r in rows], left=(0,)))
    body.append("".join(out))
    body.append("<p class='note'>Only rows with a recorded closing spread "
                "(same book as the moneyline close). <b>Model cover P</b> maps "
                "the model's win probability to a margin with σ = "
                f"{analysis.ATS_SIGMA:g} pts; it is a display mapping, not a "
                "new prediction. <b>Actual</b> is covers ÷ (covers + misses); "
                "pushes refund. Judge the ROI against the <b>ROI null</b> "
                "(minus the spread hold) and its ± SE, not zero.</p>")


def _calib_cell(a):
    if not a:
        return "<span class='mut'>—</span>"
    return (f"{pct(a['actual'])} <span class='mut'>vs {pct(a['implied'])}</span> "
            f"{pp(a['diff'])} <span class='mut'>±{100 * a['se']:.1f} · n={a['n']}</span>")


def render_calibration(native, recon, built):
    body = ["<h1>Calibration</h1><p class='lead'>Implied versus actual. "
            "First the <b>market</b> itself (devigged close vs results, one "
            "section per sportsbook), "
            "then the <b>model</b>: its probabilities against results with the "
            "market's probability on the same games, proper scores against the "
            "close, and its leans graded at the closing price.</p>"]

    # 1. Market calibration: model-independent, so one pool of distinct games.
    allg = pd.concat([d for d in (native, recon) if len(d)], ignore_index=True) \
        if (len(native) or len(recon)) else ledger.empty()
    g = ledger.graded(allg)
    if len(g):
        g = g.assign(_nat=(g["basis"] == "native")).sort_values(
            "_nat", ascending=False).drop_duplicates("game_id")
        g = g.assign(p_home=g["p_home"].fillna(0.5))   # market view needs no model
    hm_all = analysis.with_close(g)
    parts = analysis.book_split(hm_all)
    if not parts:
        body.append("<h2>Market: devigged close vs actual</h2>")
        body.append("<p class='note'>No graded games with a closing line yet.</p>")
    for book, hm in parts:
        body.append("<h2>Market: devigged close vs actual — "
                    f"<span class='basis'>{esc(market.BOOK_NAMES[book])}</span></h2>")
        _market_section(body, hm)
    _model_sections(body, native, recon)
    return page("NBA calibration", "market-calibration.html", "".join(body), built)


def _market_section(body, hm):
    rows_, totals = analysis.market_calibration(hm)
    if rows_:
        nn = int((hm["basis"] == "native").sum())
        tiles = [tile(lab, pct(t["actual"]),
                      f"vs {pct(t['implied'])} implied ({100 * t['diff']:+.1f} ± "
                      f"{100 * t['se']:.1f}) · n={t['n']}")
                 for key, lab in (("favourite", "Favourites"), ("home", "Home sides"))
                 if (t := totals.get(key))]
        body.append("<div class='tiles'>" + "".join(tiles) + "</div>")
        body.append(table(["Price rung", "Home side", "Away side", "Both sides"],
                          [[esc(r["rung"]), _calib_cell(r["home"]),
                            _calib_cell(r["away"]), _calib_cell(r["all"])]
                           for r in rows_]))
        body.append(f"<p class='note'>{len(hm)} games ({nn} native, "
                    f"{len(hm) - nn} reconstructed). Two observations per game, "
                    "one per side; ± is one SE under correct prices, so a gap "
                    "under ~2± is indistinguishable from fair. No both-sides "
                    "total: the sides sum to 1 and one wins, so it is 50% by "
                    "construction. Favourites asks once per game.</p>")
    else:
        body.append("<p class='note'>No graded games with a closing line yet.</p>")


def _model_sections(body, native, recon):
    """Model vs market, one basis x closing book at a time."""
    for label, h in _basis_split(native, recon):
        body.append(f"<h2>Model — <span class='basis'>{esc(label)}</span></h2>")
        if not len(h):
            body.append("<p class='note'>No graded model rows with a close yet.</p>")
            continue
        s = analysis.scoring(h)
        body.append("<div class='tiles'>" + "".join([
            tile("Brier (model / market)",
                 f"{s['model']['brier']:.4f} / {s['market']['brier']:.4f}",
                 f"Δ {s['d_brier']:+.4f} ± {s['d_brier_se']:.4f} · n={s['n']}"),
            tile("Log loss (model / market)",
                 f"{s['model']['logloss']:.4f} / {s['market']['logloss']:.4f}",
                 f"Δ {s['d_logloss']:+.4f} ± {s['d_logloss_se']:.4f}"),
            tile("Accuracy (model / market fav)",
                 f"{pct(s['model']['acc'])} / {pct(s['market']['acc'])}",
                 "same games"),
        ]) + "</div>")
        body.append("<p class='note'>Δ = model − market per game (negative = "
                    "model scored better). An interval spanning 0 means this "
                    "sample has not separated them.</p>")
        cal = analysis.model_calibration(h)
        body.append(table(
            ["Model P(home)", "n", "Model mean", "Market mean (same games)",
             "Actual", "Actual − model"],
            [[f"{c['lo']:.1f}–{c['hi']:.1f}", c["n"], pct(c["model"]),
              pct(c["market"]), pct(c["actual"]),
              f"{pp(c['actual'] - c['model'])} <span class='mut'>±{100 * c['se']:.1f}</span>"]
             for c in cal]))
        bands, pooled = analysis.lean_by_price(h)
        body.append("<p class='note' style='margin-top:14px'>Leans by the lean "
                    "side's closing moneyline. <b>Excess</b> = win% − mean no-vig "
                    "q. <b>EV</b> = win% − mean break-even at the posted price; "
                    "its <b>null</b> (market correct) is q − break-even, i.e. "
                    "minus the hold — not zero. Bands are descriptive monitoring "
                    "dimensions, not a filter.</p>")
        body.append(table(
            ["Close ML band", "n", "W–L", "Win %", "Model p", "Mean q",
             "Excess (pp)", "± SE", "EV (pp)", "Null (pp)", "Units", "ROI"],
            [[esc(b["band"]) if b["band"] != "Pooled" else "<b>Pooled</b>",
              b["n"], f"{b['w']}–{b['l']}", pct(b["win"]), pct(b["model_p"]),
              f"{b['q']:.3f}", pp(b["excess"]), f"{100 * b['se']:.1f}",
              pp(b["ev"]), f"{100 * b['ev_null']:+.1f}",
              f"{b['units']:+.2f}", f"{100 * b['roi']:+.1f}%"]
             for b in bands + [pooled]]))


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
        ACTIVE_TAGS = active_tags(model, load_avail())

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
