#!/usr/bin/env python3
"""Win Shares per 48 as the roster value: full ESPN box lines and BBR's
Win Shares, season to date (owner's request, 2026-10-09).

  python research/win_shares.py fetch --seasons 2016-2019 2021-2026 [--workers 8]
  python research/win_shares.py check --season 2025

talent_diff (model v5/v6) values each player by last season's BPM. This
module supplies the WS/48 alternatives research/team_quality.py tests:

  last-season WS/48   BBR's advanced table for season y - 1 (the same page
                      as the BPM), shrunk toward replacement by MP / (MP +
                      player_availability.BPM_SHRINK_MP) like the BPM.
  season to date      Win Shares computed from ESPN box lines of games
                      strictly before the date (Oliver's points produced /
                      individual defensive rating, BBR's marginal-points
                      formulas, team and league context to date), per
                      team stint, summed per player.
  Bayes               last season's shrunk WS/48 as the prior, updated as
                      minutes accumulate: (M0 x prior + MP x obs) / (M0 +
                      MP), M0 = WS_PRIOR_MP (750, the PDF's constant).

Replacement level: the WS/48 that BPM's replacement level (-2.0) maps to,
from a minutes-weighted line of WS/48 on BPM over season y - 1's table
(players with 250+ MP); unmatched players (rookies) sit there, value 0.
Values are WS/48 above replacement, so minutes share x value sums to
roughly team wins added per game, the scale of talent_diff's role x BPM.

`fetch` writes research/output/box_full_<y>.csv (one row per player-game:
minutes, +/-, PTS, FG, 3P, FT, ORB, DRB, AST, STL, BLK, TOV, PF and the
team's total turnovers) and, if absent, box_<y>.csv in
player_availability.fetch_box's format from the same pages. `check`
compares full-season WS from the box lines with BBR's table. Research
only; the model reads none of it.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import market  # noqa: E402
import nba_composite as nc  # noqa: E402
import player_availability as pav  # noqa: E402

WS_PRIOR_MP = 750.0         # in-season minutes for half weight vs the prior
REPLACEMENT_MIN_MP = 250.0  # players in the WS/48-on-BPM line
STATS = ["pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta", "orb", "drb", "ast",
         "stl", "blk", "tov", "pf"]
_KEYS = {"points": "pts", "rebounds": None, "assists": "ast", "turnovers": "tov",
         "steals": "stl", "blocks": "blk", "offensiveRebounds": "orb",
         "defensiveRebounds": "drb", "fouls": "pf"}
_PAIRS = {"fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
          "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("tpm", "tpa"),
          "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta")}


# ----------------------------------------------------------- box lines -----
def _num(x):
    try:
        v = float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def parse_full(js):
    """ESPN summary JSON -> [{team, player_id, name, minutes, pm, *STATS}]
    plus {team: total turnovers (players + team)}."""
    rows, team_tov = [], {}
    for t in (js.get("boxscore") or {}).get("teams") or []:
        team = market.bbr_code((t.get("team") or {}).get("abbreviation"))
        for s in t.get("statistics") or []:
            if s.get("name") == "totalTurnovers":
                team_tov[team] = _num(s.get("displayValue"))
    for t in (js.get("boxscore") or {}).get("players") or []:
        team = market.bbr_code((t.get("team") or {}).get("abbreviation"))
        for block in t.get("statistics") or []:
            keys = block.get("keys") or []
            for a in block.get("athletes") or []:
                ath = a.get("athlete") or {}
                if not ath.get("id"):
                    continue
                st = a.get("stats") or []
                r = dict(team=team, player_id=str(ath.get("id")),
                         name=ath.get("displayName"), minutes=0.0, pm=0.0,
                         **{c: 0.0 for c in STATS})
                for k, v in zip(keys, st):
                    if k == "minutes":
                        r["minutes"] = pav._minutes(v)
                    elif k == "plusMinus":
                        r["pm"] = pav._pm(v)
                    elif k in _PAIRS and "-" in str(v):
                        m, n = str(v).split("-", 1)
                        r[_PAIRS[k][0]], r[_PAIRS[k][1]] = _num(m), _num(n)
                    elif _KEYS.get(k):
                        r[_KEYS[k]] = _num(v)
                rows.append(r)
    return rows, team_tov


def _game_rows(g, ds):
    js = market.get_json(pav.SUMMARY.format(eid=g["game_id"]))
    players, team_tov = parse_full(js)
    out = []
    for pl in players:
        if pl["team"] not in (g["home"], g["away"]):
            continue
        home = pl["team"] == g["home"]
        out.append(dict(game_id=g["game_id"], date=pd.Timestamp(ds), team=pl["team"],
                        opp=g["away"] if home else g["home"], home=home,
                        margin=(g["home_pts"] - g["away_pts"]) * (1 if home else -1),
                        team_tov=team_tov.get(pl["team"], np.nan),
                        **{k: v for k, v in pl.items() if k != "team"}))
    return out


def fetch_full_box(y, cache_dir=pav.DEFAULT_CACHE, workers=8):
    """Every completed regular-season game of season y, full box lines
    (cached in box_full_<y>.csv); also writes box_<y>.csv if absent."""
    path = os.path.join(cache_dir, f"box_full_{y}.csv")
    if os.path.exists(path):
        return load_full(y, cache_dir)
    games = []
    for d in pav.season_dates(y):
        ds = d.strftime("%Y-%m-%d")
        try:
            sb = market.scoreboard(ds)
        except Exception as e:  # noqa: BLE001
            print(f"{ds}: scoreboard failed {e!r}", flush=True)
            continue
        games += [(g, ds) for g in sb
                  if g["season_type"] == market.REGULAR_SEASON and g["completed"]
                  and g["home"] in market.BBR_TEAMS and g["away"] in market.BBR_TEAMS]

    def one(x):
        try:
            return _game_rows(*x)
        except Exception as e:  # noqa: BLE001
            print(f"{x[0]['game_id']}: summary failed {e!r}", flush=True)
            return []
    with ThreadPoolExecutor(workers) as ex:
        rows = [r for got in ex.map(one, games) for r in got]
    df = pd.DataFrame(rows)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(path, index=False)
    slim = os.path.join(cache_dir, f"box_{y}.csv")
    if not os.path.exists(slim):
        pav.clean_box(df[["game_id", "date", "team", "opp", "home", "margin",
                          "player_id", "name", "minutes", "pm"]].copy()
                      ).to_csv(slim, index=False)
    print(f"box_full {y}: {df['game_id'].nunique()} of {len(games)} games, "
          f"{len(df)} player rows", flush=True)
    return load_full(y, cache_dir)


def load_full(y, cache_dir=pav.DEFAULT_CACHE):
    df = pd.read_csv(os.path.join(cache_dir, f"box_full_{y}.csv"),
                     dtype={"game_id": str, "player_id": str, "name": str},
                     keep_default_na=False, parse_dates=["date"])
    df = pav.clean_box(df)
    for c in STATS + ["team_tov"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[STATS] = df[STATS].fillna(0.0)
    return df


# ---------------------------------------------------------- Win Shares -----
def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b != 0)


def team_games(full):
    """One row per team-game: team totals (players summed; TOV = the team's
    total turnovers when ESPN lists it), minutes, and the opponent's."""
    g = full.groupby(["game_id", "date", "team", "opp"], as_index=False)[
        STATS + ["minutes"]].sum()
    tt = full.groupby(["game_id", "team"])["team_tov"].first()
    tov = tt.reindex(pd.MultiIndex.from_frame(g[["game_id", "team"]])).to_numpy()
    g["tov"] = np.where(np.isfinite(tov), tov, g["tov"])
    o = g[["game_id", "team"] + STATS + ["minutes"]].rename(
        columns={"team": "opp", **{c: "o_" + c for c in STATS + ["minutes"]}})
    return g.merge(o, on=["game_id", "opp"], how="inner")


def possessions(T, O):
    """BBR's team possessions estimate from team (T) and opponent (O) totals."""
    def half(a, b):
        return (a["fga"] + 0.4 * a["fta"]
                - 1.07 * _div(a["orb"], a["orb"] + b["drb"]) * (a["fga"] - a["fgm"])
                + a["tov"])
    return 0.5 * (half(T, O) + half(O, T))


def win_shares(P, T, O, L):
    """Offensive and defensive Win Shares (BBR's formulas) for player-stint
    totals P against their team's (T) and opponents' (O) totals over the same
    games and league totals L (dicts of arrays / scalars; T and O aligned
    with P). Returns (ows, dws)."""
    mp, tmp = P["minutes"], T["minutes"]
    # --- offence: Oliver's scoring possessions and points produced
    q1 = _div(mp, tmp / 5) * 1.14 * _div(T["ast"] - P["ast"], T["fgm"])
    q2 = (_div(_div(T["ast"], tmp) * mp * 5 - P["ast"],
               _div(T["fgm"], tmp) * mp * 5 - P["fgm"])
          * (1 - _div(mp, tmp / 5)))
    q_ast = q1 + q2
    shoot = _div(P["pts"] - P["ftm"], 2 * P["fga"])
    fg_part = P["fgm"] * (1 - 0.5 * shoot * q_ast)
    ast_part = 0.5 * _div((T["pts"] - T["ftm"]) - (P["pts"] - P["ftm"]),
                          2 * (T["fga"] - P["fga"])) * P["ast"]
    ft_miss = (1 - _div(P["ftm"], P["fta"])) ** 2
    ft_part = np.where(P["fta"] > 0, (1 - ft_miss) * 0.4 * P["fta"], 0.0)
    t_scposs = T["fgm"] + (1 - (1 - _div(T["ftm"], T["fta"])) ** 2) * T["fta"] * 0.4
    t_orbp = _div(T["orb"], T["orb"] + (O["orb"] + O["drb"] - O["orb"]))
    t_play = _div(t_scposs, T["fga"] + T["fta"] * 0.4 + T["tov"])
    t_orbw = _div((1 - t_orbp) * t_play,
                  (1 - t_orbp) * t_play + t_orbp * (1 - t_play))
    orb_part = P["orb"] * t_orbw * t_play
    keep = 1 - _div(T["orb"], t_scposs) * t_orbw * t_play
    sc_poss = (fg_part + ast_part + ft_part) * keep + orb_part
    fgx = (P["fga"] - P["fgm"]) * (1 - 1.07 * t_orbp)
    ftx = np.where(P["fta"] > 0, ft_miss * 0.4 * P["fta"], 0.0)
    tot_poss = sc_poss + fgx + ftx + P["tov"]
    pp_fg = 2 * (P["fgm"] + 0.5 * P["tpm"]) * (1 - 0.5 * shoot * q_ast)
    pp_ast = (2 * _div(T["fgm"] - P["fgm"] + 0.5 * (T["tpm"] - P["tpm"]),
                       T["fgm"] - P["fgm"])
              * 0.5 * _div((T["pts"] - T["ftm"]) - (P["pts"] - P["ftm"]),
                           2 * (T["fga"] - P["fga"])) * P["ast"])
    pp_orb = P["orb"] * t_orbw * t_play * _div(T["pts"], t_scposs)
    pprod = (pp_fg + pp_ast + P["ftm"]) * keep + pp_orb
    # --- league and marginal points per win
    t_poss = possessions(T, O)
    pace_t = 48 * _div(2 * t_poss, 2 * (tmp / 5))
    lg_ppp = L["pts"] / L["poss"]
    lg_pace = 48 * L["poss"] / (L["minutes"] / 5)
    mppw = 0.32 * L["pts"] / L["games"] * _div(pace_t, lg_pace)
    ows = _div(pprod - 0.92 * lg_ppp * tot_poss, mppw)
    # --- defence: individual defensive rating
    dor = _div(O["orb"], O["orb"] + T["drb"])
    dfg = _div(O["fgm"], O["fga"])
    fmwt = _div(dfg * (1 - dor), dfg * (1 - dor) + (1 - dfg) * dor)
    stops1 = P["stl"] + P["blk"] * fmwt * (1 - 1.07 * dor) + P["drb"] * (1 - fmwt)
    stops2 = ((_div(O["fga"] - O["fgm"] - T["blk"], tmp) * fmwt * (1 - 1.07 * dor)
               + _div(O["tov"] - T["stl"], tmp)) * mp
              + _div(P["pf"], T["pf"]) * 0.4 * O["fta"] * (1 - _div(O["ftm"], O["fta"])) ** 2)
    stop_pct = _div((stops1 + stops2) * O["minutes"], t_poss * mp)
    t_drtg = 100 * _div(O["pts"], t_poss)
    d_pts = _div(O["pts"], O["fgm"] + (1 - (1 - _div(O["ftm"], O["fta"])) ** 2)
                 * O["fta"] * 0.4)
    drtg = t_drtg + 0.2 * (100 * d_pts * (1 - stop_pct) - t_drtg)
    dws = _div(_div(mp, tmp) * t_poss * (1.08 * lg_ppp - drtg / 100), mppw)
    return ows, dws


def _league(tg):
    T = {c: tg[c].to_numpy(float) for c in STATS + ["minutes"]}
    O = {c: tg["o_" + c].to_numpy(float) for c in STATS + ["minutes"]}
    return dict(pts=float(tg["pts"].sum()), minutes=float(tg["minutes"].sum()),
                poss=float(possessions(T, O).sum()), games=float(len(tg)))


def season_ws(full, through=None):
    """Player Win Shares and minutes over games before `through` (all games
    if None): DataFrame [player_id, name, team, minutes, ows, dws, ws], one
    row per team stint."""
    f = full if through is None else full[full["date"] < pd.Timestamp(through)]
    f = f[f["minutes"] > 0]
    tg = team_games(f)
    if not len(tg):
        return pd.DataFrame(columns=["player_id", "name", "team", "minutes",
                                     "ows", "dws", "ws"])
    L = _league(tg)
    tot = tg.groupby("team")[STATS + ["minutes"] + ["o_" + c for c in STATS + ["minutes"]]].sum()
    p = f.groupby(["player_id", "team"], as_index=False).agg(
        name=("name", "last"), **{c: (c, "sum") for c in STATS + ["minutes"]})
    tt = tot.loc[p["team"]]
    T = {c: tt[c].to_numpy(float) for c in STATS + ["minutes"]}
    O = {c: tt["o_" + c].to_numpy(float) for c in STATS + ["minutes"]}
    P = {c: p[c].to_numpy(float) for c in STATS + ["minutes"]}
    ows, dws = win_shares(P, T, O, L)
    p["ows"], p["dws"] = ows, dws
    p["ws"] = ows + dws
    return p[["player_id", "name", "team", "minutes", "ows", "dws", "ws"]]


def ws_history(full):
    """player_id -> (dates, ws, mp): season-to-date Win Shares and minutes
    over games strictly before each game date of the season (summed over
    team stints). Look up date d with searchsorted(dates, d, 'right') - 1
    on the snapshot dates, which are the season's game dates; the snapshot
    at date d covers games before d only."""
    dates = sorted(full["date"].unique())
    snaps = {}
    for d in dates:
        s = season_ws(full, through=d)
        if len(s):
            snaps[pd.Timestamp(d)] = s.groupby("player_id")[["ws", "minutes"]].sum()
    hist = {}
    for d, s in snaps.items():
        for pid, ws, mp in zip(s.index, s["ws"].to_numpy(), s["minutes"].to_numpy()):
            hist.setdefault(pid, ([], [], []))
            hist[pid][0].append(np.datetime64(d))
            hist[pid][1].append(ws)
            hist[pid][2].append(mp)
    return {pid: (np.array(a), np.array(b, float), np.array(c, float))
            for pid, (a, b, c) in hist.items()}


# ------------------------------------------------- last-season WS/48 -------
def load_ws48(y):
    """Season y's BBR table: {norm name: (ws48, mp, games, bpm)} (use y - 1
    for season y's games: no lookahead)."""
    ws = pav.load_bpm(y, stat="WS/48")
    bpm = pav.load_bpm(y)
    return {k: (v[0], v[1], v[2], bpm[k][0] if k in bpm else np.nan)
            for k, v in ws.items()}


def replacement_ws48(table, bpm_rep=pav.REPLACEMENT_BPM):
    """WS/48 at BPM's replacement level: minutes-weighted line of WS/48 on BPM
    (players with REPLACEMENT_MIN_MP+ minutes)."""
    a = np.array([(w, b, mp) for w, mp, _, b in table.values()
                  if mp >= REPLACEMENT_MIN_MP and np.isfinite(b) and np.isfinite(w)])
    if len(a) < 50:
        return 0.030
    slope, icpt = np.polyfit(a[:, 1], a[:, 0], 1, w=np.sqrt(a[:, 2]))
    return float(icpt + slope * bpm_rep)


def prior_values(box, table, rep):
    """player_id -> last-season WS/48 shrunk toward rep by MP / (MP +
    BPM_SHRINK_MP), minus rep (0 for unmatched players)."""
    names = box.drop_duplicates("player_id").set_index("player_id")["name"]
    out = {}
    for pid, nm in names.items():
        rec = table.get(pav.norm_name(nm))
        if rec is None or not np.isfinite(rec[0]):
            out[pid] = 0.0
            continue
        w, mp = rec[0], rec[1]
        out[pid] = (w - rep) * mp / (mp + pav.BPM_SHRINK_MP)
    return out


def dynamic_value(hist, prior, m0=WS_PRIOR_MP, rep=0.0):
    """(player_id, date) -> Bayes WS/48 above replacement: (m0 x prior + MP x
    (obs - rep)) / (m0 + MP), MP and obs over games strictly before date."""
    memo = {}

    def v(pid, date):
        key = (pid, date)
        if key in memo:
            return memo[key]
        p0 = prior.get(pid, 0.0)
        h = hist.get(pid)
        out = p0
        if h is not None:
            k = int(np.searchsorted(h[0], np.datetime64(pd.Timestamp(date)), "right")) - 1
            if k >= 0 and h[2][k] > 0:
                ws, mp = h[1][k], h[2][k]
                out = (m0 * p0 + mp * (48.0 * ws / mp - rep)) / (m0 + mp)
        memo[key] = out
        return out
    return v


def dyn_talent_fn(box, value, role):
    """talent_fn with a date-dependent value(player_id, date)."""
    b = box[box["minutes"] > 0]
    by_team = {tm: g.sort_values("date") for tm, g in b.groupby("team")}

    def f(tm, date):
        g = by_team.get(tm)
        if g is None:
            return float("nan")
        prev = g[g["date"] < pd.Timestamp(date)]
        if not len(prev):
            return float("nan")
        last = prev[prev["date"] == prev["date"].max()]
        return float(sum(role(pid, date) * value(pid, date) for pid in last["player_id"]))
    return f


def season_terms(t, box, role, cache_dir=pav.DEFAULT_CACHE):
    """{column: (team, date) -> talent} for season t: talent_ws (last-season
    WS/48), talent_wsb (Bayes, prior updated in season), talent_wso (this
    season only, shrunk to replacement by MP / (MP + M0)); plus a summary."""
    table = load_ws48(t - 1)
    rep = replacement_ws48(table)
    prior = prior_values(box, table, rep)
    out = {"talent_ws": pav.talent_fn(box, prior, role)}
    info = f"replacement WS/48 {rep:.3f}"
    if os.path.exists(os.path.join(cache_dir, f"box_full_{t}.csv")):
        hist = ws_history(load_full(t, cache_dir))
        rep_t = rep
        out["talent_wsb"] = dyn_talent_fn(box, dynamic_value(hist, prior, rep=rep_t), role)
        out["talent_wso"] = dyn_talent_fn(
            box, dynamic_value(hist, {}, rep=rep_t), role)
        info += f"; season to date for {len(hist)} players"
    return out, info


# ---------------------------------------------------------------- CLI ------
def check(y, cache_dir=pav.DEFAULT_CACHE):
    """Full-season WS from the box lines vs BBR's table for season y."""
    s = season_ws(load_full(y, cache_dir))
    s = s.groupby("player_id").agg(name=("name", "last"), minutes=("minutes", "sum"),
                                   ws=("ws", "sum"), ows=("ows", "sum"),
                                   dws=("dws", "sum"))
    ws = pav.load_bpm(y, stat="WS")
    ows = pav.load_bpm(y, stat="OWS")
    dws = pav.load_bpm(y, stat="DWS")
    s["key"] = s["name"].map(pav.norm_name)
    m = s[s["key"].isin(ws)].copy()
    m["bbr_ws"] = m["key"].map(lambda k: ws[k][0])
    m["bbr_ows"] = m["key"].map(lambda k: ows.get(k, (np.nan,))[0])
    m["bbr_dws"] = m["key"].map(lambda k: dws.get(k, (np.nan,))[0])
    m["bbr_mp"] = m["key"].map(lambda k: ws[k][1])
    m = m[(m["minutes"] - m["bbr_mp"]).abs() < 0.1 * m["bbr_mp"] + 20]
    print(f"season {y}: {len(m)} of {len(s)} players matched (minutes within 10%)")
    for c in ("ws", "ows", "dws"):
        d = m[c] - m["bbr_" + c]
        print(f"  {c.upper():3s} corr {m[c].corr(m['bbr_' + c]):.3f}  mean diff "
              f"{d.mean():+.2f}  MAE {d.abs().mean():.2f}  league total "
              f"{m[c].sum():.0f} vs {m['bbr_' + c].sum():.0f}")
    big = m[m["minutes"] >= 1000]
    w48, b48 = 48 * big["ws"] / big["minutes"], 48 * big["bbr_ws"] / big["bbr_mp"]
    print(f"  WS/48 (1000+ MP, n={len(big)}): corr {w48.corr(b48):.3f}, MAE "
          f"{(w48 - b48).abs().mean():.4f}")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--seasons", nargs="+", default=["2016-2019", "2021-2026"])
    f.add_argument("--workers", type=int, default=8)
    c = sub.add_parser("check")
    c.add_argument("--season", type=int, default=2025)
    for s in (f, c):
        s.add_argument("--cache", default=pav.DEFAULT_CACHE)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        for y in nc.parse_years(a.seasons):
            fetch_full_box(y, a.cache, a.workers)
    else:
        check(a.season, a.cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
