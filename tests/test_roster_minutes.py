import importlib.util
import os

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "roster_minutes",
    os.path.join(os.path.dirname(__file__), "..", "research", "roster_minutes.py"))
rm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rm)


def test_allocate_fills_240_and_respects_the_cap():
    m = rm.allocate([30, 30, 30, 30, 30, 30, 30, 30])
    assert np.allclose(m, 30.0)
    m = rm.allocate([48, 10, 10, 10, 10, 10, 10, 10, 0])
    assert abs(m.sum() - 240) < 1e-9 and abs(m[0] - rm.CAP) < 1e-9 and m[-1] == 0
    assert np.allclose(m[1:-1], (240 - rm.CAP) / 7)
    assert np.allclose(rm.allocate([30, 30, 30]), rm.CAP)      # too few to fill 240
    assert rm.allocate([0, 0]).sum() == 0


def team_box(n_games=4, debut=None):
    """Team T: players a..f each play 40 min in every game (6 x 40 = 240);
    `debut` joins at the last game (listed, 0 minutes before)."""
    rows = []
    for k in range(n_games):
        for p, mins in (("a", 40), ("b", 40), ("c", 40), ("d", 40), ("e", 40), ("f", 40)):
            rows.append(dict(game_id=f"g{k}", date=pd.Timestamp("2026-01-01")
                             + pd.Timedelta(days=k), team="T", player_id=p,
                             minutes=float(mins)))
        if debut and k == n_games - 1:
            rows.append(dict(game_id=f"g{k}", date=pd.Timestamp("2026-01-01")
                             + pd.Timedelta(days=k), team="T", player_id=debut,
                             minutes=30.0))
    return pd.DataFrame(rows)


VALUE = {"a": 6.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0, "f": 0.0, "z": 3.0}


def test_out_players_minutes_go_to_teammates_and_value_is_lost():
    tg = team_box()
    base = rm.team_lineups(tg, VALUE, lambda p, d: 0.0, {})["g3"]
    assert abs(base["list_od"] - 40 / 48 * 6) < 1e-9          # a's 40 of 240
    out = rm.team_lineups(tg, VALUE, lambda p, d: 0.0,
                          {"od": {("g3", "a"): 0.0}})["g3"]
    assert out["list_od"] == 0.0 and out["prev_od"] == 0.0     # a out, 240 to others
    assert abs(out["hind"] - base["hind"]) < 1e-9              # hind ignores the report


def test_game_k_minutes_reach_only_the_hindsight_variant():
    tg = team_box()
    a = rm.team_lineups(tg, VALUE, lambda p, d: 0.0, {})["g3"]
    tg2 = tg.copy()
    tg2.loc[(tg2.game_id == "g3") & (tg2.player_id == "a"), "minutes"] = 0.0
    b = rm.team_lineups(tg2, VALUE, lambda p, d: 0.0, {})["g3"]
    for var in ("prev_od", "list_od", "list_q"):
        assert a[var] == b[var]
    assert b["hind"] == 0.0 != a["hind"]


def test_trade_debut_counts_on_the_listed_roster_only():
    tg = team_box(debut="z")
    r = rm.team_lineups(tg, VALUE, lambda p, d: 30 / 48, {})["g3"]
    # listed: z joins with 30 raw minutes (arrival role); 7 players share 240
    want = 240 * 40 / (6 * 40 + 30) / 48 * 6 + 240 * 30 / (6 * 40 + 30) / 48 * 3
    assert abs(r["list_od"] - want) < 1e-9
    assert abs(r["prev_od"] - 40 / 48 * 6) < 1e-9             # prev box: no z


def test_listing_check_reports_share_listed():
    box = pd.DataFrame(dict(game_id=["g1", "g1"], player_id=["a", "b"]))
    st = pd.DataFrame(dict(game_id=["g1", "g1", "g1"], player_id=["a", "b", "c"],
                           status=["Out", "Out", "Questionable"]))
    chk = rm.listing_check(st, box)
    assert chk["Out"] == (1.0, 2) and chk["Questionable"] == (0.0, 1)


def synthetic_season(seed):
    import nba_composite as nc
    import player_availability as pav
    from test_team_quality import league
    logs = league(seed=seed, rounds=30)
    rng = np.random.default_rng(seed)
    rows = []
    for tm, lg in logs.items():
        for r in lg.itertuples(index=False):
            h, a = (tm, r.opp) if r.home else (r.opp, tm)
            gid = f"{r.date:%Y%m%d}{h}{a}"
            for k in range(8):
                mins = 0.0 if rng.random() < 0.1 else float(rng.uniform(15, 38))
                rows.append(dict(game_id=gid, date=r.date, team=tm, opp=r.opp,
                                 home=bool(r.home), margin=float(r.pts - r.opp_pts),
                                 player_id=f"{tm}p{k}", name=f"{tm} p{k}",
                                 minutes=mins, pm=0.0))
    box = pd.DataFrame(rows)
    value = {p: float(rng.normal(2, 2)) for p in box["player_id"].unique()}
    st = pd.DataFrame(columns=["game_id", "team", "player_id", "status", "played",
                               "report"])
    av = pav.game_availability(box, value)
    S = dict(box=box, value=value, role=lambda p, d: 0.5, st=st,
             covered=set(av["game_id"]), av=av,
             talent=pav.talent_fn(box, value, lambda p, d: 0.5))
    return logs, S


def test_season_frame_and_report_run_end_to_end(monkeypatch):
    import nba_composite as nc
    from test_team_quality import WEIGHTS
    logs, S = synthetic_season(4)
    monkeypatch.setattr(nc, "load_logs", lambda y, refresh=False: logs)
    f = rm.common(rm.season_frame(2026, S, WEIGHTS, {}))
    assert len(f) > 30 and (f[["gp_home", "gp_away"]] >= nc.MIN_GAMES).all().all()
    for arm, feats in rm.ARMS.items():
        fm = nc.fit_logit(f[feats].to_numpy(float), f["win"], feats)
        f[f"p_{arm}"] = nc.predict(fm, f[feats].to_numpy(float))
    f["year"], f["close_book"], f["home_won"] = 2026, "dk", f["win"]
    f["close_q_home"] = 0.55
    txt = rm.report(f)
    assert f"DraftKings close · games 10+, report-covered (n={len(f)})" in txt
    for arm in rm.ARMS:
        assert f"  {arm:10s} logloss" in txt


def test_pos_number_reads_bbr_positions():
    assert rm.pos_number("C") == 5 and rm.pos_number("PF-C") == 4.5
    assert rm.pos_number("SG-PG") == 1.5 and np.isnan(rm.pos_number(""))


def test_parse_positions_keeps_the_most_minutes_row():
    html = ('<table id="advanced"><thead><tr><th>Player</th><th>Pos</th><th>MP</th>'
            '</tr></thead><tbody><tr><td>Nikola Jokić</td><td>C</td><td>2700</td></tr>'
            '<tr><td>Dual Guy</td><td>SG</td><td>300</td></tr>'
            '<tr><td>Dual Guy</td><td>SF</td><td>900</td></tr></tbody></table>')
    assert rm.parse_positions(html) == {"nikola jokic": 5.0, "dual guy": 3.0}


def test_centre_minutes_go_to_bigs_point_guard_minutes_to_guards():
    #               PG  SG  SF  PF  C   bigC bigPG
    pos = np.array([1, 2, 3, 4, 5, 5, 1], float)
    full = np.array([36, 34, 34, 32, 34, 34, 36], float)    # 240
    out_c = np.array([1, 1, 1, 1, 0, 1, 1], float)
    m = rm.positional_fill(full * out_c, full * (1 - out_c), pos, cap=240)
    assert abs(m.sum() - 240) < 1e-9
    gain = m - full * out_c
    assert gain[5] > gain[3] > 0                           # C -> C, then PF
    assert gain[0] == gain[1] == gain[2] == gain[6] == 0   # guards, SF: nothing
    prop = rm.allocate(full * out_c, cap=240)
    assert m[5] > prop[5] and m[6] < prop[6]
    capped = rm.positional_fill(full * out_c, full * (1 - out_c), pos)
    assert capped.max() <= rm.CAP + 1e-9 and abs(capped.sum() - 240) < 1e-9
    out_pg = np.array([0, 1, 1, 1, 1, 1, 1], float)
    g = rm.positional_fill(full * out_pg, full * (1 - out_pg), pos,
                           cap=240) - full * out_pg
    assert g[6] > g[1] > 0 and g[2] == g[3] == g[4] == g[5] == 0


def test_positional_fill_falls_back_without_positions_or_neighbours():
    e = np.array([40, 40, 40, 40, 40, 0], float)          # 6th (40 min) is out
    nan = np.full(6, np.nan)
    miss = np.array([0, 0, 0, 0, 0, 40], float)
    assert np.allclose(rm.positional_fill(e, miss, nan), rm.allocate(e))
    far = np.array([1, 1, 1, 1, 1, 5], float)                # nobody near a C
    assert np.allclose(rm.positional_fill(e, miss, far), rm.allocate(e))


def test_team_lineups_position_variant_moves_value_to_the_backup_big():
    rows = []
    roster = [("pg", 34, 1), ("sg", 32, 2), ("sf", 32, 3), ("pf", 30, 4),
              ("c", 34, 5), ("bc", 14, 5), ("bg", 14, 1)]
    for k in range(4):
        for p, mins, _ in roster:
            rows.append(dict(game_id=f"g{k}", date=pd.Timestamp("2026-01-01")
                             + pd.Timedelta(days=k), team="T", player_id=p,
                             minutes=float(mins)))
    tg = pd.DataFrame(rows)
    pos = {p: float(x) for p, _, x in roster}
    value = {"bc": 4.0}                                      # only the backup C counts
    r = rm.team_lineups(tg, value, lambda p, d: 0.0, {"od": {("g3", "c"): 0.0}},
                        pos=pos)["g3"]
    assert r["list_od_pos"] > r["list_od"] > 0
