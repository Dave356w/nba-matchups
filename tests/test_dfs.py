"""DFS lineups page (CLAUDE.md, DFS lineup page amendment): Yahoo scoring,
the Yahoo snapshot, Sleeper matching, the exact optimizer and the page."""
import itertools
import json
from collections import Counter
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import build_site
import dfs
import ledger

NOW = datetime(2026, 10, 29, 21, 0, tzinfo=timezone.utc)


def test_yahoo_points_from_a_sleeper_line():
    # the owner's example row: James Harden, 2026-10-22 first snapshot
    harden = dict(pts=17.50, reb=5.21, ast=7.44, stl=1.18, blk=0.44, to=3.22, tpm=2.45)
    assert dfs.yahoo_points(harden) == pytest.approx(
        17.50 + 1.2 * 5.21 + 1.5 * 7.44 + 3 * 1.18 + 3 * 0.44 - 3.22)
    assert dfs.yahoo_points(harden) == pytest.approx(36.552)
    assert np.isnan(dfs.yahoo_points(dict(harden, pts="")))


def test_names_and_teams_match_across_feeds():
    assert dfs.norm_name("Darius Acuff Jr.") == dfs.norm_name("darius acuff")
    assert dfs.norm_name("Nikola Jokić") == "nikola jokic"
    assert dfs.norm_name("Nickeil Alexander-Walker") == "nickeil alexander walker"
    assert [dfs.team_code(t) for t in ("NO", "GS", "NY", "SA", "BKN", "CHA", "ATL")] == \
        ["NOP", "GSW", "NYK", "SAS", "BRK", "CHO", "ATL"]


def _feed(players, cap=200):
    return {"salaryCapInfo": {"result": [{"singleGameSalaryCapMap": {}, "multiGameSalaryCap": cap}]},
            "players": {"result": players}}


def _yp(code, name, team, pos, salary, home="NO", away="SAC",
        start="Thu, 29 Oct 2026 17:00:00 -0700"):
    return {"playerCode": code, "gameCode": "nba.g.1", "position": pos, "team": team,
            "salary": salary, "fppg": 30.0, "name": name, "homeTeam": home,
            "awayTeam": away, "gameStartTime": start}


def test_feed_snapshot_is_replaced_before_start_and_frozen_after():
    rows = dfs.parse_feed(_feed([_yp("p1", "Zion Williamson", "NO", "PF", 36),
                                 _yp("p2", "Bad Pos", "NO", "G", 20)]), now=NOW)
    assert len(rows) == 1 and rows[0]["team"] == "NOP" and rows[0]["home"] == "NOP"
    assert rows[0]["slate_date"] == "2026-10-29" and rows[0]["start_utc"] == "2026-10-30T00:00:00Z"
    assert rows[0]["cap"] == 200
    df, n, k = dfs.upsert(dfs.load("/nonexistent"), rows, now=NOW)
    assert (n, k) == (1, 0)
    later = dfs.parse_feed(_feed([_yp("p1", "Zion Williamson", "NO", "PF", 40)]),
                           now=NOW + timedelta(hours=1))
    df2, n, _ = dfs.upsert(df, later, now=NOW + timedelta(hours=1))
    assert n == 1 and df2["salary"].astype(int).tolist() == [40]
    after = NOW + timedelta(hours=4)          # after the 00:00Z start
    df3, n, k = dfs.upsert(df2, dfs.parse_feed(_feed([_yp("p1", "Zion Williamson", "NO", "PF", 9)]),
                                               now=after), now=after)
    assert (n, k) == (0, 1) and df3["salary"].astype(int).tolist() == [40]


def test_snapshot_never_raises():
    msgs = []
    dfs.snapshot(log=msgs.append, fetch=lambda: (_ for _ in ()).throw(OSError("down")))
    assert msgs and "unavailable" in msgs[0]


def _native():
    return pd.DataFrame([dict(game_id="401", slate_date="2026-10-29", home="NOP", away="SAC",
                              tip_utc="2026-10-30T00:00Z")])


def _sleeper(rows):
    base = {c: "" for c in ledger.SLEEPER_COLUMNS}
    return pd.DataFrame([{**base, "game_id": "401", "tip_utc": "2026-10-30T00:00Z",
                          "snapshot_utc": "2026-10-29T20:00:00Z", **r} for r in rows],
                        columns=ledger.SLEEPER_COLUMNS)


def test_slate_reads_the_latest_sleeper_line_in_yahoo_points():
    yahoo = pd.DataFrame(dfs.parse_feed(_feed([
        _yp("p1", "Zion Williamson", "NO", "PF", 36), _yp("p2", "Darius Acuff Jr.", "SAC", "PG", 25),
        _yp("p3", "Hurt Guy", "SAC", "C", 12), _yp("p4", "Deep Bench", "NO", "SG", 10)]), now=NOW),
        columns=dfs.YAHOO_COLUMNS).astype(str)
    line = dict(minutes="32", pts="20", reb="5", ast="3", stl="1", blk="1", to="2", tpm="1")
    sl = _sleeper([dict(kind="first", team="NOP", sleeper_id="1", name="Zion Williamson", **dict(line, pts="10")),
                   dict(kind="pre", team="NOP", sleeper_id="1", name="Zion Williamson", **line),
                   dict(kind="pre", team="SAC", sleeper_id="2", name="Darius Acuff", **line),
                   dict(kind="pre", team="SAC", sleeper_id="3", name="Hurt Guy", injury_status="Out", **line)])
    d = dfs.slate("2026-10-29", yahoo=yahoo, sleeper=sl, native=_native(), now=NOW)
    by = {p["name"]: p for p in d["players"]}
    want = 20 + 1.2 * 5 + 1.5 * 3 + 3 + 3 - 2
    assert by["Zion Williamson"]["proj"] == pytest.approx(want)   # the "pre" line, not "first"
    assert by["Darius Acuff Jr."]["proj"] == pytest.approx(want)  # suffix dropped to match
    assert by["Hurt Guy"]["proj"] == 0 and by["Hurt Guy"]["inj"] == "Out"
    assert by["Deep Bench"]["proj"] is None and d["unmatched"] == ["Deep Bench"]
    assert d["cap"] == 200 and d["min_teams"] == 3 and d["max_per_team"] == 6
    assert by["Zion Williamson"]["opp"] == "SAC" and by["Zion Williamson"]["home"]
    assert dfs.slate("2026-10-30", yahoo=yahoo, sleeper=sl, native=_native()) is None


# --------------------------------------------------- the exact optimizer ---
NODE = shutil.which("node")


def _legal(lineup, cap, min_sal):
    pos = [p["pos"] for p in lineup]
    g, f, c = (sum(x in s for x in pos) for s in (("PG", "SG"), ("SF", "PF"), ("C",)))
    teams = Counter(p["team"] for p in lineup)
    sal = sum(p["sal"] for p in lineup)
    return (all(x in pos for x in ("PG", "SG", "SF", "PF", "C")) and g >= 3 and f >= 3 and c >= 1
            and min_sal <= sal <= cap and len(teams) >= 3 and max(teams.values()) <= 6)


def _brute(pool, cap, min_sal, lock=(), out=()):
    pool = [p for p in pool if p["id"] not in out]
    best = []
    for combo in itertools.combinations(pool, 8):
        ids = {p["id"] for p in combo}
        if set(lock) <= ids and _legal(combo, cap, min_sal):
            best.append(round(sum(p["v"] for p in combo), 6))
    return sorted(best, reverse=True)


def _node(pools):
    script = dfs.SOLVER_JS + """
var cases=JSON.parse(require('fs').readFileSync(0,'utf8'));
var out=cases.map(function(c){var team={};c.pool.forEach(function(p){team[p.id]=p.team;});
 var lock={};c.lock.forEach(function(i){lock[i]=1;});
 var P=c.pool.filter(function(p){return c.out.indexOf(p.id)<0;});
 var r=DFS.kbest(P,c.cap,c.min,lock,c.n,function(s){return DFS.teamsOk(s.ids,team,3,6);},5000);
 return r.lineups.map(function(l){return {v:l.v,sal:l.sal,ids:l.ids};});});
process.stdout.write(JSON.stringify(out));"""
    res = subprocess.run([NODE, "-e", script], input=json.dumps(pools), capture_output=True,
                         text=True, timeout=120, check=True)
    return json.loads(res.stdout)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_optimizer_matches_brute_force_best_first():
    rng = np.random.default_rng(11)
    cases = []
    for k in range(6):
        pool = [dict(id=f"p{i}", pos=pos, team=f"T{rng.integers(0, 3 if k == 5 else 6)}",
                     sal=int(rng.integers(10, 46)), v=round(float(rng.uniform(5, 50)), 2))
                for i, pos in enumerate(["PG", "SG", "SF", "PF", "C"] * 3 + ["PG", "SF", "C"])]
        lock = [] if k < 3 else ["p0"]
        out = [] if k < 4 else ["p1", "p2"]
        cases.append(dict(pool=pool, cap=200, min=0 if k % 2 == 0 else 185, lock=lock,
                          out=out, n=8))
    got = _node(cases)
    for c, lineups in zip(cases, got):
        want = _brute(c["pool"], c["cap"], c["min"], c["lock"], c["out"])[:c["n"]]
        assert [round(l["v"], 6) for l in lineups] == want
        byid = {p["id"]: p for p in c["pool"]}
        for l in lineups:
            team = [byid[i] for i in l["ids"]]
            assert len(set(l["ids"])) == 8 and _legal(team, c["cap"], c["min"])
            assert set(c["lock"]) <= set(l["ids"]) and not set(c["out"]) & set(l["ids"])
            assert l["sal"] == sum(p["sal"] for p in team)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_optimizer_returns_nothing_when_no_lineup_is_legal():
    pool = [dict(id=f"p{i}", pos=pos, team=f"T{i % 6}", sal=40, v=10.0)
            for i, pos in enumerate(["PG", "SG", "SF", "PF", "C"] * 2)]
    assert _node([dict(pool=pool, cap=200, min=0, lock=[], out=[], n=3)]) == [[]]


# ---------------------------------------------------------------- the page ---
def test_page_embeds_the_slate_and_says_whose_projections(tmp_path):
    yahoo = pd.DataFrame(dfs.parse_feed(_feed([_yp("p1", "Zion Williamson", "NO", "PF", 36)]),
                                        now=NOW), columns=dfs.YAHOO_COLUMNS).astype(str)
    line = dict(minutes="32", pts="20", reb="5", ast="3", stl="1", blk="1", to="2", tpm="1")
    sl = _sleeper([dict(kind="pre", team="NOP", sleeper_id="1", name="Zion Williamson", **line)])
    html = dfs.page(build_site, "2026-10-29", "built", native=_native(), yahoo=yahoo, sleeper=sl)
    assert "<h1>DFS lineups</h1>" in html and "Sleeper" in html and "not this site’s model" in html
    data = json.loads(html.split("id='dfs-data'>")[1].split("</script>")[0])
    assert data["players"][0]["name"] == "Zion Williamson" and data["cap"] == 200
    assert "aria-current='page'>DFS<" in html
    empty = dfs.page(build_site, "2026-10-30", "built", native=_native(), yahoo=yahoo, sleeper=sl)
    assert "No Yahoo slate today" in empty and "dfs-data" not in empty


def test_dfs_is_a_published_page_and_reads_no_record():
    assert "dfs.html" in build_site.PUBLIC_FILES
    src = open(dfs.__file__).read()
    assert "import analysis" not in src and "p_home" not in src


def test_validation_checks_the_yahoo_snapshot(tmp_path):
    import validate_data_files as v
    rows = dfs.parse_feed(_feed([_yp("p1", "Zion Williamson", "NO", "PF", 36)]), now=NOW)
    path = tmp_path / "y.csv"
    dfs.save(pd.DataFrame(rows, columns=dfs.YAHOO_COLUMNS), path)
    assert v.check_yahoo(str(path)) == []
    bad = pd.read_csv(path, dtype=str).assign(salary="0")
    bad.to_csv(path, index=False)
    assert any("salary" in e for e in v.check_yahoo(str(path)))
    assert v.check_yahoo(str(tmp_path / "missing.csv")) == []


# ------------------------------------------------------- tournament mode ---
def _node_js(body, payload):
    script = dfs.SOLVER_JS + "\nvar I=JSON.parse(require('fs').readFileSync(0,'utf8'));\n" + body
    res = subprocess.run([NODE, "-e", script], input=json.dumps(payload), capture_output=True,
                         text=True, timeout=180, check=True)
    return json.loads(res.stdout)


def _sim_pool(seed=5):
    rng = np.random.default_rng(seed)
    pool = []
    for g, teams in enumerate((("A", "B"), ("C", "D"))):
        for t in teams:
            for k, pos in enumerate(["PG", "SG", "SF", "PF", "C", "PG", "SF", "C"]):
                pool.append(dict(id=f"{t}{k}", pos=pos, sal=int(rng.integers(10, 40)),
                                 v=round(float(rng.uniform(15, 45)), 2), team=t, game=f"g{g}",
                                 starter=k < 5, star=(k + 1) if k < 2 else 0,
                                 # the starting centre (k 4) and the bench one (k 7)
                                 big={4: 1, 7: 2}.get(k, 0),
                                 big_split=dfs.SIM["team"] - dfs.SIM["center_pair"] if k in (4, 7) else 0))
    return pool


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_tournament_without_variance_is_the_cash_best_and_is_seeded():
    pool = _sim_pool()
    flat = dict(dfs.SIM, sd_base=0.0, sd_slope=0.0)
    out = _node_js("""
var team={};I.pool.forEach(function(p){team[p.id]=p.team;});var ok=function(ids){return DFS.teamsOk(ids,team,3,6);};
var best=DFS.kbest(I.pool,200,0,{},1,function(s){return ok(s.ids);},200).lineups[0];
var flat=DFS.simulate(I.pool,200,0,{},I.flat,50,7,ok),a=DFS.simulate(I.pool,200,0,{},I.sim,200,7,ok),b=DFS.simulate(I.pool,200,0,{},I.sim,200,7,ok);
process.stdout.write(JSON.stringify({best:best.ids.slice().sort(),flat:flat.cands.map(function(c){return [c.ids.slice().sort(),c.n];}),
 a:a.cands.map(function(c){return c.n;}),b:b.cands.map(function(c){return c.n;}),na:a.cands.length,
 legal:a.cands.every(function(c){return ok(c.ids)&&c.sal<=200&&new Set(c.ids).size===8;})}));""",
        dict(pool=pool, flat=flat, sim=dfs.SIM))
    assert out["flat"] == [[out["best"], 50]]            # every simulated slate: the cash best
    assert out["a"] == out["b"] and out["na"] > 5         # seeded, and it diversifies
    assert sum(out["a"]) == 200 and out["legal"]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_simulated_points_carry_the_measured_correlations():
    pool = [dict(p, v=200.0) for p in _sim_pool()]       # far from the floor at 0
    zs = _node_js("""
var r=DFS.rng(11),Z=[];for(var s=0;s<20000;s++){var x=DFS.draw(I.pool,I.sim,r);
 Z.push(Array.from(x).map(function(v,i){var p=I.pool[i];return (v-p.v)/(I.sim.sd_base+I.sim.sd_slope*p.v);}));}
process.stdout.write(JSON.stringify(Z));""", dict(pool=pool, sim=dfs.SIM))
    z = np.array(zs)
    c = np.corrcoef(z.T)
    ix = {p["id"]: i for i, p in enumerate(pool)}
    m = dfs.SIM
    pairs = {("A2", "A3"): m["team"] + m["starter"],               # two starters
             ("A5", "A6"): m["team"],                              # two bench players
             ("A0", "A1"): m["team"] + m["starter"] + m["game_star"] - m["star_split"],  # own stars
             ("A0", "B0"): m["game_star"], ("A0", "B1"): m["game_star"],             # opposing stars
             ("A4", "A7"): m["center_pair"],                       # the team's two centres
             ("A2", "B2"): 0.0, ("A0", "C0"): 0.0}                 # opponents; other game
    for (a, b), want in pairs.items():
        assert c[ix[a], ix[b]] == pytest.approx(want, abs=0.02), (a, b)
    assert np.allclose(z.std(0), 1, atol=0.03)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_sim_model_and_flags_reach_the_page():
    yahoo = pd.DataFrame(dfs.parse_feed(_feed([_yp("p1", "Zion Williamson", "NO", "PF", 36),
                                               _yp("p2", "Other Guy", "NO", "C", 20),
                                               _yp("p3", "Backup Big", "NO", "C", 10)]), now=NOW),
                         columns=dfs.YAHOO_COLUMNS).astype(str)
    line = dict(minutes="32", pts="20", reb="5", ast="3", stl="1", blk="1", to="2", tpm="1")
    sl = _sleeper([dict(kind="pre", team="NOP", sleeper_id="1", name="Zion Williamson", **line),
                   dict(kind="pre", team="NOP", sleeper_id="2", name="Other Guy", **dict(line, pts="5")),
                   dict(kind="pre", team="NOP", sleeper_id="3", name="Backup Big", **dict(line, pts="2"))])
    d = dfs.slate("2026-10-29", yahoo=yahoo, sleeper=sl, native=_native(), now=NOW)
    by = {p["name"]: p for p in d["players"]}
    assert d["sim"] == dfs.SIM and by["Zion Williamson"]["star"] == 1 and by["Other Guy"]["star"] == 2
    assert by["Zion Williamson"]["starter"] and by["Other Guy"]["starter"]
    # the two centres (both starters here): split to the measured pair
    m = dfs.SIM
    assert (by["Other Guy"]["big"], by["Backup Big"]["big"]) == (1, 2) and "big" not in by["Zion Williamson"]
    assert by["Other Guy"]["big_split"] == by["Backup Big"]["big_split"] == pytest.approx(
        m["team"] + m["starter"] - m["center_pair"], abs=1e-4)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_tournament_rank_puts_the_lineups_that_often_score_highest_first():
    pool = _sim_pool()
    flat = dict(dfs.SIM, sd_base=0.0, sd_slope=0.0)
    out = _node_js("""
var team={};I.pool.forEach(function(p){team[p.id]=p.team;});var ok=function(ids){return DFS.teamsOk(ids,team,3,6);};
var top=DFS.kbest(I.pool,200,0,{},20,function(s){return ok(s.ids);},400).lineups;
var flat=DFS.rank(top.map(function(c){return {ids:c.ids,v:c.v,sal:c.sal};}),DFS.simulate(I.pool,200,0,{},I.flat,40,3,ok),.05);
var sim=DFS.rank(top.map(function(c){return {ids:c.ids,v:c.v,sal:c.sal};}),DFS.simulate(I.pool,200,0,{},I.sim,300,3,ok),.25);
process.stdout.write(JSON.stringify({best:top[0].ids.slice().sort(),flat:flat.map(function(c){return [c.ids.slice().sort(),c.top];}),
 sim:sim.map(function(c){return [c.top,c.p90,c.v];})}));""", dict(pool=pool, flat=flat, sim=dfs.SIM))
    assert out["flat"][0] == [out["best"], 1.0]          # no variance: the best always tops
    assert all(t == 0.0 for _, t in out["flat"][1:])
    tops = [t for t, _, _ in out["sim"]]
    assert tops == sorted(tops, reverse=True) and 0 < tops[0] <= 1
    assert all(p90 >= v for _, p90, v in out["sim"])      # the 90th percentile is above the mean
