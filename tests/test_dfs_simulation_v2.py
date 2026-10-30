"""DFS audit regressions: marginal moments, fresh evaluation, legal search,
final-size exposure caps and chronological native-only calibration."""
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import dfs
from research import dfs_calibration as cal

NODE = shutil.which("node")


def node(body, payload=None):
    if NODE is None:
        pytest.skip("node is not installed")
    script = dfs.SOLVER_JS + "\nvar I=JSON.parse(require('fs').readFileSync(0,'utf8'));\n"+body
    result = subprocess.run([NODE, "-e", script], input=json.dumps(payload or {}),
                            text=True, capture_output=True, timeout=120, check=True)
    return json.loads(result.stdout)


def test_low_minutes_and_zero_projection_preserve_their_means_and_spread():
    pool = [dict(id=str(i), team=str(i), game=str(i), v=v, min=m)
            for i, (v, m) in enumerate([(0, 0), (0, 30), (1, 3), (3, 3), (5, 8), (20, 25), (40, 35)])]
    out = node("""
var n=200000,r=DFS.rng(321),sum=I.pool.map(()=>0),ss=I.pool.map(()=>0);
for(var s=0;s<n;s++){var x=DFS.draw(I.pool,I.sim,r);for(var i=0;i<x.length;i++){sum[i]+=x[i];ss[i]+=x[i]*x[i];}}
process.stdout.write(JSON.stringify(I.pool.map((p,i)=>({mean:sum[i]/n,sd:Math.sqrt(Math.max(0,ss[i]/n-(sum[i]/n)**2)),want:DFS.simSd(p,I.sim)}))));
""", dict(pool=pool, sim=dfs.SIM))
    for p, got in zip(pool, out):
        assert got["mean"] == pytest.approx(p["v"], abs=max(.025, p["v"]*.005))
        if p["v"]:
            assert got["sd"] == pytest.approx(got["want"], rel=.03)
        else:
            assert got["sd"] == 0
    assert out[3]["want"] < 3   # a 3-minute player no longer gets an 8-point SD


def test_independent_evaluation_is_repeatable_and_distinct_from_generation():
    out = node("""
var P=[{id:'p',team:'T',game:'G',v:30,min:30}],a=DFS.evaluations(P,I.sim,100,356),
 b=DFS.evaluations(P,I.sim,100,356),c=DFS.evaluations(P,I.sim,100,(356^0x9E3779B9)>>>0);
process.stdout.write(JSON.stringify({a:Array.from(a.scores(['p'])),b:Array.from(b.scores(['p'])),c:Array.from(c.scores(['p']))}));
""", dict(sim=dfs.SIM))
    assert out["a"] == out["b"]
    assert out["a"] != out["c"]
    # The production UI freezes candidates before fresh evaluation, then
    # ranks only the evaluation sample (not generation's winning scores).
    assert dfs.UI_JS.index("generation.cands.concat") < dfs.UI_JS.index("DFS.evaluations")
    assert "DFS.rank(cands,res,.05)" in dfs.UI_JS
    assert "Median evaluation example" in dfs.UI_JS


def test_team_constrained_search_finds_legal_solution_beyond_old_200_limit():
    out = node("""
var P=[];['PG','SG','SF','PF','C'].forEach(pos=>{for(var j=0;j<3;j++){var i=P.length;P.push({id:'x'+i,pos:pos,sal:10,v:100+i/100,team:i%2?'B':'A'});}});
P.push({id:'z',pos:'PG',sal:10,v:1,team:'C'});var team=Object.fromEntries(P.map(p=>[p.id,p.team])),ok=s=>DFS.teamsOk(s.ids,team,3,6);
var old=DFS.kbest(P,200,0,{},1,ok,200),fixed=DFS.kbest(P,200,0,{},1,ok,200,{minTeams:3,maxPer:6});
process.stdout.write(JSON.stringify({old:old,fixed:fixed,legal:fixed.lineups.every(ok)}));
""")
    assert out["old"]["lineups"] == [] and out["old"]["truncated"]
    assert len(out["fixed"]["lineups"]) == 1 and not out["fixed"]["truncated"]
    assert "z" in out["fixed"]["lineups"][0]["ids"] and out["legal"]


def test_capped_search_is_not_reported_as_infeasibility():
    assert "if(search.truncated)throw new Error" in dfs.SOLVER_JS
    assert "if(r.truncated)throw new Error" in dfs.UI_JS


def test_partial_portfolio_cannot_inflate_exposure_and_small_caps_allow_zero():
    out = node("""
var P=[{id:'p'}],share={p:.7};
var one=DFS.portfolio(P,1,{p:1},{},(n,cap)=>({L:[{ids:['p']}],cnt:{p:1}}));
// make() normally respects caps; exercise the production greedy selection.
function make(n,cap){return {L:cap.p?Array.from({length:Math.min(1,cap.p)},()=>({ids:['p']})):[],cnt:{p:cap.p?1:0}};}
var strict=DFS.portfolio(P,10,share,{},make),zero=DFS.portfolio(P,1,{p:.7},{},make);
process.stdout.write(JSON.stringify({strict:strict,zero:zero,one:one}));
""")
    assert out["strict"]["L"] == [] and out["zero"]["L"] == []
    assert len(out["one"]["L"]) == 1


def test_portfolio_revalidates_counts_even_if_builder_returns_bad_caps():
    out = node("""
var bad=DFS.portfolio([{id:'p'}],10,{p:.7},{},(n,cap)=>({L:[{ids:['p']}],cnt:{p:1}}));
process.stdout.write(JSON.stringify(bad));
""")
    assert out["L"] == []


def test_final_box_includes_explicit_dnps_and_rejects_unfinished_results():
    box = {"header": {"competitions": [{"status": {"type": {"completed": True}}}]},
           "boxscore": {"players": [{"team": {"abbreviation": "BOS"}, "statistics": [{
               "keys": list(cal.SCORE_KEYS.values()), "athletes": [
                   {"athlete": {"displayName": "A"}, "stats": [10, 2, 3, 1, 1, 2]},
                   {"athlete": {"displayName": "DNP"}, "stats": [], "didNotPlay": True},
                   {"athlete": {"displayName": "Missing"}, "stats": []}]}]}]}}
    lines = cal.final_lines(box)
    assert [r["actual"] for r in lines] == pytest.approx([20.9, 0])
    box["header"]["competitions"][0]["status"]["type"]["completed"] = False
    assert cal.final_lines(box) == []


def native_frames():
    native = pd.DataFrame([dict(game_id="1", basis="native", home_won=1,
                                slate_date="2026-10-22", tip_utc="2026-10-23T00:00Z"),
                           dict(game_id="2", basis="rebuilt", home_won=1,
                                slate_date="2026-10-22", tip_utc="2026-10-23T00:00Z")])
    base = dict(game_id="1", team="BOS", sleeper_id="a", name="A", minutes="30",
                pts="20", reb="5", ast="2", stl="1", blk="1", to="2", injury_status="")
    sleeper = pd.DataFrame([{**base, "kind": "first", "snapshot_utc": "2026-10-22T12:00Z"},
                            {**base, "kind": "pre", "pts": "25", "snapshot_utc": "2026-10-22T23:00Z"},
                            {**base, "kind": "pre", "pts": "99", "snapshot_utc": "2026-10-23T01:00Z"},
                            {**base, "game_id": "2", "kind": "pre", "snapshot_utc": "2026-10-22T23:00Z"}])
    return sleeper, native


def test_only_native_before_tip_snapshots_enter_calibration():
    sleeper, native = native_frames()
    got = cal.snapshots(sleeper, native, "2026-10-23")
    assert got["game_id"].tolist() == ["1"]
    assert got["projection"].tolist() == [38]
    assert cal.snapshots(sleeper, native, "2026-10-22").empty


def test_temporal_holdout_does_not_affect_fitted_parameters():
    rng = np.random.default_rng(77)
    rows = []
    for date in pd.date_range("2026-10-20", periods=8):
        for i in range(120):
            rows.append(dict(date=str(date.date()), bin="high:starter", base_var=100.,
                             projection=30., actual=30+rng.normal(0, 10), game_id=f"{date}-{i//20}",
                             team=str(i//10), starter=True, star=i%10<2))
    f = pd.DataFrame(rows)
    a = cal.calibrate(f, "2026-10-28")
    f.loc[f.date >= "2026-10-26", "actual"] += 100
    b = cal.calibrate(f, "2026-10-28")
    assert a["bins"] == b["bins"] and a["factors"] == b["factors"]
    assert a["trained_through"] < a["validation_from"] <= a["validated_through"] < "2026-10-28"
    assert not b["accepted"]


def test_future_or_failed_calibration_cannot_activate(tmp_path):
    fit = dict(version=dfs.SIM["version"], accepted=True, trained_through="2026-10-26",
               validation_from="2026-10-27", validated_through="2026-10-29",
               bins={"low:bench": 1.2}, factors={k: dfs.SIM[k] for k in
                    ("team", "starter", "game_star", "star_split", "center_pair")})
    path = tmp_path/"fit.json"
    path.write_text(json.dumps(fit))
    assert dfs.simulation_model("2026-10-29", path) == dfs.SIM
    assert dfs.simulation_model("2026-10-30", path)["bins"] == fit["bins"]
    fit["accepted"] = False; path.write_text(json.dumps(fit))
    assert dfs.simulation_model("2026-10-30", path) == dfs.SIM


def test_page_labels_fixed_candidate_comparison_and_uncertainty():
    assert "candidate top 5%" in dfs.UI_JS
    assert "pp (1 SE)" in dfs.UI_JS
    assert "not a contest finish" in dfs.guide()
    assert "Gaussian factors" in dfs.guide()
