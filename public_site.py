"""Public ML forecasts and the owner-approved mixed v11 record.

Replay is explicitly identified and never replaces a native game.
All selections and grades come from analysis; this module never scores a game.
"""
from __future__ import annotations

import re
from statistics import NormalDist

import numpy as np
import pandas as pd

import analysis
import dfs
import ledger
import live_wp
import market
import wp_calibration
from matchup_context import MatchupContext

PAGES = (("index.html", "Today"), ("results.html", "Results"),
         ("performance.html", "ML stats"), ("how-it-works.html", "How it works"),
         ("dfs.html", "DFS"))
RULES = (("H5·3", "Spread", "spread"), ("H2·F·5", "Value favorite", "favorite"),
         ("OU", "Total (over/under)", "total"))
PREDICTION_STAGES = (("established", "Established · games 10+"),
                     ("early", "Early · games 1–9"),
                     ("opener", "Opening games · game 0"), ("unknown", "Stage unavailable"))


def rule_key(root, early, ou="ou2"):
    """The archived row of a test and stage (CLAUDE.md amendments of
    2026-10-29): the floored v11-family rows (H5·3, H2·F·5) at the
    5 h lock, e.g. H5·3·E·L·v11 (H4 retired 2026-11-10, internal only), and the O/U test's rows of the live totals
    model `ou`: OU·L·ou2 / OU·E·L·ou2 (ou2 amendment, 2026-11-03), OU·L·ou3 /
    OU·E·L·ou3 once ou3 is active (ou3 amendment, 2026-11-06)."""
    if root == "OU":
        return "OU" + ("·E" if early else "") + "·L·" + ou
    return root + ("·E" if early else "") + "·L·v11"

CSS = """
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
.public-head{display:flex;justify-content:space-between;gap:24px;align-items:end;margin:30px 0 24px}
.public-head h1{font-size:clamp(28px,4vw,40px);line-height:1.1;letter-spacing:-.04em;margin:8px 0}
.public-head p{color:var(--muted);max-width:650px;margin:8px 0 0;font-size:15px}
.update{color:var(--muted);font-size:12px;text-align:right;white-space:nowrap}
.update time{display:block;font-family:var(--mono);margin-top:4px;font-size:11px}
.public-intro{display:flex;justify-content:space-between;align-items:center;gap:16px;margin:20px 0}
.public-intro p{margin:0;color:var(--muted);font-size:13px}.public-intro a{font-size:13px;font-weight:650}
.public-note{font-size:13px;color:var(--muted);max-width:780px;margin:18px 0}
.tool-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;align-items:start}
.tool-card{border:1px solid var(--line);border-radius:16px;background:var(--surface);box-shadow:var(--shadow);overflow:hidden}
.tool-card[open]{border-color:rgba(var(--accent),.5)}
.tool-card summary{list-style:none;cursor:pointer;padding:16px}.tool-card summary::-webkit-details-marker{display:none}
.tool-card .g-meta{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;font-size:12px}
.pub-team{display:grid;grid-template-columns:30px minmax(0,1fr) 52px 28px;gap:9px;align-items:center;padding:8px 0}
.pub-team .tchip{width:28px;height:28px}.pub-team .tn{font-weight:700;font-size:15px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pub-team small{display:block;font-size:10px;color:var(--muted)}.pub-team .px{text-align:right;font:600 12px var(--mono)}
.pub-team .sc{text-align:right;font:700 16px var(--mono)}
.team-record{font-weight:400;white-space:nowrap;margin-left:5px}
.factor-table{width:100%;border-collapse:collapse;font-size:11px;font-variant-numeric:tabular-nums}
.factor-table th,.factor-table td{padding:7px 3px;border-bottom:1px solid var(--line-2);text-align:right}
.factor-table th:first-child,.factor-table td:first-child{text-align:left}
.factor-table thead th{font-size:10px;color:var(--muted)}.factor-table .factor-group th{padding-top:12px;color:var(--muted);font-size:10px;text-align:left}
.factor-best{font-weight:750;color:rgb(var(--accent-tx));background:rgba(var(--accent),.07)}
.tool-card .context-note{font-size:11px;line-height:1.5}.inj-team{margin-top:12px}
.inj-team h4{font-size:12px;margin:0 0 4px}.inj-team ul{list-style:none;margin:0;padding:0}
.inj-team li{padding:7px 0;border-bottom:1px solid var(--line-2)}.inj-player{display:flex;justify-content:space-between;align-items:start;gap:8px}
.inj-team small{display:block;color:var(--muted);font-size:11px;margin-top:3px;overflow-wrap:anywhere}.inj-team .badge{white-space:normal;text-align:right}
.forecast-readout{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:14px 0;margin-top:8px;border-top:1px solid var(--line-2)}
.forecast-readout small,.tracked-label{display:block;font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.forecast-readout strong{display:block;font:750 24px/1.35 var(--sans);letter-spacing:-.03em}.forecast-readout span{font-size:12px;color:var(--muted)}
.ou-readout{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;padding:0 0 12px;font-size:12px;color:var(--muted)}
.ou-readout b{color:var(--ink);font-family:var(--mono)}
.tracked-box{border-radius:9px;background:rgba(var(--accent),.07);padding:12px;border:1px solid rgba(var(--accent),.18)}
.tracked-box .tracked-label{color:rgb(var(--accent-tx));margin-bottom:7px}.tracked-line{font-size:12px;display:flex;justify-content:space-between;gap:8px;margin:5px 0}
.tracked-line b{font-family:var(--mono);font-size:12px;white-space:nowrap}.forecast-only{padding:10px 0 0;font-size:12px;color:var(--muted);border-top:1px solid var(--line-2)}
.forecast-only b{color:var(--ink)}
.live-wp{padding:12px 0 4px;margin-top:8px;border-top:1px solid var(--line-2)}[data-live-wp][hidden]{display:none!important}
.live-wp .lwp-head{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:8px}
.live-wp .lwp-head small{font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:rgb(var(--bad))}
.live-wp .lwp-head span{font-size:11px;color:var(--muted);text-align:right}
.live-wp .lwp-row{display:flex;align-items:center;gap:10px;font-size:13px}.live-wp .lwp-row>span{white-space:nowrap}
.live-wp .lwp-row em{font:750 18px var(--sans);font-style:normal;letter-spacing:-.02em}.live-wp .lwp-row b{font-size:12px;color:var(--muted)}
.lwp-bar{flex:1;min-width:40px;height:6px;border-radius:3px;background:rgb(var(--accent));overflow:hidden}
.lwp-bar i{display:block;height:100%;width:50%;background:var(--faint);transition:width .3s}
.details-cue{display:flex;justify-content:space-between;margin-top:14px;padding-top:10px;border-top:1px solid var(--line-2);font-size:11px;color:var(--muted)}
.details-cue i{font-style:normal;transition:transform .15s}.tool-card[open] .details-cue i{transform:rotate(180deg)}
.tool-card .g-detail{padding:0 16px 16px;border-top:1px solid var(--line-2);font-size:13px}.tool-card .g-detail p{color:var(--muted)}
.tool-card .g-detail h3{font-size:12px;margin:16px 0 8px}.public-filters{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0}
.public-filters label{font-size:11px;color:var(--muted);font-weight:650}.public-filters select{display:block;min-height:42px;margin-top:4px;border:1px solid var(--line);border-radius:8px;padding:8px 30px 8px 10px;background:var(--surface);color:var(--ink);font:inherit;font-size:13px}
.seg{flex-wrap:wrap}.seg button{min-height:42px}.rule-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:24px 0}
.rule-panel{border:1px solid var(--line);border-radius:16px;background:var(--surface);padding:20px;min-width:0}
.rule-panel h2{font-size:18px;margin:0 0 6px;letter-spacing:-.02em;text-transform:none}.rule-panel>p{font-size:12px;color:var(--muted);margin:0 0 18px;min-height:38px}
.stage-panel{padding:16px 0;border-top:1px solid var(--line-2)}.stage-panel h3{font-size:12px;margin:0 0 10px;color:var(--muted);text-transform:none;letter-spacing:0}
.stage-panel .record{font:750 28px var(--sans);letter-spacing:-.03em}.stage-panel .sample{font-size:12px;color:var(--muted);margin-left:8px}
.stage-panel .empty-stage{font-size:14px;font-weight:600;margin:6px 0}.stage-panel .empty-help{font-size:12px;color:var(--muted)}
.stage-stats{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;font-size:11px;color:var(--muted)}
.stage-stats b{display:block;font:600 16px var(--mono);color:var(--ink);margin:3px 0}.stage-panel .null{font-size:11px;color:var(--muted);margin:12px 0 0}
.stage-panel details{font-size:12px;margin-top:12px}.stage-panel summary{cursor:pointer;color:rgb(var(--accent-tx));padding:6px 0}
.stage-panel .uchart{margin-top:12px;padding:8px}.stage-panel .uchart .head{font-size:10px}.stage-panel .uchart .v{font-size:12px}
.public-table{width:100%;table-layout:fixed;border-collapse:collapse;font-size:13px;background:var(--surface)}
.public-table th,.public-table td{white-space:normal;overflow-wrap:anywhere}
.public-table th{text-align:left;color:var(--muted);font-size:10px;letter-spacing:.04em;text-transform:uppercase;padding:14px 12px;background:var(--surface-2)}
.public-table td{text-align:left;padding:14px 12px;border-top:1px solid var(--line-2)}.public-table small{display:block;color:var(--muted);font-size:11px;margin-top:3px}
.result-badge,.prediction-badge{display:inline-block;border:1px solid var(--line);border-radius:6px;padding:3px 7px;font-size:11px;font-weight:700}
.model-pick{display:flex;justify-content:space-between;gap:8px;padding:10px 0;border-top:1px solid var(--line-2);font-size:12px}
.model-pick small{display:block;font-size:10px;color:var(--muted);margin-top:3px}.model-pick>b{text-align:right;white-space:nowrap}
.prediction-grid .stage-panel{border:0;padding:0}.prediction-grid .stage-panel .sample{display:block;margin:6px 0 0}
.prediction-grid .rule-panel>p{min-height:0}.prediction-comparison{width:100%;font-size:12px;border-collapse:collapse;margin-top:14px}
.prediction-comparison th,.prediction-comparison td{text-align:right;padding:8px 4px;border-bottom:1px solid var(--line-2)}
.prediction-comparison th:first-child,.prediction-comparison td:first-child{text-align:left}.prediction-comparison th{font-size:10px;color:var(--muted)}
[data-record-panel][hidden],[data-prediction-row][hidden]{display:none!important}
.public-empty{padding:32px;border:1px dashed var(--line);border-radius:14px;margin:24px 0;background:var(--surface);color:var(--muted)}
.public-empty h2{color:var(--ink);font-size:18px;margin:0 0 8px}.public-empty p{font-size:14px;margin:0;max-width:600px}
.guide{max-width:760px}.guide section{margin:28px 0;padding:22px;border:1px solid var(--line);border-radius:14px;background:var(--surface)}
.guide h2{font-size:19px;margin:0 0 12px;letter-spacing:-.02em}.guide h3{font-size:15px;margin:22px 0 6px}.guide p,.guide li{font-size:14px;color:var(--muted);line-height:1.65}
.guide p:last-child{margin-bottom:0}.guide .rule-number{font:600 11px var(--mono);color:rgb(var(--accent-tx));margin-right:8px}
.ml-hero{padding:24px;border:1px solid var(--line);border-radius:16px;background:var(--surface);margin:20px 0}
.ml-hero h2{margin:0 0 8px;font-size:20px}.ml-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:20px;margin:24px 0}
.ml-metrics small{display:block;color:var(--muted);font-size:12px}.ml-metrics strong{display:block;font:700 clamp(24px,3vw,36px) var(--sans);letter-spacing:-.04em;margin-top:6px}
.ml-hero .prediction-comparison{max-width:650px}.ml-hero .public-note{margin-bottom:0}
.roi-matrix-wrap{margin-top:22px;max-width:100%}.roi-matrix{width:100%;table-layout:fixed;border-collapse:separate;border-spacing:3px;font-size:12px;text-align:center}.roi-matrix caption{font-weight:700;text-align:left;margin-bottom:10px;font-size:15px}.roi-matrix th{white-space:normal;text-align:center;font-size:11px;color:var(--muted);padding:8px 3px}.roi-matrix td{white-space:normal;text-align:center;padding:12px 3px;background:var(--surface-2);border:1px solid var(--line-2);border-radius:7px;vertical-align:middle}.roi-matrix strong{display:block;font:700 16px var(--mono);overflow-wrap:anywhere}.roi-matrix small{display:block;color:var(--muted);font-size:10px;margin-top:5px}.roi-matrix .positive{background:rgba(var(--good),.1)}.roi-matrix .negative{background:rgba(var(--bad),.1)}.roi-matrix .positive strong{color:rgb(var(--good))}.roi-matrix .negative strong{color:rgb(var(--bad))}@media(max-width:640px){.roi-matrix{border-spacing:2px}.roi-matrix th{font-size:9px;padding:6px 1px;overflow-wrap:anywhere}.roi-matrix td{padding:9px 1px}.roi-matrix strong{font-size:10px}.roi-matrix small{font-size:8px;overflow-wrap:anywhere}}
.record-label{color:var(--muted);font-size:12px}
.intro-links{display:flex;gap:18px;flex-wrap:wrap}
@media(max-width:640px){.intro-extra{display:none}.game-filters .filter-group>span{display:none}.today-intro.public-intro{align-items:center}.intro-links{gap:12px;flex-wrap:nowrap}.intro-links a{font-size:12px}}
.table-more{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:14px 0 0;font-size:13px;color:var(--muted)}
.table-more button{min-height:42px;padding:8px 16px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink);font:inherit;font-weight:650;cursor:pointer}
.table-more[hidden]{display:none}
@media(max-width:640px){.public-filters{display:grid;grid-template-columns:1fr 1fr;gap:10px}.public-filters select{width:100%;font-size:14px;padding-right:24px}.table-more{flex-direction:column;align-items:stretch;text-align:center}}
.prediction-table td{vertical-align:top}.prediction-table td:first-child{font-weight:650}
details.public-note summary{cursor:pointer;color:rgb(var(--accent-tx));font-weight:650}details.public-note p{margin:10px 0}
.panel-facts{display:flex;flex-wrap:wrap;gap:6px 18px;margin:0 0 6px;padding:0;list-style:none;font-size:13px;color:var(--muted)}.panel-facts b{color:var(--ink);font-weight:650}
.ml-hero details{margin-top:14px;font-size:13px}.ml-hero details>summary{cursor:pointer;color:rgb(var(--accent-tx));font-weight:650;padding:4px 0}
.ml-hero details>p{color:var(--muted);max-width:780px}
.ml-section{margin-top:28px;padding-top:20px;border-top:1px solid var(--line-2)}.ml-section>h3{font-size:15px;margin:0 0 6px}
.forecast-readout.ml-focus{grid-template-columns:1.2fr 1fr}.stage-label{font-size:11px;color:var(--muted)}
[data-source-summary][hidden],[data-ml-panel][hidden],.gcard[hidden]{display:none!important}
.game-filters{display:flex;flex-wrap:wrap;gap:8px 16px;margin:16px 0 12px}.filter-group>span{display:block;font-size:11px;color:var(--muted);margin-bottom:4px}.filter-group .seg{display:flex;margin:0}
.forecast-outcome{display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:12px;margin:0 0 10px}.calibration-main{order:0}.actual-result{order:1}.calibration-evidence{order:2;grid-column:1/-1}.forecast-outcome .calibration-main strong,.forecast-outcome .break-even strong{font:750 22px/1.35 var(--sans);color:var(--ink)}.forecast-outcome .break-even{margin:0 0 8px}.forecast-outcome small{display:block;color:var(--muted);font-size:10px}.forecast-outcome strong{display:block;font-size:14px}.forecast-outcome .calibration-evidence small{font-size:11px}.break-even{font-size:11px;color:var(--muted);margin:0 0 12px}.lock-summary{margin-top:4px}.lock-summary strong{display:block;font-size:13px;text-transform:none;letter-spacing:0}.lock-summary small{display:block;font-size:11px;color:var(--muted);margin-top:5px}
.source-summary{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:16px 0}.source-summary article{padding:16px;background:var(--surface);border:1px solid var(--line);border-radius:12px}.source-summary h3{margin:0 0 6px;font-size:15px}.source-summary p{font-size:12px;color:var(--muted);margin:6px 0}.source-summary strong{color:var(--ink)}
.public-empty button{margin-top:14px;min-height:44px;padding:8px 14px;border:1px solid var(--line);border-radius:8px;background:var(--surface-2);color:var(--ink);font:inherit;cursor:pointer}
@media(max-width:640px){.today-head.public-head{margin:12px 0}.today-head .eyebrow{display:none}.today-head h1{font-size:26px}.today-head.public-head p{font-size:12px;margin-top:4px}.today-head .update{margin-top:8px}.today-intro.public-intro{margin:12px 0}.game-filters{margin:12px 0;gap:8px}.filter-group{width:100%}.filter-group .seg{width:100%}.filter-group .seg button{flex:1;padding:8px 6px}.source-summary{grid-template-columns:1fr}.forecast-readout{padding:10px 0}.tool-card .g-meta{margin-bottom:8px}.lock-summary{padding:10px}}
@media(max-width:640px){.ml-metrics{grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.ml-hero{padding:18px}}
@media(max-width:1000px){.tool-grid,.rule-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:640px){.public-head{display:block;margin:20px 0 18px}.public-head p{font-size:14px}.update{text-align:left;margin-top:12px;white-space:normal}.update time{display:inline;margin-left:6px}
.tool-grid,.rule-grid{grid-template-columns:1fr}.public-intro{align-items:start}.public-intro a{white-space:nowrap}.public-intro p{font-size:12px}
.tool-card summary{padding:16px}.public-table thead{display:none}.public-table,.public-table tbody{display:block;background:none}
.public-table tr{display:grid;grid-template-columns:1fr 1fr;gap:12px;border:1px solid var(--line);border-radius:12px;background:var(--surface);padding:16px;margin-bottom:12px}
.public-table td{display:block;padding:0;border:0;min-width:0}.public-table td:before{content:attr(data-label);display:block;font-size:10px;text-transform:uppercase;color:var(--muted);margin-bottom:5px}
.public-table td:first-child{grid-column:1/-1;font-size:16px;font-weight:700}.public-empty{padding:22px}.guide section{padding:18px}.nav a{min-height:42px;display:grid;place-items:center}}
"""

# Today's filter bar and card order (owner request, 2026-11-07): live games
# first, then upcoming, then final, tip order (data-order) within each; it
# re-sorts whenever the live-score overlay changes a card's state
# (window.nbaCount) and moves nothing when the order is unchanged.
FILTER_JS = """<script>(function(){var bar=document.querySelector('[data-filter]');if(!bar)return;bar.hidden=false;
var grid=document.querySelector('.tool-grid[data-slate]'),rank={live:0,upcoming:1,final:2},stage='established',state='all';
function order(){if(!grid)return;var cs=[].slice.call(grid.querySelectorAll(':scope>.gcard')),
s=cs.slice().sort(function(a,b){return(rank[a.dataset.state]-rank[b.dataset.state])||(a.dataset.order-b.dataset.order);});
if(s.some(function(x,i){return x!==cs[i];}))s.forEach(function(x){grid.appendChild(x);});}
function update(){order();var cards=[].slice.call(document.querySelectorAll('.gcard')),n=0;
cards.forEach(function(x){x.hidden=!(stage==='all'||x.dataset.stage===stage)||!(state==='all'||x.dataset.state===state);if(!x.hidden)n++;});
bar.querySelectorAll('button[data-show]').forEach(function(b){var group=b.dataset.group,k=b.dataset.show;
var count=cards.filter(function(x){return group==='stage'?(k==='all'||x.dataset.stage===k)&&(state==='all'||x.dataset.state===state):(k==='all'||x.dataset.state===k)&&(stage==='all'||x.dataset.stage===stage);}).length;
b.querySelector('.ct').textContent=count;b.setAttribute('aria-pressed',k===(group==='stage'?stage:state)?'true':'false');});
var e=document.querySelector('[data-filter-empty]');if(e){e.hidden=n>0;
var scope=stage==='all'?'':stage==='early'?' in games 1–9':' with both teams at 10+ games';
e.querySelector('p').textContent=state==='live'?'No games are live right now'+scope+'.':state==='upcoming'?'No upcoming games'+scope+' right now.':state==='final'?'No final games'+scope+' yet.':'No games'+scope+' today.';}}
window.nbaCount=update;bar.addEventListener('click',function(e){var b=e.target.closest('button[data-show]');if(b){if(b.dataset.group==='stage')stage=b.dataset.show;else state=b.dataset.show;update();}});
var reset=document.querySelector('[data-filter-reset]');if(reset)reset.addEventListener('click',function(){stage='all';state='all';update();});update();})();</script>"""

# Results shows the matching rows a page at a time (newest first); a filter
# change starts again from the first page. Nothing is recalculated.
PREDICTION_PAGE = 50
PREDICTION_PAGE_MOBILE = 20
PREDICTION_FILTER_JS = """<script>(function(){var controls=document.querySelectorAll('[data-prediction-filter]'),
rows=[].slice.call(document.querySelectorAll('[data-prediction-row]')),more=document.querySelector('[data-predictions-more]'),
step=window.matchMedia&&matchMedia('(max-width:640px)').matches?""" + str(PREDICTION_PAGE_MOBILE) + """:""" + str(PREDICTION_PAGE) + """,limit=step;
function update(){var n=0;rows.forEach(function(r){
var show=true;controls.forEach(function(f){if(f.value!=='all'&&r.dataset[f.dataset.predictionFilter]!==f.value)show=false;});
r.hidden=!(show&&n<limit);if(show)n++;});
var empty=document.querySelector('[data-predictions-empty]');if(empty)empty.hidden=n>0;
if(more){more.hidden=n<=limit;more.querySelector('[data-predictions-count]').textContent='Showing '+Math.min(n,limit)+' of '+n;
more.querySelector('button').textContent='Show '+Math.min(step,Math.max(n-limit,0))+' more';}}
controls.forEach(function(c){c.addEventListener('change',function(){limit=step;update();});});
if(more)more.querySelector('button').addEventListener('click',function(){limit+=step;update();});update();
})();</script>"""

PERFORMANCE_FILTER_JS = """<script>(function(){var cs=document.querySelectorAll('[data-performance-filter]');
function update(){document.querySelectorAll('[data-ml-panel], [data-source-summary]').forEach(function(p){var show=true;cs.forEach(function(c){var key=c.dataset.performanceFilter;if(p.hasAttribute('data-source-summary')&&key==='source')return;if(p.dataset[key]!==c.value)show=false;});p.hidden=!show;});}
cs.forEach(function(c){c.addEventListener('change',update);});update();})();</script>"""


class PublicSite:
    def __init__(self, ui, native, today, built, model_ok=True, context=None, injuries=None, replay=None):
        self.ui, self.today, self.built, self.model_ok = ui, today, built, model_ok
        self.built_utc = pd.Timestamp(ledger.utc_now())
        self.native = native.copy() if native is not None else ledger.empty()
        if len(self.native):
            self.native = self.native[self.native["basis"].eq("native")]
        self.season = ui.season_for(today)
        day = self.native[self.native.slate_date.astype(str).eq(today)]
        self.context = context if context is not None else MatchupContext.saved(day)
        if injuries is None:
            try:
                injuries = ledger.load_injuries()
            except (OSError, ValueError, KeyError):
                injuries = pd.DataFrame(columns=ledger.INJURY_COLUMNS)
        self.injuries = injuries
        # Live win chance (owner request, 2026-11-07): coefficients only; the
        # reader's browser computes it from ESPN's live score (display only)
        self.live_fit = live_wp.load()
        self.predictions = analysis.moneyline_model_predictions(self.native, replay)
        self.calibration = wp_calibration.RollingCalibration(self.native, replay)
        # the O/U rows of the live totals model: ou3's once Fit model has
        # written it (or a native lock carries it), else ou2's
        self.ou = "ou3" if analysis.ou3_active(self.native) else "ou2"
        self.hypotheses = {h["key"]: h for h in analysis.public_hypotheses()
                           + analysis.public_ou_tests(self.native)}
        self.records = []
        # Preserve archival selection/grade parity for the internal tests.
        # Public renders never include these records (ML-only amendment).
        by_id = self.native.set_index("game_id", drop=False)
        for key, name, slug in RULES:
            for early in (True, False):
                hyp = self.hypotheses[rule_key(key, early, self.ou)]
                # the whole ledger: the graders drop ungraded games themselves
                if hyp.get("market") == "ou":
                    signals = analysis.ou_signals(self.native, hyp)
                    grades = analysis.ou_picks(self.native, hyp)
                    at = "lock_ou_snapshot_utc"
                else:
                    src = analysis.on_book(self.native, hyp["native"], analysis.HYPOTHESIS_BOOK)
                    signals = analysis.hypothesis_signals(src, hyp, hyp["native"])
                    grades = analysis.hypothesis_picks(src, hyp, hyp["native"])
                    at = f"{hyp['native']}_snapshot_utc"
                grades = {str(r.game_id): r for _, r in grades.iterrows()} if len(grades) else {}
                for _, s in signals.iterrows():
                    row = by_id.loc[s.game_id]
                    grade = grades.get(str(s.game_id))
                    self.records.append(dict(hyp=hyp, name=name, slug=slug, signal=s,
                        row=row, grade=grade, season=int(row.season), early=hyp["early"],
                        price=hyp["native"], at=row.get(at)))

    def page(self, title, active, body):
        return self.ui.page(title, active, body, self.built, extra_css=CSS)

    def head(self, title, desc, update=False):
        u = self.ui
        return (f"<div class='public-head'><div><div class='eyebrow'>{u.season_txt(self.season)} · NBA forecasts</div>"
                f"<h1>{u.esc(title)}</h1><p>{desc}</p></div>"
                + (f"<div class='update'>Page updated<time>{u.esc(self.built)}</time></div>" if update else "") + "</div>")

    def selection(self, rec):
        u, s = self.ui, rec["signal"]
        if rec["hyp"].get("market") == "ou":
            return f"{u.esc(s.side.title())} {s.line:g} <span class='mono'>{u.ml_txt(s.ml)}</span>"
        line = f" {u.line_txt(s.line)}" if rec["hyp"].get("market") == "ats" else ""
        return f"{u.esc(s.side)}{line} <span class='mono'>{u.ml_txt(s.ml)}</span>"

    def ou_readout(self, row, dk):
        """The game's total: the model's, the book's and the model's lean
        (P(over) against the no-vig over price). Empty without a model total."""
        u = self.ui
        T = pd.to_numeric(row.get("ou_total"), errors="coerce")
        if not np.isfinite(T):
            return ""
        line = pd.to_numeric(row.get("pre_ou_line"), errors="coerce") if dk else np.nan
        p = pd.to_numeric(row.get("ou_p_over"), errors="coerce")
        o, un = (pd.to_numeric(row.get(c), errors="coerce") for c in ("pre_ou_over", "pre_ou_under"))
        q = market.devig(o, un) if dk else np.nan
        if np.isfinite(line) and np.isfinite(p) and np.isfinite(q):
            side, pp = ("Over", p) if p > q else ("Under", 1 - p)
            lean = f"<span>Lean <b>{side} {line:g}</b> · {u.pct(pp, 0)}</span>"
        else:
            lean = "<span>No DraftKings total</span>"
        book = f"DraftKings <b>{line:g}</b>" if np.isfinite(line) else ""
        return (f"<div class='ou-readout'><span>Total · model <b>{T:.1f}</b>"
                + (f" · {book}" if book else "") + f"</span>{lean}</div>")

    def card(self, row, order=None):
        u = self.ui
        number = lambda k: pd.to_numeric(row.get(k), errors="coerce")
        p = number("p_home")
        gp = np.minimum(number("gp_home"), number("gp_away"))
        stage = 'established' if gp >= 10 else 'early' if 1 <= gp <= 9 else 'opener' if gp == 0 else 'unknown'
        stage_text = dict(PREDICTION_STAGES)[stage]
        dk = str(row.get("pre_book")) == "dk"
        final = np.isfinite(number("home_won"))
        status = "Final" if final else "Pregame"
        teams = []
        context = {side: self.context.team(row, side) for side in ('away', 'home')}
        for side in ("away", "home"):
            team = row[side]
            pts = number(f"{side}_pts")
            line = number("pre_spread") * (-1 if side == "away" else 1) if dk else np.nan
            record = context[side]['record']
            record_text = f'{record[0]}–{record[1]}' if record is not None else '—'
            record_title = 'Season W/L before this slate' if record is not None else 'Season record unavailable'
            teams.append(f"<div class='pub-team'><div class='tchip'><span>{u.esc(team)}</span>{u.logo_html(team)}</div>"
                f"<div class='tn'>{u.esc(u.TEAM_NAMES.get(team,team))}<small>{'@ ' if side == 'home' else ''}{u.esc(team)}"
                f"<span class='team-record' title='{record_title}' aria-label='{u.esc(team)} {record_title}: {record_text}'>({record_text})</span></small></div>"
                f"<div class='px'>{u.ml_txt(number(f'pre_{side}_ml') if dk else np.nan)}<small>{u.line_txt(line) if np.isfinite(line) else '—'}</small></div>"
                f"<span class='sc' data-side='{side}'>{int(pts) if np.isfinite(pts) else ''}</span></div>")
        if np.isfinite(p):
            team = row.home if p >= .5 else row.away
            prob = max(p, 1-p)
            margin = number("margin_home")
            if not np.isfinite(margin):
                margin = analysis.ATS_SIGMA * NormalDist().inv_cdf(min(max(p,1e-6),1-1e-6))
            margin_team = row.home if margin >= 0 else row.away
            price = number('pre_home_ml' if p >= .5 else 'pre_away_ml') if dk else np.nan
            breakeven = market.implied(price)
            q_home = number('pre_q_home') if dk else np.nan
            q_pick = (q_home if p >= .5 else 1-q_home) if 0 < q_home < 1 else np.nan
            forecast = (f"<div class='forecast-readout ml-focus'><div><small>Model probability</small><strong>{u.pct(prob)}</strong><span>{u.esc(team)} to win</span></div>"
                        f"<div data-market-probability><small>DraftKings no vig</small><strong>{u.pct(q_pick)}</strong><span>{u.esc(team)}</span></div></div>")
            actual = 'Pending' if not final else 'Win' if (p >= .5) == (number('home_won') == 1) else 'Loss'
            threshold = (f"<small>Price required break even</small><strong>{u.pct(breakeven)}</strong>"
                         + ("" if np.isfinite(breakeven) else "<small>DraftKings price unavailable.</small>"))
            forecast += ("<div class='forecast-outcome'>" + self.calibrated_readout(row, team)
                         + f"<div class='actual-result'><div class='break-even'>{threshold}</div><small>Actual result · {u.esc(team)}</small>"
                         f"<strong data-forecast-result data-pick-side='{'home' if p >= .5 else 'away'}'>{actual}</strong></div></div>")
        else:
            forecast = ("<div class='forecast-only'><b>Forecast unavailable</b><br>A required model input was unavailable.</div>"
                        + "<div class='forecast-outcome'>" + self.calibrated_readout(row, None) + "</div>")
        tracking = ""
        saved = self.predictions[self.predictions.game_id.astype(str).eq(str(row.game_id))] if len(self.predictions) else self.predictions
        if not len(saved):
            tag = str(row.get("lock_model_tag", ""))
            tip = pd.to_datetime(row.get("tip_utc"), utc=True, errors="coerce")
            started = final or (pd.notna(tip) and self.built_utc >= tip)
            why = ("The 5 h lock used an earlier model." if pd.notna(row.get("lock_p_home")) and not tag.endswith(analysis.FAMILY_TAGS) else
                   "No valid five-hour lock was saved before tip." if started else
                   "The ML pick freezes at the first valid snapshot within 5 hours of tip.")
            tracking = f"<div class='forecast-only'><b>Forecast only</b><br>{why}</div>"
        if len(self.predictions):
            saved = self.predictions[self.predictions.game_id.astype(str).eq(str(row.game_id))]
            if len(saved):
                pick = saved.iloc[0]
                result = self.prediction_result(pick)
                priced = analysis.model_prediction_moneyline(saved).iloc[0]
                lock_price = u.ml_txt(priced.ml) if pd.notna(priced.ml) else 'price unavailable'
                outcome = 'awaiting result' if result == 'Pending' else result
                tracking = (f"<div class='tracked-box lock-summary'><strong class='tracked-label'>Locked pick: {u.esc(pick.prediction)} {lock_price} · {outcome}</strong>"
                            f"<small>Saved {u.tip_pt(pick.get('lock_snapshot_utc'))} · {u.pct(pick.prediction_p)} at lock · "
                            f"{'10+ game record' if stage == 'established' else 'Separate stage record'} · 1 unit</small></div>")
        book = "DraftKings" if dk else "DraftKings unavailable; no market"
        q = number("pre_q_home") if dk else np.nan
        live_spread, live = self.live_block(row, dk)
        detail = (f"<h3>Forecast and market</h3><p>{u.esc(row.away)} @ {u.esc(row.home)} · {u.esc(book)} prices. "
                  f"Market home win probability: {u.pct(q)} with the bookmaker margin removed. "
                  f"Forecast saved at {u.tip_pt(row.get('snapshot_utc'))}. Prices and forecasts freeze at tip.</p>")
        if live:
            detail += ("<p>During the game, <b>live win chance</b> updates with the score from the current margin, "
                       "the time left and the DraftKings spread at tip, using a formula fitted on past NBA games. "
                       "It is not the model’s forecast, is not saved and never changes a saved ML pick.</p>")
        if np.isfinite(p):
            detail += f"<p>Predicted margin: {u.esc(margin_team)} by {abs(margin):.1f} points. This is forecast context.</p>"
        if len(saved):
            detail += ("<h3>Recorded before play</h3><p>The saved ML winner and price stay fixed even if the forecast or market changes later. "
                       "The main record requires both teams to have played at least 10 games.</p>")
        if str(row.get("model_tag", "")) in u.AVAIL_TAGS:
            detail += "<p>The model includes the NBA injury report available when this forecast was saved.</p>"
        detail += self.factor_details(row, context) + self.injury_details(row)
        detail += "<p><a href='how-it-works.html'>How ML forecasts work →</a></p>"
        order_attr = f" data-order='{int(order)}'" if order is not None else ""
        return (f"<details class='gcard tool-card' data-gid='{u.esc(row.game_id)}' data-tracked='{int(bool(len(saved)))}' data-state='{'final' if final else 'upcoming'}'{order_attr} data-stage='{stage}'{live_spread}>"
                f"<summary><div class='g-meta'><span>{u.tip_pt(row.tip_utc)}</span><span class='badge st'>{status}</span></div>"
                + "".join(teams) + f"<div class='stage-label'>{stage_text}</div>" + live + forecast + tracking
                + "<span class='details-cue'>Game details<i aria-hidden='true'>⌄</i></span></summary>"
                + f"<div class='g-detail'>{detail}</div></details>")

    def calibrated_readout(self, row, team):
        u = self.ui
        estimate = self.calibration.estimate(row)
        if estimate is None:
            return ("<div class='calibration-main' data-season-calibration><small>Realized WP</small><strong>—</strong>"
                    "<small>Requires a v11 forecast, team experience and valid DraftKings prices.</small></div>")
        n, wins, losses = estimate['games'], estimate['wins'], estimate['losses']
        support = (f"Pooled cell: {wins}–{losses} across {n} game{'s' if n != 1 else ''}, all teams" if n
                   else 'No historical results in this cell yet')
        if n:
            gap = 100 * (estimate['cell_rate'] - estimate['cell_market'])
            support += (f" · <span data-cell-adjustment>won {u.pct(estimate['cell_rate'])} vs a "
                        f"{u.pct(estimate['cell_market'])} market average ({gap:+.1f} pts)</span>")
        return (f"<div class='calibration-main' data-season-calibration data-cell-games='{n}' data-cell-wins='{wins}'>"
                f"<small>Realized WP</small><strong>{u.pct(estimate['probability'])}</strong><small>{u.esc(team)} estimate</small></div>"
                f"<div class='calibration-evidence'><small>{support}</small>"
                f"<small>{estimate['previous_games']} previous season + {estimate['current_games']} current season · prices at {u.tip_pt(row.get('snapshot_utc'))}</small></div>")

    def live_block(self, row, dk):
        """The live win chance (owner request, 2026-11-07; live_wp.py): the
        card's frozen DraftKings home spread as data-live-spread and an empty,
        hidden readout that window.nbaLive fills from ESPN's live score while
        the game is in progress. Nothing without a fit or a DraftKings spread;
        no number is written at build time."""
        u = self.ui
        spread = pd.to_numeric(row.get("pre_spread"), errors="coerce") if dk else np.nan
        if not self.live_fit or not np.isfinite(spread):
            return "", ""
        side = lambda s: f"<b>{u.esc(row[s])}</b>"  # noqa: E731
        block = ("<div class='live-wp' data-live-wp hidden><div class='lwp-head'><small>Live win chance</small>"
                 "<span>Score, time left and spread at tip</span></div>"
                 f"<div class='lwp-row'><span>{side('away')} <em data-wp='away'>—</em></span>"
                 "<span class='lwp-bar' aria-hidden='true'><i></i></span>"
                 f"<span><em data-wp='home'>—</em> {side('home')}</span></div></div>")
        return f" data-live-spread='{spread:g}'", block

    def factor_details(self, row, context):
        u = self.ui
        heading = '<h3>Four factors · season to date</h3>'
        labels = (('Effective FG% ↑', '%', True), ('Turnovers / 100 poss. ↓', '', False),
                  ('Offensive rebound% ↑', '%', True), ('FTA / FGA ↑', '%', True),
                  ('Effective FG% ↓', '%', False), ('Turnovers / 100 poss. ↑', '', True),
                  ('Offensive rebound% ↓', '%', False), ('FTA / FGA ↓', '%', False))
        if all(context[s]['factors'] is None for s in ('away', 'home')):
            return heading + '<p class="context-note">Season comparison unavailable: no complete prior-game box-score totals were saved.</p>'
        html = (heading + "<table class='factor-table'><caption class='sr-only'>Season four-factor comparison before this slate</caption>"
                f"<thead><tr><th scope='col'>Factor</th><th scope='col'>{u.esc(row.away)}</th><th scope='col'>{u.esc(row.home)}</th></tr></thead><tbody>")
        for i, (label, unit, higher) in enumerate(labels):
            if i in (0, 4):
                html += f"<tr class='factor-group'><th colspan='3' scope='rowgroup'>{'Team offense' if i == 0 else 'Opponent rates allowed'}</th></tr>"
            values = [np.nan if context[s]['factors'] is None else context[s]['factors'][i] for s in ('away', 'home')]
            html += f"<tr><th scope='row'>{label}</th>"
            for j, value in enumerate(values):
                # Do not mark invisible differences between identically rounded values.
                a, b = round(value, 1), round(values[1-j], 1)
                best = np.isfinite(values).all() and ((a > b) if higher else (a < b))
                text = f'{value:.1f}{unit}' if np.isfinite(value) else '—'
                html += f"<td class='{'factor-best' if best else ''}'>{text}</td>"
            html += '</tr>'
        samples = ' · '.join(f"{u.esc(row[s])}: {context[s]['games']} games" for s in ('away', 'home') if context[s]['factors'] is not None)
        before = (pd.Timestamp(row.slate_date) - pd.Timedelta(days=1)).strftime('%b %-d, %Y')
        return (html + '</tbody></table>' + f"<p class='context-note'>Regular season through {before} · {samples}. "
                'Basketball-Reference season totals, without decay or prior-season blending. Arrows show the better direction; shaded values lead this comparison. '
                'eFG% = (FG + 0.5 × 3P) / FGA; rebound% = ORB / (ORB + opponent DRB); FTA / FGA measures free-throw attempts per field-goal attempt.</p>')

    def injury_details(self, row):
        u = self.ui
        html = '<h3>Player injuries</h3>'
        g = self.injuries[self.injuries.game_id.astype(str).eq(str(row.game_id))]
        at = pd.to_datetime(g.snapshot_utc, utc=True, errors='coerce')
        tip = pd.to_datetime(row.tip_utc, utc=True, errors='coerce')
        g = g[at.notna() & (at < tip)]
        if not len(g):
            return html + '<p class="context-note">No pregame injury list was captured for this game.</p>'
        order = {'Out': 0, 'Doubtful': 1, 'Questionable': 2, 'Day-To-Day': 3, 'Probable': 4}
        for side in ('away', 'home'):
            team = row[side]
            t = g[g.team.eq(team)]
            # Keep only this team's latest captured list, never a mixed edition.
            if len(t):
                at = pd.to_datetime(t.snapshot_utc, utc=True)
                t = t[at.eq(at.max())]
            listed = t[~t.status.isin(('NONE', ''))].copy()
            listed['order'] = listed.status.map(order).fillna(len(order))
            items = []
            for _, player in listed.sort_values(['order', 'name'], kind='stable').iterrows():
                detail = f'<small>{u.esc(player.detail)}</small>' if player.detail else ''
                items.append(f"<li><div class='inj-player'><b>{u.esc(player['name'])}</b><span class='badge'>{u.esc(player.status)}</span></div>{detail}</li>")
            if not items:
                items = ['<li>None listed in captured report.</li>' if len(t) else '<li>Injury list unavailable for this team.</li>']
            captured = pd.Timestamp(t.snapshot_utc.iloc[0]).tz_convert(u.PT) if len(t) else None
            stamp = f"<small>Captured {captured.strftime('%b %-d, %Y · %-I:%M %p PT')}</small>" if captured is not None else ''
            html += f"<section class='inj-team'><h4>{u.esc(u.TEAM_NAMES.get(team, team))} · {u.esc(team)}</h4><ul>{''.join(items)}</ul>{stamp}</section>"
        return (html + "<p class='context-note'>Source: ESPN pregame injury list. Display context only; the model uses the NBA’s official injury report separately. "
                'Lists refresh before tip and freeze afterward; the capture time can differ from the model lock.</p>')

    def today_page(self):
        u = self.ui
        day = self.native[self.native.slate_date.astype(str).eq(self.today)]
        title = pd.Timestamp(self.today).strftime("%A, %B %-d")
        body = self.head(title, "Model win chances for each game. Picks lock five hours before tip.", True).replace("class='public-head'", "class='public-head today-head'")
        body += ("<div class='public-intro today-intro'><p>DraftKings prices · Pacific time<span class='intro-extra'> · Main record: both teams at 10+ games</span>"
                 "<span data-scores-asof hidden> · Scores as of <time></time></span></p>"
                 "<span class='intro-links'><a href='performance.html'>ML performance →</a><a href='how-it-works.html'>How to read →</a></span></div>")
        if not len(day):
            body += "<div class='public-empty'><h2>No games to show</h2><p>There are no saved regular-season forecasts for this slate. Check back for the next update.</p></div>"
        else:
            body += "<div class='game-filters' data-filter hidden>"
            for group, label, default, options in (
                    ('stage', 'Team experience', 'established', (("established", "10+ games"), ("all", "All games"), ("early", "Games 1–9"))),
                    ('state', 'Game status', 'all', (("all", "All"), ("upcoming", "Upcoming"), ("live", "Live"), ("final", "Final")))):
                body += f"<div class='filter-group'><span>{label}</span><div class='seg' role='group' aria-label='{label}'>" + "".join(
                    f"<button type='button' data-group='{group}' data-show='{key}' aria-pressed='{'true' if key == default else 'false'}'>{name}<span class='ct'>0</span></button>"
                    for key, name in options) + "</div></div>"
            body += "</div>"
            # tip order, games already final at the bottom (the browser keeps
            # live games first and re-sorts as states change: FILTER_JS)
            rows = list(day.sort_values("tip_utc", kind="stable").iterrows())
            done = [np.isfinite(pd.to_numeric(r.get("home_won"), errors="coerce")) for _, r in rows]
            cards = sorted(((bool(f), i, self.card(r, i)) for i, ((_, r), f) in enumerate(zip(rows, done))),
                           key=lambda t: (t[0], t[1]))
            body += f"<div class='tool-grid' data-slate='{u.esc(self.today)}'>" + "".join(c for *_, c in cards) + "</div>"
            body += ("<div class='public-empty' data-filter-empty role='status' hidden><p>No games match these filters.</p><button type='button' data-filter-reset>Show all games</button></div>"
                     + FILTER_JS + live_wp.script(self.live_fit) + u.SCORES_JS)
        return self.page("Today · NBA Composite", "index.html", body)

    def results_page(self):
        body = self.head("ML results", "Every v11 moneyline pick and how it finished, newest first. Live picks and historical replays are labeled.")
        body += self.prediction_results() + self.void_note() + PREDICTION_FILTER_JS
        return self.page("Results · NBA Composite", "results.html", body)

    @staticmethod
    def prediction_result(row):
        return "Pending" if pd.isna(row.prediction_won) else "Win" if row.prediction_won == 1 else "Loss"

    def prediction_results(self):
        u, d = self.ui, analysis.model_prediction_moneyline(self.predictions)
        body = self.prediction_moneyline_note(d)
        if not len(d):
            return body + "<div class='public-empty'><h3>No locked v11 predictions yet</h3><p>Predictions appear here after a valid five-hour snapshot is saved.</p></div>"
        options = (("stage", "Prediction stage", list(PREDICTION_STAGES)+[("all", "All stages")]),
                   ("source", "Record source", [("all", "Mixed · live + replay"),("live", "Saved live locks"),("replay", "Historical replay")]),
                   ("season", "Prediction season", [("all", "All seasons")]+[(str(s),u.season_txt(s)) for s in sorted(set(d.season.astype(int)), reverse=True)]),
                   ("outcome", "Prediction outcome", [("all", "All outcomes"),("win", "Win"),("loss", "Loss"),("pending", "Pending")]))
        body += "<div class='public-filters'>" + "".join(
            f"<div><label for='prediction-{key}'>{name}</label><select id='prediction-{key}' data-prediction-filter='{key}'>"
            + "".join(f"<option value='{v}'>{label}</option>" for v,label in opts)+"</select></div>"
            for key,name,opts in options) + "</div>"
        labels = ["Game", "Model vs market", "Source / price", "Actual result", "Closing favorite",
                  "Moneyline return", "Running ROI"]
        widths = (17, 21, 14, 10, 12, 12, 14)
        rows = []
        for _, r in d.sort_values(['tip_utc','game_id'], ascending=False).iterrows():
            result = self.prediction_result(r)
            q = pd.to_numeric(r.get('close_q_home'), errors='coerce')
            done = pd.notna(r.prediction_won)
            close_prices = [pd.to_numeric(r.get(c), errors='coerce') for c in ('close_home_ml','close_away_ml')]
            if done and r.get('close_book') == analysis.HYPOTHESIS_BOOK and 0 < q < 1 and np.isfinite(close_prices).all():
                fav = r.home if q >= .5 else r.away
                won = (q >= .5) == (float(r.home_won) == 1)
                closing = f"{u.esc(fav)}<small>{'Win' if won else 'Loss'} · {u.pct(max(q,1-q))}</small>"
            else:
                closing = '—<small>' + ('Close unavailable' if done else 'Awaiting close') + '</small>'
            replay = r.prediction_source == 'replay'
            source_q_home = pd.to_numeric(r.get('close_q_home' if replay else 'lock_q_home'), errors='coerce')
            source_book = r.get('close_book' if replay else 'lock_book')
            same_home = r.prediction == r.home
            source_q = (source_q_home if same_home else 1-source_q_home) if source_book == 'dk' and 0 < source_q_home < 1 else np.nan
            close_q = (q if same_home else 1-q) if r.get('close_book') == 'dk' and 0 < q < 1 else np.nan
            comparison = (f"{u.esc(r.prediction)}<small>Model: {u.pct(r.prediction_p)} · {'replay' if replay else '5 h lock'}</small>"
                          f"<small>Market implied: {u.pct(source_q)} · {'close' if replay else '5 h lock'} · no vig</small>")
            if not replay:
                comparison += f"<small>Market implied: {u.pct(close_q)} · close · no vig</small>"
            provenance = (f"Historical replay<small>{u.esc(r.basis)} · DraftKings close</small>" if replay else
                          f"Saved live lock<small>{u.tip_pt(r.lock_snapshot_utc)} · {u.esc(str(r.lock_snapshot_utc)[:10])}</small>")
            cells = [f"{u.esc(r.away)} @ {u.esc(r.home)}<small>{u.esc(r.slate_date)} · {u.tip_pt(r.tip_utc)}</small>"
                     f"<small>{dict(PREDICTION_STAGES)[r.prediction_stage]} · {u.season_txt(int(r.season))}</small>",
                     comparison,
                     provenance,
                     f"<span class='prediction-badge'>{result}</span><small>{u.esc(r.prediction)}</small>", closing,
                     *self.prediction_moneyline_cells(r)]
            rows.append(f"<tr data-prediction-row data-stage='{r.prediction_stage}' data-season='{int(r.season)}' data-outcome='{result.lower()}' data-source='{r.prediction_source}'{' hidden' if r.prediction_stage != 'established' else ''}>"
                        + "".join(f"<td data-label='{label}'>{value}</td>" for label,value in zip(labels,cells))+"</tr>")
        return (body + "<table class='public-table prediction-table'><colgroup>" + "".join(f"<col style='width:{w}%'>" for w in widths)
                + "</colgroup><thead><tr>"+"".join(f"<th>{l}</th>" for l in labels)
                + "</tr></thead><tbody>"+"".join(rows)+"</tbody></table>"
                + "<div class='table-more' data-predictions-more hidden><span data-predictions-count></span><button type='button'>Show more</button></div>"
                + "<div class='public-empty' data-predictions-empty hidden><p>No predictions match these filters.</p></div>")

    def prediction_moneyline_cells(self, r):
        """Results' two moneyline cells for one prediction (owner request,
        2026-11-09): its flat 1-unit return at the DraftKings lock price,
        and its season × stage's ROI through that game."""
        u = self.ui
        if pd.isna(r.ml):
            return ["—<small>No DraftKings price at the lock</small>", "—"]
        price = f"<small>{u.ml_txt(r.ml)} · {'replay close' if r.prediction_source == 'replay' else '5 h lock'}</small>"
        if pd.isna(r.ml_units):
            return ["—" + price, "—"]
        return [f"{r.ml_units:+.2f}u" + price,
                f"{100 * r.ml_run_roi:+.1f}%<small>{int(r.ml_run_n)} priced · {r.ml_run_units:+.2f}u</small>"]

    def mixed_note(self):
        return ("<p class='public-note'><b>Mixed record:</b> historical v11 replays and saved live picks are combined, "
                "without duplicate games. Replays use DraftKings closing prices; live picks use five-hour lock prices. "
                "The ML rule was selected after historical outcomes were known. This is not a pure forward test.</p>")

    def prediction_moneyline_note(self, d):
        return ("<p class='public-note' data-prediction-moneyline><b>Mixed v11 record · flat 1u.</b> Live picks use five-hour lock prices; "
                "historical replays use closing prices. <a href='performance.html'>ML performance →</a></p>"
                "<details class='public-note'><summary>How to read this table</summary>"
                "<p>Model and market probabilities refer to the same selected team; Actual result is that team’s Win or Loss. Market implied probabilities remove the bookmaker’s margin. A realized win rate is measured across games on ML stats.</p>"
                + self.mixed_note()
                + "<p><b>Moneyline return</b> stakes 1 unit per priced winner. "
                "<b>Running ROI</b> combines live and replay within each season and stage in game order; changing a filter does not recalculate it. "
                "Missing prices keep their forecast W/L but have no return. No edge or confidence filter. "
                "<a href='performance.html'>ROI, uncertainty and market-price baseline →</a></p></details>")

    def prediction_performance(self):
        u, d = self.ui, analysis.model_prediction_moneyline(self.predictions)
        seasons = sorted(set(d.season.astype(int)) if len(d) else {self.season}, reverse=True)
        stages = [(k, v) for k, v in PREDICTION_STAGES if k in ('established', 'early') or (len(d) and d.prediction_stage.eq(k).any())]
        sources = [('live', 'Saved live locks'), ('replay', 'Historical replay'), ('all', 'Mixed · live + replay')]
        season_options = [('all', 'All seasons')] + [(str(x), u.season_txt(x)) for x in seasons]
        options = [('stage', 'Model stage', stages), ('source', 'Record source', sources), ('season', 'Season', season_options)]
        body = "<section id='model-predictions'>" + self.mixed_note()
        body += "<div class='public-filters'>" + ''.join(
            f"<div><label for='performance-{key}'>{name}</label><select id='performance-{key}' data-performance-filter='{key}'>"
            + ''.join(f"<option value='{v}'>{label}</option>" for v, label in opts) + '</select></div>'
            for key, name, opts in options) + '</div>'
        # Compare sources within the selected stage and season, independently of
        # the detailed source filter. No sources or stages are pooled here.
        for stage, label in stages:
            for season, season_label in season_options:
                part = d
                if len(part):
                    part = part[part.prediction_stage.eq(stage)]
                    if season != 'all':
                        part = part[part.season.astype(int).eq(int(season))]
                hidden = '' if (stage, season) == ('established', 'all') else ' hidden'
                body += f"<section class='source-summary' aria-label='Live and replay comparison' data-source-summary data-stage='{stage}' data-season='{season}'{hidden}>"
                for source, name in [('live', 'Saved live locks'), ('replay', 'Historical replay')]:
                    subset = part[part.prediction_source.eq(source)] if len(part) else part
                    r = analysis.model_prediction_summary(subset)
                    ml = analysis.model_prediction_moneyline_summary(subset)
                    record = f"{r['w']}–{r['l']}" if r['n'] else '—'
                    roi = f"{100 * ml['roi']:+.1f}%" if ml else '—'
                    timing = 'Saved five-hour prices' if source == 'live' else 'Reconstructed at closing prices'
                    body += (f"<article><h3>{name}</h3><p>{label} · {season_label}</p>"
                             f"<p><strong>{record}</strong> W/L · <strong>{roi}</strong> ROI</p>"
                             f"<p>{r['n']} graded · {r['pending']} pending · {ml['n'] if ml else 0} priced</p><p>{timing}</p></article>")
                body += '</section>'
        for stage, label in stages:
            for source, source_label in sources:
                for season, season_label in season_options:
                    part = d
                    if len(part):
                        part = part[part.prediction_stage.eq(stage)]
                        if source != 'all':
                            part = part[part.prediction_source.eq(source)]
                        if season != 'all':
                            part = part[part.season.astype(int).eq(int(season))]
                    hidden = '' if (stage, source, season) == ('established', 'live', 'all') else ' hidden'
                    body += (f"<article class='ml-hero' data-ml-panel data-stage='{stage}' data-source='{source}' data-season='{season}'{hidden}>"
                             f"<h2>{label}</h2><div class='record-label'>{season_label} · {source_label} · v11</div>"
                             + self.ml_performance_panel(part) + '</article>')
        return body + '</section>' + PERFORMANCE_FILTER_JS

    def ml_performance_panel(self, part):
        u = self.ui
        r = analysis.model_prediction_summary(part)
        ml = analysis.model_prediction_moneyline_summary(part)
        if not len(part):
            return "<div class='public-empty'><h3>No predictions in this view yet</h3><p>Games enter the 10+ record only after both teams have completed 10 games and a valid prediction is available.</p></div>"
        live_n = int(part.prediction_source.eq('live').sum())
        replay_n = len(part) - live_n
        roi = f"{100 * ml['roi']:+.1f}%" if ml else '—'
        units = f"{ml['units']:+.2f}u" if ml else '—'
        record = f"{r['w']}–{r['l']}" if r['n'] else '—'
        body = ("<div class='ml-metrics'>"
                f"<div><small>Moneyline ROI</small><strong>{roi}</strong></div>"
                f"<div><small>Net return · flat 1u</small><strong>{units}</strong></div>"
                f"<div><small>Winner record</small><strong class='record'>{record}</strong></div>"
                f"<div><small>Winner accuracy</small><strong>{u.pct(r['accuracy'])}</strong></div></div>")
        facts = [f"{r['n']} graded · {r['pending']} pending",
                 f"{live_n} saved live locks · {replay_n} historical replays",
                 f"95% Wilson interval: {u.pct(r['low'], 0)}–{u.pct(r['high'], 0)}"]
        if ml:
            error = f"±{u.se_txt(ml['roi_se'])} pp (1 SE)" if np.isfinite(ml['roi_se']) and ml['n'] > 1 else 'not estimable from one pick'
            null = f"{100 * ml['roi_null']:+.1f}%" if np.isfinite(ml['roi_null']) else 'unavailable'
            bench = f"{100 * ml['fav_roi']:+.1f}% on {ml['fav_n']} matched priced games" if ml['fav_n'] else 'unavailable'
            facts += [f"{ml['n']} priced picks · ROI uncertainty {error}",
                      f"Market-price baseline {null}", f"Same-game market favorite ROI: {bench}"]
        body += "<ul class='panel-facts'>" + "".join(f"<li>{f}</li>" for f in facts) + "</ul>"
        if ml:
            ordered = part[part.ml_units.notna()].sort_values(['tip_utc', 'game_id'])
            body += "<details><summary>Return over time</summary>" + u.units_svg(ordered.ml_units.tolist(), 'Cumulative ML return', width=800) + '</details>'
        else:
            body += "<p class='public-note'>No graded, priced ML picks yet. Pending games have no return.</p>"
        if ml and ml['n'] > 1 and np.isfinite(ml['roi_se']) and abs(ml['roi']) <= ml['roi_se']:
            body += "<p class='public-note'>The observed ROI is within one standard error of zero; this sample does not separate the return from break-even at that uncertainty level.</p>"
        body += ("<p class='public-note'>Every result is retained. Accuracy counts winners; ROI includes the price paid, so a high win rate can still lose money. "
                 "Missing-price forecasts stay in winner W/L. Small samples have wide uncertainty; a high win rate alone does not establish profitable betting.</p>")
        c = r['comparison']
        body += "<section class='ml-section'><h3>Against the closing market</h3>"
        if c:
            mw, fw = (int(round(c[k]['acc'] * c['n'])) for k in ('model', 'market'))
            losses = [c['model'][k] - c['market'][k] for k in ('logloss', 'brier')]
            quality = 'worse than' if all(x > 0 for x in losses) else 'better than' if all(x < 0 for x in losses) else 'mixed against'
            body += (f"<p class='public-note'>Closing favorite: {fw}–{c['n']-fw} · {u.pct(c['market']['acc'])}. "
                     f"Model on those same {c['n']} games: {mw}–{c['n']-mw}. "
                     f"The model’s probability scores are {quality} the closing market in this view.</p>"
                     "<table class='prediction-comparison'><caption>Probability quality · lower is better</caption>"
                     "<thead><tr><th>Score</th><th>Model</th><th>Market at close</th></tr></thead><tbody>"
                     + ''.join(f"<tr><td>{name}</td><td>{c['model'][key]:.4f}</td><td>{c['market'][key]:.4f}</td></tr>"
                               for key, name in [('logloss', 'Log loss'), ('brier', 'Brier score')]) + '</tbody></table>'
                     "<details><summary>What these scores mean</summary><p>Log loss and Brier score measure how well probabilities match outcomes; lower is better. "
                     "The closing market is a later benchmark than a five-hour pick: it has seen more news.</p></details>")
        else:
            body += "<p class='public-note'>The market comparison appears when graded games have usable DraftKings closing prices.</p>"
        body += "</section>"
        return body + self.roi_matrix(part)

    def roi_matrix(self, part):
        """Latest 50 graded, priced model-winner picks per joint WP cell.

        Part already has the page's stage/source/season filters. Native rows
        use saved lock P/prices; replay rows use saved model P/closing prices.
        """
        if not len(part):
            return ''
        u = self.ui
        d = analysis.model_prediction_moneyline(part)
        cells = {}
        for _, r in d.sort_values(['slate_date', 'game_id'], kind='stable').iterrows():
            if pd.isna(r.ml_units) or pd.isna(r.prediction_won):
                continue
            prefix = 'lock' if r.prediction_source == 'live' else 'close'
            q_home = market.devig(r.get(f'{prefix}_home_ml'), r.get(f'{prefix}_away_ml'))
            if not np.isfinite(q_home):
                continue
            q = q_home if r.prediction == r.home else 1-q_home
            key = (wp_calibration.band(r.prediction_p, wp_calibration.MODEL_BOUNDS),
                   wp_calibration.band(q, wp_calibration.MARKET_BOUNDS))
            cells.setdefault(key, []).append(r)
        model_labels = ['50–60%', '60–70%', '70–80%', '80–100%']
        market_labels = ['<50%', '50–60%', '60–70%', '70–80%', '80–100%']
        body = ("<section class='ml-section'><h3>ROI by model and market win probability</h3>"
                "<p class='public-note'>Each square: 1 unit on the model’s pick, latest 50 priced results in that band pair.</p>"
                "<div class='roi-matrix-wrap'><table class='roi-matrix' data-roi-matrix>"
                "<caption class='sr-only'>ROI lookup matrix · latest 50 per cell</caption>"
                "<thead><tr><th scope='col'>Market WP →<br>Model WP ↓</th>"
                + ''.join(f"<th scope='col'>{label}</th>" for label in market_labels) + '</tr></thead><tbody>')
        retained, units = 0, 0.0
        for pb, label in enumerate(model_labels):
            body += f"<tr><th scope='row'>{label}</th>"
            for qb in range(len(market_labels)):
                cell = cells.get((pb, qb), [])[-wp_calibration.WINDOW_GAMES:]
                n = len(cell)
                if not n:
                    body += f"<td data-model-band='{pb}' data-market-band='{qb}' data-n='0'>—</td>"
                    continue
                profit = sum(float(r.ml_units) for r in cell)
                wins = sum(int(r.prediction_won) for r in cell)
                roi = profit/n
                color = 'positive' if profit > 0 else 'negative' if profit < 0 else 'neutral'
                body += (f"<td class='{color}' data-model-band='{pb}' data-market-band='{qb}' data-n='{n}'>"
                         f"<strong>{100*roi:+.1f}%</strong><small>{n} game{'s' if n != 1 else ''} · {wins}–{n-wins}</small>"
                         f"<small>{profit:+.2f}u</small></td>")
                retained += n
                units += profit
            body += '</tr>'
        body += '</tbody></table></div>'
        return body + (f"<p class='public-note'>{retained} retained priced results · {units:+.2f}u. "
                       "Cells are descriptive, not a filter or a betting rule. Small samples can show extreme ROI.</p>"
                       "<details><summary>How the matrix works</summary><p>"
                       "Each square stakes 1 unit on the model’s selected team; both WP bands refer to that same team, across all NBA teams. "
                       "Market WP removes the bookmaker’s margin. Below 50% includes model-picked market underdogs. "
                       "Uses the current stage/source/season filters, then the latest 50 graded priced games per cell; pending and unpriced games are excluded. "
                       "Live uses five-hour lock probabilities and prices; replay uses historical model forecasts and closing prices.</p></details></section>")

    def stage(self, hyp, season):
        u = self.ui
        records = [r for r in self.records if r['hyp']['key'] == hyp['key'] and r['season'] == season]
        grades = [r['grade'] for r in records if r['grade'] is not None]
        pending = len(records)-len(grades)
        label = "Early · games 1–9" if hyp['early'] else "Established · games 10+"
        start = f"<section class='stage-panel'><h3>{label}</h3>"
        if not grades:
            return start+f"<div class='empty-stage'>No graded picks yet</div><div class='empty-help'>{pending} pending · The record begins with the first qualifying result.</div></section>"
        d = pd.DataFrame(grades)
        ou = hyp.get('market') == 'ou'
        r = (analysis.ou_row(hyp['key'], d) if ou else
             analysis.ats_row(hyp['key'], d) if hyp.get('market') == 'ats' else analysis.roi_row(hyp['key'], d))
        pushes = r.get('push',0)
        record = f"{r['w']}–{r['l']}" + (f"–{pushes}" if pushes else "")
        tags = (rec['row'].get(f"{rec['price']}_model_tag") for rec in records)
        models = (["totals model"] if ou else
                  sorted({m.group(0) for t in tags for m in [re.search(r'v\d+$', str(t))] if m}))
        bench = (f"Same-game over ROI {100 * r['over_roi']:+.1f}%." if ou else
                 f"Same-game market favorite ROI {u.fav_txt(r)}.")
        z = analysis.z_vs_null(r)
        c = u.sig_cls(z)
        chart_records = sorted((rec for rec in records if rec['grade'] is not None),key=lambda rec:str(rec['row'].tip_utc))
        chart = u.units_svg([rec['grade']['units'] for rec in chart_records], "Cumulative return",width=560)
        uncertainty = (f"ROI uncertainty ±{u.se_txt(r['roi_se'])} pp (1 SE)."
                       if np.isfinite(r['roi_se']) else "More graded picks are needed to estimate uncertainty.")
        return (start+f"<span class='record'>{record}</span><span class='sample'>{r['n']} graded · {pending} pending</span>"
                f"<div class='stage-stats'><div>Return<b>{r['units']:+.2f}u</b></div><div>ROI<b class='{c}'>{100*r['roi']:+.1f}%</b></div></div>"
                f"<p class='null'>{uncertainty} Market-price baseline {100*r['roi_null']:+.1f}%.</p>"
                f"<details><summary>Benchmarks and return chart</summary><p>Flat 1 unit per pick. {bench} "
                f"Model at the {analysis.PRICE_NAMES[hyp['native']]}: {u.esc(' / '.join(models) or 'recorded model')}. "
                "Color marks a gap of at least two standard errors from the market-price baseline.</p>"+chart+"</details></section>")

    def restart_txt(self):
        """'October 29, 2026': the reset slate, where the v11-family record starts."""
        d = pd.Timestamp(analysis.FAMILY_SLATE)
        return f"{d:%B} {d.day}, {d.year}"

    def restart_note(self):
        """The one dated line (model-family reset amendment): the record
        restarted with the current model; earlier picks are kept apart."""
        return (f"<p class='public-note'>Tracked records restarted with the {self.restart_txt()} games, "
                "when the current model version began. Picks recorded under earlier versions are kept "
                "separately and are not included here.</p>")

    def ou2_txt(self):
        """'November 3, 2026': the last slate whose O/U picks were scored by
        the live totals model's one-time rescore (the slate before its first
        native lock; the latest rescored slate before activation): ou2's, or
        ou3's once active."""
        if self.ou == "ou3":
            act, rs = analysis.ou3_activation_slate(self.native), analysis.load_ou3_rescore()
        else:
            act, rs = analysis.ou2_activation_slate(self.native), analysis.load_ou_rescore()
        dates = rs["slate_date"].astype(str) if len(rs) and "slate_date" in rs else pd.Series(dtype=str)
        if act is not None:
            dates = dates[dates < act]
        if not len(dates):
            return None
        d = pd.Timestamp(dates.max())
        return f"{d:%B} {d.day}, {d.year}"

    def ou2_clause(self):
        """How it works only (owner, 2026-11-03: not on Results or
        Performance): one clause that O/U picks before ou2's activation were
        scored with the current totals model from lock-time information.
        Nothing without any."""
        last = self.ou2_txt()
        if last is None:
            return ""
        return f" Picks through {last} were scored with this model from the information available at each lock."

    def void_note(self):
        """The one dated line (stale game-log locks amendment, 2026-11-03):
        eight locks are void and left out of every record."""
        return ("<p class='public-note'>Picks and predictions for eight games on October 31 and "
                "November 2, 2026 are void and not counted: they locked before the previous "
                "night’s games reached the model’s game data.</p>")

    def gap_note(self):
        """The one dated line (H4 retirement amendment, 2026-11-10): a fourth
        test left the pages; its picks are kept apart. No H4 figure."""
        d = pd.Timestamp(analysis.GAP_CLOSE_SLATE) - pd.Timedelta(days=1)
        return ("<p class='public-note'>A fourth test, large probability gap, was retired after the "
                f"{d:%B} {d.day}, {d.year} games. Its picks are kept separately and are not included here.</p>")

    def performance_page(self):
        body = self.head("ML performance", "The v11 moneyline record. Saved live locks lead; historical replay is shown separately for comparison.")
        body += self.prediction_performance() + self.void_note()
        return self.page("Performance · NBA Composite", "performance.html", body)

    def guide_page(self):
        body = self.head("How it works", "One ML forecast per game. A fixed rule and a visible record.")
        body += """<div class='guide'><section><h2>The 10+ game ML rule</h2>
<p>Both teams must have completed at least 10 regular-season games before the slate to enter the main record. Early games (1–9), opening games and games with unavailable stage data remain accessible separately.</p>
<p>Pick the team with the higher v11 win probability; at exactly 50%, pick the home team. Track a flat <b>1 unit</b> at its saved DraftKings moneyline, with both prices available. There is no edge, favorite, confidence or price-band filter.</p>
<p>A live pick freezes at the first valid priced snapshot within 5 hours of tip whose game data include every completed game. Later forecasts and closing prices never replace it. A missing lock or price means no ML return; a forecast without a price still keeps its winner W/L.</p>
<p>This rule was frozen on November 10, 2026 after reviewing historical results. Future model families need a separate decision. Accuracy is a forecast measure; it does not establish a profitable bet.</p></section>
<section><h2>Reading ML performance</h2>""" + self.mixed_note() + """
<p>The default is the <b>saved live games-10+ record across seasons</b>. Live and replay summaries appear side by side. Source and season filters retain the combined record and each source separately. Every game counts once: a native game is never replaced by a replay, including when its live lock is missing or void.</p>
<p>Historical replays must carry a v11-family tag and a graded result, and use their saved DraftKings <b>closing</b> prices. They are reconstructed or rebuilt with historical inputs, rather than predictions captured live. Replays do not demonstrate executable five-hour returns. The record never uses another book’s prices.</p>
<p><b>ROI</b> is net return divided by stakes. Uncertainty is ±1 standard error, narrower than a 95% interval. The market-price baseline is the return expected if the no-vig market probability is right, including the bookmaker’s margin. The same-game favorite is a separate betting benchmark.</p>
<p>Winner accuracy has a 95% Wilson interval. Log loss and Brier compare the model with the closing DraftKings market on identical games; lower is better. <b>ROI lookup matrix</b> groups model-selected teams jointly by saved model WP (rows) and no-vig DraftKings WP for the same team (columns). It applies the stage, source and season filters, then retains the latest 50 completed priced picks in each cell, ordered by slate date and game ID. Each pick stakes 1 unit at its source price: five-hour locks for live and closing prices for replay. Cells show net ROI, count, W/L and units; missing prices and pending games do not enter. Model-picked market underdogs appear below 50% market WP. The filtered matrix can have a different population from the cards’ pooled previous/current-season Realized WP. Cells are descriptive and never select picks. Results keeps each game, source, price and outcome. Its running ROI combines sources within that season and stage and does not change with filters.</p>""" + self.void_note() + """</section>
<section><h2>Reading today’s games</h2>
<p><b>Model probability</b> is the chance of the model’s predicted winner winning; <b>DraftKings no vig</b> is that team’s current market-implied win probability with the bookmaker’s margin removed. Posted moneylines remain beside each team. The <b>break-even probability</b> is the win rate needed to cover that price, including the bookmaker’s margin. A predicted winner is not necessarily good value. Model and market probabilities refer to the same team and can update until tip. Actual result is that team’s Win or Loss; realized win rates across games appear in ML performance. A saved <b>locked pick</b> is shown separately at its original price. Times are Pacific.</p>
<p><b>Live win chance</b> updates from the current score, time left and spread at tip every 30 seconds while games are in progress. It is display context, is not saved and never changes a pick or record.</p>
<p>Game details retain the predicted margin, both teams’ season records, four factors from games before the slate, and timestamped ESPN injuries. The model uses the NBA’s official injury report separately.</p></section>
<section id='in-season-calibration'><h2>Realized WP</h2>
<p>This experimental estimate is a lookup of realized model–market interactions for the <b>same team as the current model forecast</b>. Model WP selects the matrix row and no-vig DraftKings WP selects the column. The cell pools games across <b>all NBA teams</b>; it is not that franchise’s record.</p>
<p><b>Realized WP = this game’s no-vig DraftKings WP + (cell win rate − cell average no-vig market WP)</b>, limited to 0–100%. The cell win rate is wins ÷ games among the latest 50 actual results in that cell, and the cell average is those same games’ no-vig DraftKings WP for their model-selected teams. A market band is 10 to 20 points wide, so the cell’s games rarely carry the same price as tonight’s game; the adjustment measures how far the cell beat or missed its own prices and applies that difference to this game’s price. Since November 12, 2026; before that the display was the raw cell win rate.</p>
<p>There are no market pseudo-games and no shrinkage. Use all available results if a cell has fewer than 50; an empty cell has no estimate. With 50 games the cell win rate is uncertain by roughly ±6–7 points (1 SE), more than a typical bookmaker margin, and a small cell can push the estimate to 0% or 100%. That is an observed frequency and does not establish certainty about the next game.</p>
<p>The <b>immediately previous season</b> supplies the starting matrix. Current-season evidence uses valid graded native v11 five-hour locks on strictly earlier slates. Same-season replay, older seasons, pending games and void locks do not enter. New results replace the oldest matching observations once the cell reaches 50. Order is slate date, then game ID for ties.</p>
<p>Early games and games 10+ share calibration evidence; opening games remain separate. Each historical game contributes once, oriented to its model’s selected team. Model and market picks can disagree: a model-picked market underdog belongs in a market-WP-below-50% cell. The other side’s implied estimate is the complement; the two sides are not counted as independent results.</p>
<p>Previous-season replay uses its saved v11 model forecast and DraftKings closing prices. Current native evidence uses saved five-hour model probabilities and prices. These price timings differ, and the available previous-season DraftKings seed contains games 10+ only. The card shows how many retained observations come from each season.</p>
<p>Model bands: 50–60%, 60–70%, 70–80%, 80–100%. Market bands: below 50%, 50–60%, 60–70%, 70–80%, 80–100%. Boundaries enter the higher band; 100% enters the final band.</p>
<p>The current forecast and price define the new game’s cell and can update until tip. This display is separate from live win chance and the frozen pick. It does not change the model, tracked selections, grades or separate early/10+ performance records. A predictive or betting advantage has not been established.</p></section>
<section><h2>The model</h2>
<p>The same v11 model predicts a points margin and converts it to a win probability. Inputs include recent shooting, turnovers, rebounding and free-throw rates for and against; home court, back-to-backs and season phase; roster value and the NBA injury report. Last season’s team rating fades out over the first 10 games. Market prices are benchmarks, never model inputs.</p>""" + self.model_note() + """</section>
"""
        body += dfs.guide() + "</div>"
        return self.page("How it works · NBA Composite", "how-it-works.html", body)

    def model_note(self):
        """How it works: the live v10 / v11 model in plain words, and the one
        dated line about earlier picks (CLAUDE.md v10 and v11 amendments; for
        v11, the model-family reset's restart line). Nothing when the v9 files
        are live."""
        tags = [str(t) for t in getattr(self.ui, "ACTIVE_TAGS", [])]
        ver = "_v11" if any(t.endswith("_v11") for t in tags) else \
            "_v10" if any(t.endswith("_v10") for t in tags) else None
        if ver is None:
            return ""
        if ver == "_v11":
            dated = (f"<p class='note'>The current model version’s saved live record began with the {self.restart_txt()} games. "
                     "Picks recorded under earlier versions are kept separately from the v11 record. Historical v11 replays are labeled independently.</p>")
        else:
            first = pd.Series(dtype=str)
            if len(self.native) and "first_model_tag" in self.native:
                new = self.native["first_model_tag"].astype(str).str.endswith(ver)
                first = pd.to_datetime(self.native.loc[new, "first_snapshot_utc"], utc=True, errors="coerce").dropna()
            when = first.min().tz_convert(self.ui.PT) if len(first) else pd.Timestamp(self.today)
            dated = (f"<p class='note'>Model updated {when:%B} {when.day}, {when.year}. "
                     "Earlier tracked picks keep the probabilities of the previous version, recorded before their games.</p>")
        questionable = (" Players listed as questionable, doubtful or probable count by how often players with that status have played at that point before tip."
                        if ver == "_v11" else "")
        return ("<p>Drafted rookies are valued by draft position until they have an NBA record, and the injury-report part is re-estimated before each slate with this season’s completed games."
                + questionable + "</p>" + dated)

    def render(self):
        return {"index.html":self.today_page(),"results.html":self.results_page(),
                "performance.html":self.performance_page(),"how-it-works.html":self.guide_page(),
                # a separate tool (CLAUDE.md, DFS lineup page): no record or forecast
                "dfs.html":dfs.page(self.ui, self.today, self.built, native=self.native, css=CSS)}
