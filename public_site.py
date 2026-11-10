"""Public presentation of native forecasts and the registered v11-family tests.

Research renderers, reconstructed data and legacy lean records are not inputs.
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
from matchup_context import MatchupContext

PAGES = (("index.html", "Today"), ("results.html", "Results"),
         ("performance.html", "Performance"), ("how-it-works.html", "How it works"),
         ("dfs.html", "DFS"))
RULES = (("H5·3", "Spread", "spread"), ("H2·F·5", "Value favorite", "favorite"),
         ("OU", "Total (over/under)", "total"))
PREDICTION_STAGES = (("early", "Early · games 1–9"),
                     ("established", "Established · games 10+"),
                     ("opener", "Opening games · game 0"), ("unknown", "Stage unavailable"))


def rule_key(root, early, ou="ou2"):
    """The public row of a test and stage (CLAUDE.md amendments of
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
.public-table{width:100%;border-collapse:collapse;font-size:13px;background:var(--surface)}
.public-table th{text-align:left;color:var(--muted);font-size:10px;letter-spacing:.04em;text-transform:uppercase;padding:14px 12px;background:var(--surface-2)}
.public-table td{text-align:left;padding:14px 12px;border-top:1px solid var(--line-2)}.public-table small{display:block;color:var(--muted);font-size:11px;margin-top:3px}
.result-badge,.prediction-badge{display:inline-block;border:1px solid var(--line);border-radius:6px;padding:3px 7px;font-size:11px;font-weight:700}
.model-pick{display:flex;justify-content:space-between;gap:8px;padding:10px 0;border-top:1px solid var(--line-2);font-size:12px}
.model-pick small{display:block;font-size:10px;color:var(--muted);margin-top:3px}.model-pick>b{text-align:right;white-space:nowrap}
.prediction-grid .stage-panel{border:0;padding:0}.prediction-grid .stage-panel .sample{display:block;margin:6px 0 0}
.prediction-grid .rule-panel>p{min-height:0}.prediction-comparison{width:100%;font-size:12px;border-collapse:collapse;margin-top:14px}
.prediction-comparison th,.prediction-comparison td{text-align:right;padding:8px 4px;border-bottom:1px solid var(--line-2)}
.prediction-comparison th:first-child,.prediction-comparison td:first-child{text-align:left}.prediction-comparison th{font-size:10px;color:var(--muted)}
[data-record-panel][hidden],[data-prediction-row][hidden],[data-result-row][hidden]{display:none!important}
.public-empty{padding:32px;border:1px dashed var(--line);border-radius:14px;margin:24px 0;background:var(--surface);color:var(--muted)}
.public-empty h2{color:var(--ink);font-size:18px;margin:0 0 8px}.public-empty p{font-size:14px;margin:0;max-width:600px}
.guide{max-width:760px}.guide section{margin:28px 0;padding:22px;border:1px solid var(--line);border-radius:14px;background:var(--surface)}
.guide h2{font-size:19px;margin:0 0 12px;letter-spacing:-.02em}.guide h3{font-size:15px;margin:22px 0 6px}.guide p,.guide li{font-size:14px;color:var(--muted);line-height:1.65}
.guide p:last-child{margin-bottom:0}.guide .rule-number{font:600 11px var(--mono);color:rgb(var(--accent-tx));margin-right:8px}
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
var grid=document.querySelector('.tool-grid[data-slate]'),rank={live:0,upcoming:1,final:2};
function order(){if(!grid)return;var cs=[].slice.call(grid.querySelectorAll(':scope>.gcard')),
s=cs.slice().sort(function(a,b){return(rank[a.dataset.state]-rank[b.dataset.state])||(a.dataset.order-b.dataset.order);});
if(s.some(function(x,i){return x!==cs[i];}))s.forEach(function(x){grid.appendChild(x);});}
var filter='all';function update(){order();var c={all:0,tracked:0,upcoming:0,live:0,final:0};
document.querySelectorAll('.gcard').forEach(function(x){var s=x.dataset.state;c.all++;c[s]++;if(x.dataset.tracked==='1')c.tracked++;
x.hidden=filter==='tracked'?x.dataset.tracked!=='1':filter!=='all'&&s!==filter;});
bar.querySelectorAll('button').forEach(function(b){var k=b.dataset.show;b.querySelector('.ct').textContent=c[k];b.setAttribute('aria-pressed',k===filter?'true':'false');});
var e=document.querySelector('[data-filter-empty]');if(e)e.hidden=!!c[filter];}
window.nbaCount=update;bar.addEventListener('click',function(e){var b=e.target.closest('button');if(b){filter=b.dataset.show;update();}});update();})();</script>"""

RESULT_FILTER_JS = """<script>(function(){var controls=document.querySelectorAll('[data-result-filter]');
controls.forEach(function(c){c.addEventListener('change',function(){var n=0;document.querySelectorAll('[data-result-row]').forEach(function(r){
var show=true;controls.forEach(function(f){if(f.value!=='all'&&r.dataset[f.dataset.resultFilter]!==f.value)show=false;});r.hidden=!show;if(show)n++;});
document.querySelector('[data-results-empty]').hidden=n>0;});});})();</script>"""

PREDICTION_FILTER_JS = """<script>(function(){var controls=document.querySelectorAll('[data-prediction-filter]');
controls.forEach(function(c){c.addEventListener('change',function(){var n=0;document.querySelectorAll('[data-prediction-row]').forEach(function(r){
var show=true;controls.forEach(function(f){if(f.value!=='all'&&r.dataset[f.dataset.predictionFilter]!==f.value)show=false;});r.hidden=!show;if(show)n++;});
document.querySelector('[data-predictions-empty]').hidden=n>0;});});
var bar=document.querySelector('[data-record-view]');bar.hidden=false;
bar.addEventListener('click',function(e){var b=e.target.closest('button');if(!b)return;
bar.querySelectorAll('button').forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false');});
document.querySelectorAll('[data-record-panel]').forEach(function(x){x.hidden=x.dataset.recordPanel!==b.dataset.view;});});
document.querySelector('[data-record-panel="tracked"]').hidden=true;
})();</script>"""


class PublicSite:
    def __init__(self, ui, native, today, built, model_ok=True, context=None, injuries=None):
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
        self.predictions = analysis.model_predictions(self.native)
        # the O/U rows of the live totals model: ou3's once Fit model has
        # written it (or a native lock carries it), else ou2's
        self.ou = "ou3" if analysis.ou3_active(self.native) else "ou2"
        self.hypotheses = {h["key"]: h for h in analysis.public_hypotheses()
                           + analysis.public_ou_tests(self.native)}
        self.records = []
        # Model-family reset and edge floors (CLAUDE.md amendments 2026-10-29):
        # the floored v11-family rows (H5·3, H2·F·5) and the O/U test, at
        # the 5 h lock. Every other record stays internal (presentation only;
        # analysis defines every record).
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
        dk = str(row.get("pre_book")) == "dk"
        recs = [r for r in self.records if str(r["row"].game_id) == str(row.game_id)]
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
            forecast = (f"<div class='forecast-readout'><div><small>Model win chance</small><strong>{u.pct(prob,0)}</strong><span>{u.esc(team)}</span></div>"
                        f"<div><small>Predicted margin</small><strong>{abs(margin):.1f} <span>pts</span></strong><span>{u.esc(margin_team)} by {abs(margin):.1f}</span></div></div>")
        else:
            forecast = "<div class='forecast-only'><b>Forecast unavailable</b><br>A required model input was unavailable.</div>"
        if recs:
            tracking = "<div class='tracked-box'><span class='tracked-label'>Tracked picks · 5 h lock</span>" + "".join(
                f"<div class='tracked-line'><span>{r['name']}</span><b>{self.selection(r)}</b></div>" for r in recs) + "</div>"
        else:
            tag = str(row.get("lock_model_tag", ""))
            gp = np.fmin(number("gp_home"), number("gp_away"))
            tip = pd.to_datetime(row.get("tip_utc"), utc=True, errors="coerce")
            started = final or (pd.notna(tip) and self.built_utc >= tip)
            why = ("Season openers are outside the tracked rules." if gp == 0 else
                   f"Tracked records restart with the {self.restart_txt()} games." if str(row.slate_date) < analysis.FAMILY_SLATE else
                   ("No priced snapshot was saved in the 5 hours before tip." if started else
                    "Tracked picks lock about 5 hours before tip.") if pd.isna(row.get("lock_p_home")) else
                   "The 5 h lock used an earlier model." if not tag.endswith(analysis.FAMILY_TAGS) else
                   "No tracked rule qualified at the 5 h lock.")
            tracking = f"<div class='forecast-only'><b>Forecast only</b><br>{why}</div>"
        forecast += self.ou_readout(row, dk)
        if len(self.predictions):
            saved = self.predictions[self.predictions.game_id.astype(str).eq(str(row.game_id))]
            if len(saved):
                pick = saved.iloc[0]
                result = self.prediction_result(pick)
                forecast += (f"<div class='model-pick'><span>Model pick · 5 h lock"
                             f"<small>{u.esc(pick.prediction)} · {u.pct(pick.prediction_p)} at lock</small></span>"
                             f"<b>{result}<small><a href='results.html'>Model record →</a></small></b></div>")
        book = "DraftKings" if dk else "DraftKings unavailable; no market"
        q = number("pre_q_home") if dk else np.nan
        live_spread, live = self.live_block(row, dk)
        detail = (f"<h3>Forecast and market</h3><p>{u.esc(row.away)} @ {u.esc(row.home)} · {u.esc(book)} prices. "
                  f"Market home win probability: {u.pct(q)} with the bookmaker margin removed. "
                  f"Forecast saved at {u.tip_pt(row.get('snapshot_utc'))}. Prices and forecasts freeze at tip.</p>")
        if live:
            detail += ("<p>During the game, <b>live win chance</b> updates with the score from the current margin, "
                       "the time left and the DraftKings spread at tip, using a formula fitted on past NBA games. "
                       "It is not the model’s forecast, is not saved and never changes a tracked pick.</p>")
        if recs:
            detail += (f"<h3>Recorded before play</h3><p>Tracked at {u.tip_pt(recs[0]['at'])}. "
                       "The selections and prices above stay fixed even if the forecast or market changes later. "
                       "A game can qualify for multiple tests; each keeps its own record.</p>")
        if str(row.get("model_tag", "")) in u.AVAIL_TAGS:
            detail += "<p>The model includes the NBA injury report available when this forecast was saved.</p>"
        detail += self.factor_details(row, context) + self.injury_details(row)
        detail += "<p><a href='how-it-works.html'>How forecasts and tracked picks work →</a></p>"
        order_attr = f" data-order='{int(order)}'" if order is not None else ""
        return (f"<details class='gcard tool-card' data-gid='{u.esc(row.game_id)}' data-tracked='{int(bool(recs))}' data-state='{'final' if final else 'upcoming'}'{order_attr}{live_spread}>"
                f"<summary><div class='g-meta'><span>{u.tip_pt(row.tip_utc)}</span><span class='badge st'>{status}</span></div>"
                + "".join(teams) + live + forecast + tracking
                + "<span class='details-cue'>Game details<i aria-hidden='true'>⌄</i></span></summary>"
                + f"<div class='g-detail'>{detail}</div></details>")

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
        body = self.head(title, "Today’s model forecasts. Tracked picks are marked separately and lock about 5 hours before tip.", True)
        body += "<div class='public-intro'><p>DraftKings prices · All times Pacific (PT)<span data-scores-asof hidden> · Scores as of <time></time></span></p><a href='how-it-works.html'>How to read →</a></div>"
        if not len(day):
            body += "<div class='public-empty'><h2>No games to show</h2><p>There are no saved regular-season forecasts for this slate. Check back for the next update.</p></div>"
        else:
            body += "<div class='seg' data-filter role='group' aria-label='Filter games' hidden>" + "".join(
                f"<button type='button' data-show='{key}' aria-pressed='{'true' if key == 'all' else 'false'}'>{name}<span class='ct'>0</span></button>"
                for key,name in (("all","All"),("tracked","Tracked"),("upcoming","Upcoming"),("live","Live"),("final","Final"))) + "</div>"
            # tip order, games already final at the bottom (the browser keeps
            # live games first and re-sorts as states change: FILTER_JS)
            rows = list(day.sort_values("tip_utc", kind="stable").iterrows())
            done = [np.isfinite(pd.to_numeric(r.get("home_won"), errors="coerce")) for _, r in rows]
            cards = sorted(((bool(f), i, self.card(r, i)) for i, ((_, r), f) in enumerate(zip(rows, done))),
                           key=lambda t: (t[0], t[1]))
            body += f"<div class='tool-grid' data-slate='{u.esc(self.today)}'>" + "".join(c for *_, c in cards) + "</div>"
            body += ("<div class='public-empty' data-filter-empty hidden><p>No games match this filter.</p></div>"
                     + FILTER_JS + live_wp.script(self.live_fit) + u.SCORES_JS)
        return self.page("Today · NBA Composite", "index.html", body)

    def results_page(self):
        u = self.ui
        body = self.head("Results", "Every v11 model prediction and every tracked selection, recorded before play.")
        body += ("<div class='seg' data-record-view role='group' aria-label='Results view' hidden>"
                 "<button type='button' data-view='predictions' aria-pressed='true'>Model predictions</button>"
                 "<button type='button' data-view='tracked' aria-pressed='false'>Tracked picks</button></div>"
                 "<section data-record-panel='predictions'>" + self.prediction_results() + "</section>"
                 "<section data-record-panel='tracked'><h2 class='sec'>Tracked picks</h2>")
        body += ("<p class='public-note'>Three separate tests. A game can appear in more than one; their returns are never added together. Early games (1–9) and established games (10+) remain separate. "
                 "Picks lock at the first priced snapshot within 5 hours of tip whose game data include every completed game.</p>"
                 + self.restart_note() + self.void_note() + self.gap_note())
        if not self.records:
            body += "<div class='public-empty'><h2>The forward record starts here</h2><p>No qualifying picks have been recorded since the record restarted. Today’s forecasts remain available while the tests build their records.</p></div><p><a href='index.html'>View today’s forecasts →</a></p>"
        else:
            seasons = sorted({r['season'] for r in self.records}, reverse=True)
            options = (("rule","Test",[("all","All tests")]+[(s,n) for _,n,s in RULES]),
                       ("stage","Stage",[("all","Both stages"),("early","Early · games 1–9"),("established","Established · games 10+")]),
                       ("season","Season",[("all","All seasons")]+[(str(s),u.season_txt(s)) for s in seasons]))
            body += "<div class='public-filters'>" + "".join(f"<div><label for='filter-{key}'>{name}</label><select id='filter-{key}' data-result-filter='{key}'>"+"".join(f"<option value='{v}'>{label}</option>" for v,label in opts)+"</select></div>" for key,name,opts in options) + "</div>"
            labels = ["Game", "Test / stage", "Recorded pick", "Recorded at", "Outcome", "Return"]
            rows = []
            for rec in sorted(self.records, key=lambda r:(str(r['row'].tip_utc),r['slug']), reverse=True):
                row, grade = rec['row'], rec['grade']
                if grade is None:
                    result, ret = "Pending", "—"
                else:
                    win = grade['result'] if rec['hyp'].get('market') in ('ats', 'ou') else grade['won']
                    result = "Push" if win == .5 else "Win" if win == 1 else "Loss"
                    ret = f"{grade['units']:+.2f}u"
                stage = "Early · games 1–9" if rec['early'] else "Established · games 10+"
                cells = [f"{u.esc(row.away)} @ {u.esc(row.home)}<small>{u.esc(row.slate_date)} · {u.tip_pt(row.tip_utc)}</small>",
                         f"{rec['name']}<small>{stage} · {u.season_txt(rec['season'])}</small>", self.selection(rec),
                         f"{u.tip_pt(rec['at'])}<small>5 h lock · {u.esc(str(rec['at'])[:10])}</small>",
                         f"<span class='result-badge'>{result}</span>", ret]
                rows.append(f"<tr data-result-row data-rule='{rec['slug']}' data-stage='{'early' if rec['early'] else 'established'}' "
                            f"data-season='{rec['season']}'>"+"".join(f"<td data-label='{l}'>{v}</td>" for l,v in zip(labels,cells))+"</tr>")
            body += "<table class='public-table'><thead><tr>"+"".join(f"<th>{l}</th>" for l in labels)+"</tr></thead><tbody>"+"".join(rows)+"</tbody></table>"
            body += "<div class='public-empty' data-results-empty hidden><p>No picks match these filters.</p></div>"+RESULT_FILTER_JS
        body += "</section>" + PREDICTION_FILTER_JS
        return self.page("Results · NBA Composite", "results.html", body)

    @staticmethod
    def prediction_result(row):
        return "Pending" if pd.isna(row.prediction_won) else "Win" if row.prediction_won == 1 else "Loss"

    def prediction_results(self):
        u, d = self.ui, self.predictions
        body = ("<h2 class='sec'>Model predictions · v11</h2><p class='public-note'>One winner prediction per game, "
                "frozen at the 5 h lock, whether or not a tracked betting rule qualifies. "
                "Pending games have no W/L result. The closing favorite is a later benchmark.</p>")
        if not len(d):
            return body + "<div class='public-empty'><h3>No locked v11 predictions yet</h3><p>Predictions appear here after a valid five-hour snapshot is saved.</p></div>"
        options = (("stage", "Prediction stage", [("all", "All stages")]+list(PREDICTION_STAGES)),
                   ("season", "Prediction season", [("all", "All seasons")]+[(str(s),u.season_txt(s)) for s in sorted(set(d.season.astype(int)), reverse=True)]),
                   ("outcome", "Prediction outcome", [("all", "All outcomes"),("win", "Win"),("loss", "Loss"),("pending", "Pending")]))
        body += "<div class='public-filters'>" + "".join(
            f"<div><label for='prediction-{key}'>{name}</label><select id='prediction-{key}' data-prediction-filter='{key}'>"
            + "".join(f"<option value='{v}'>{label}</option>" for v,label in opts)+"</select></div>"
            for key,name,opts in options) + "</div>"
        labels = ["Game", "Stage / season", "Model pick", "Recorded at", "Outcome", "Closing favorite"]
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
            cells = [f"{u.esc(r.away)} @ {u.esc(r.home)}<small>{u.esc(r.slate_date)} · {u.tip_pt(r.tip_utc)}</small>",
                     f"{dict(PREDICTION_STAGES)[r.prediction_stage]}<small>{u.season_txt(int(r.season))}</small>",
                     f"{u.esc(r.prediction)}<small>{u.pct(r.prediction_p)} win chance · v11</small>",
                     f"{u.tip_pt(r.lock_snapshot_utc)}<small>5 h lock · {u.esc(str(r.lock_snapshot_utc)[:10])}</small>",
                     f"<span class='prediction-badge'>{result}</span>", closing]
            rows.append(f"<tr data-prediction-row data-stage='{r.prediction_stage}' data-season='{int(r.season)}' data-outcome='{result.lower()}'>"
                        + "".join(f"<td data-label='{label}'>{value}</td>" for label,value in zip(labels,cells))+"</tr>")
        return (body + "<table class='public-table'><thead><tr>"+"".join(f"<th>{l}</th>" for l in labels)
                + "</tr></thead><tbody>"+"".join(rows)+"</tbody></table>"
                + "<div class='public-empty' data-predictions-empty hidden><p>No predictions match these filters.</p></div>")

    def prediction_performance(self):
        u, d = self.ui, self.predictions
        body = ("<section id='model-predictions'><h2 class='sec'>Model predictions · v11</h2>"
                "<p class='public-note'>Every winner prediction frozen at the 5 h lock, including forecast-only games. "
                "This W/L record measures game predictions; betting returns are shown separately below. "
                "Each season and stage has its own record.</p>")
        seasons = sorted({self.season} | (set(d.season.astype(int)) if len(d) else set()), reverse=True)
        for season in seasons:
            sd = d[d.season.astype(int).eq(season)] if len(d) else d
            body += f"<h3>{u.season_txt(season)} · v11 · 5 h lock</h3><div class='rule-grid prediction-grid'>"
            for stage, label in PREDICTION_STAGES:
                part = sd[sd.prediction_stage.eq(stage)] if len(sd) else sd
                if stage in ('opener', 'unknown') and not len(part):
                    continue
                r = analysis.model_prediction_summary(part)
                body += f"<article class='rule-panel'><section class='stage-panel'><h3>{label}</h3>"
                if not r['n']:
                    body += f"<div class='empty-stage'>No graded predictions yet</div><div class='empty-help'>{r['pending']} pending</div>"
                else:
                    body += (f"<span class='record'>{r['w']}–{r['l']}</span><span class='sample'>{r['n']} graded · {r['pending']} pending</span>"
                             f"<div class='stage-stats'><div>Winner accuracy<b>{u.pct(r['accuracy'])}</b></div>"
                             f"<div>95% Wilson interval<b>{u.pct(r['low'],0)}–{u.pct(r['high'],0)}</b></div></div>")
                c = r['comparison']
                if c:
                    mw, fw = (int(round(c[k]['acc']*c['n'])) for k in ('model','market'))
                    body += (f"<p class='null'>Closing favorite: {fw}–{c['n']-fw} · {u.pct(c['market']['acc'])}. "
                             f"Model on those same {c['n']} games: {mw}–{c['n']-mw}. "
                             "Games without a usable DraftKings close stay in the model record above.</p>"
                             "<table class='prediction-comparison'><caption>Probability quality · lower is better</caption>"
                             "<thead><tr><th>Score</th><th>Model at lock</th><th>Market at close</th></tr></thead><tbody>"
                             + "".join(f"<tr><td>{name}</td><td>{c['model'][key]:.4f}</td><td>{c['market'][key]:.4f}</td></tr>"
                                       for key,name in (('logloss','Log loss'),('brier','Brier score')))
                             + "</tbody></table>")
                else:
                    body += "<p class='null'>The market comparison appears when graded games have usable DraftKings closing prices.</p>"
                body += "</section></article>"
            body += "</div>"
        return body + "<p class='public-note'>Accuracy is wins divided by graded predictions. Small samples have wide uncertainty. A high win rate alone does not establish profitable betting.</p></section>"

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
        u = self.ui
        body = self.head("The forward record", "Every v11 model prediction, followed by three separate betting tests. Each season and stage stands on its own.")
        body += self.prediction_performance()
        body += "<h2 class='sec'>Tracked betting tests</h2><p class='public-note'>Three registered tests, each with its own forward record. Every qualifying pick is included, with wins, losses, pushes and pending picks.</p>"
        body += "<p class='public-note'>Returns use a flat 1 unit per pick. ROI is return divided by stakes. The market-price baseline includes the bookmaker’s margin. A small sample can move sharply.</p>"
        body += self.restart_note() + self.void_note() + self.gap_note()
        seasons = sorted({self.season}|{r['season'] for r in self.records},reverse=True)
        desc = {"H5·3": "Model margin at least 3 points from the spread.",
                "H2·F·5": "The market favorite when the model rates it at least 5 points higher.",
                "OU": "The model’s total versus the market’s, at least 5 points of probability apart."}
        for season in seasons:
            body += f"<h2 class='sec'>{u.season_txt(season)} · DraftKings · 5 h lock</h2><div class='rule-grid'>"
            for key,name,slug in RULES:
                body += f"<article class='rule-panel'><h2>{name}</h2><p>{desc[key]} <a href='how-it-works.html#{slug}'>Rule →</a></p>"
                body += self.stage(self.hypotheses[rule_key(key, True, self.ou)], season) + \
                    self.stage(self.hypotheses[rule_key(key, False, self.ou)], season) + "</article>"
            body += "</div>"
        body += "<p class='public-note'>These tests can select the same game. There is no combined portfolio return. All qualifying results remain in <a href='results.html'>Results</a>, including losses. Forecast-only games do not enter these records.</p>"
        return self.page("Performance · NBA Composite", "performance.html", body)

    def guide_page(self):
        body = self.head("How it works", "Forecasts update until a game locks. Tracked picks follow fixed rules and never change.")
        body += """<div class='guide'><section><h2>Reading a game</h2>
<p><b>Model win chance</b> is the probability that the model’s favored team wins; <b>predicted margin</b> is the expected points difference. <b>Total</b> sets the model’s expected combined score beside the DraftKings total and shows the side it leans to. Prices are DraftKings’; the small number under each moneyline is the spread. Times are Pacific.</p>
<p>While a game is in progress, <b>live win chance</b> gives each team’s chance of winning from the current score, the time left and the DraftKings spread at tip. It comes from a formula fitted on more than 12,000 past NBA games; the pregame spread keeps more weight than a lead of the same size, because leads tend to shrink. Scores and live win chance refresh every 30 seconds while a game is in progress and the page is open. It is not the model’s forecast, is not saved, and never changes a tracked pick or a record.</p>
<p>A game <b>locks</b> at the first priced snapshot within 5 hours of tip whose game data include every completed game. Before that you see a forecast, which can change. At the lock, any rule that qualifies records a <b>tracked pick</b> at its price, fixed from then on. A game with no such snapshot before tip has no tracked pick; <b>forecast only</b> means no rule qualified.</p>
<p>Game details show each team’s record and its four factors (shooting, turnovers, offensive rebounding, free throws, for and against) as plain season totals before the slate; the model itself weights recent games more. Injuries are ESPN’s pregame list, shown for context only: the model uses the NBA’s official injury report.</p></section>
<section><h2>The model</h2>
<p>One model predicts each game’s points margin and converts it into a win probability, using only information saved before tip:</p>
<ul><li><b>Team strength:</b> shooting, turnovers, offensive rebounding and free-throw rate, for and against, with recent games counting most. Last season’s ratings fade out over a team’s first 10 games.</li>
<li><b>Adjustments:</b> home court, back-to-backs, the point in the season, opponents’ three-point luck and free-throw shooting.</li>
<li><b>Roster:</b> players valued by last season’s box plus-minus, weighted by their recent minutes.</li>
<li><b>NBA injury report:</b> the minutes and value of listed players not expected to play.</li></ul>
""" + self.model_note() + """<p>Market prices are a benchmark, never an input. A team’s first game gets a forecast but no tracked pick.</p></section>
<section><h2>Every v11 model prediction has a record</h2>
<p>Results lists one winner pick per game from its five-hour v11 lock, even when no betting rule qualifies (at exactly 50%, the home team). Performance gives W/L and accuracy by season and stage with a 95% Wilson interval, beside the closing DraftKings favorite on the same games (W/L, log loss and Brier score; lower is better). Pending games, rebuilt games and earlier model versions are not counted.</p></section>
<section><h2>Three fixed tracking rules</h2>
<p>Each rule reads the probability and DraftKings prices saved at the lock and is never changed to fit later results. The minimum gaps exist because the bookmaker’s margin swallows small disagreements.</p>
<h3 id='spread'><span class='rule-number'>01</span>Spread</h3><p>Convert the win probability into a home margin with 13.5 × Φ⁻¹(P) (not the margin shown on Today). Take the model’s side when it differs from the spread by at least 3 points, with both spread prices between −135 and +115.</p>
<h3 id='favorite'><span class='rule-number'>02</span>Value favorite</h3><p>Take the market favorite when the model rates it at least 5 percentage points above the market’s no-vig probability (the market’s probability with the bookmaker’s margin removed).</p>
<h3 id='total'><span class='rule-number'>03</span>Total (over/under)</h3><p>A separate totals model predicts the combined score from both teams’ recent pace and points scored and allowed per 100 possessions, with back-to-backs""" + (", both teams’ days of rest" if self.ou == "ou3" else "") + """ and the point in the season, adjusted for players on the NBA injury report. Take its side when its chance of the over differs from the market’s by at least 5 percentage points, on a main-line DraftKings total.""" + self.ou2_clause() + """</p>
<p>Each rule keeps separate records for games 1–9 and games 10+, by the team with fewer games played; the current record, its minimum gaps and the total began with the """ + self.restart_txt() + """ games.</p>""" + self.gap_note().replace("class='public-note'", "class='note'") + """</section>
<section><h2>Reading the results fairly</h2>
<p>Each pick stakes one unit at its recorded price; a push returns the stake. ROI is profit divided by stakes. The market-price baseline is the ROI expected if the market is right, usually slightly negative because of the bookmaker’s margin; the benchmark on the same games is the favorite, or the over for totals.</p>
<p>Uncertainty is ±1 standard error, narrower than a 95% interval, and small samples prove little. Rules can pick the same game, so their returns are never combined, nor are early and established records.</p></section>"""
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
            dated = (f"<p class='note'>The current model version and its tracked record began with the {self.restart_txt()} games. "
                     "Picks recorded under earlier versions are kept separately and are not included in Results or Performance.</p>")
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
