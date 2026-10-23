"""Public presentation of native forecasts and registered first-snapshot tests.

Research renderers, reconstructed data and legacy lean records are not inputs.
All selections and grades come from analysis; this module never scores a game.
"""
from __future__ import annotations

import re
from statistics import NormalDist

import numpy as np
import pandas as pd

import analysis
import ledger
import market

PAGES = (("index.html", "Today"), ("results.html", "Results"),
         ("performance.html", "Performance"), ("how-it-works.html", "How it works"))
RULES = (("H5", "Spread", "spread"), ("H2·F", "Value favourite", "favourite"),
         ("H4", "Large probability gap", "gap"))

CSS = """
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
.forecast-readout{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:14px 0;margin-top:8px;border-top:1px solid var(--line-2)}
.forecast-readout small,.tracked-label{display:block;font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.forecast-readout strong{display:block;font:750 24px/1.35 var(--sans);letter-spacing:-.03em}.forecast-readout span{font-size:12px;color:var(--muted)}
.tracked-box{border-radius:9px;background:rgba(var(--accent),.07);padding:12px;border:1px solid rgba(var(--accent),.18)}
.tracked-box .tracked-label{color:rgb(var(--accent-tx));margin-bottom:7px}.tracked-line{font-size:12px;display:flex;justify-content:space-between;gap:8px;margin:5px 0}
.tracked-line b{font-family:var(--mono);font-size:12px;white-space:nowrap}.forecast-only{padding:10px 0 0;font-size:12px;color:var(--muted);border-top:1px solid var(--line-2)}
.forecast-only b{color:var(--ink)}.details-cue{display:flex;justify-content:space-between;margin-top:14px;padding-top:10px;border-top:1px solid var(--line-2);font-size:11px;color:var(--muted)}
.details-cue i{font-style:normal;transition:transform .15s}.tool-card[open] .details-cue i{transform:rotate(180deg)}
.tool-card .g-detail{padding:0 16px 16px;border-top:1px solid var(--line-2);font-size:13px}.tool-card .g-detail p{color:var(--muted)}
.tool-card .g-detail h3{font-size:12px;margin:16px 0 8px}.public-filters{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0}
.public-filters label{font-size:11px;color:var(--muted);font-weight:650}.public-filters select{display:block;min-height:42px;margin-top:4px;border:1px solid var(--line);border-radius:8px;padding:8px 30px 8px 10px;background:var(--surface);color:var(--ink);font:inherit;font-size:13px}
.seg{flex-wrap:wrap}.seg button{min-height:42px}.rule-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:24px 0}
.rule-panel{border:1px solid var(--line);border-radius:16px;background:var(--surface);padding:20px;min-width:0}
.rule-panel h2{font-size:18px;margin:0 0 6px;letter-spacing:-.02em}.rule-panel>p{font-size:12px;color:var(--muted);margin:0 0 18px}
.stage-panel{padding:16px 0;border-top:1px solid var(--line-2)}.stage-panel h3{font-size:12px;margin:0 0 10px;color:var(--muted)}
.stage-panel .record{font:750 28px var(--sans);letter-spacing:-.03em}.stage-panel .sample{font-size:12px;color:var(--muted);margin-left:8px}
.stage-panel .empty-stage{font-size:14px;font-weight:600;margin:6px 0}.stage-panel .empty-help{font-size:12px;color:var(--muted)}
.stage-stats{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;font-size:11px;color:var(--muted)}
.stage-stats b{display:block;font:600 16px var(--mono);color:var(--ink);margin:3px 0}.stage-panel .null{font-size:11px;color:var(--muted);margin:12px 0 0}
.stage-panel details{font-size:12px;margin-top:12px}.stage-panel summary{cursor:pointer;color:rgb(var(--accent-tx));padding:6px 0}
.stage-panel .uchart{margin-top:12px;padding:8px}.stage-panel .uchart .head{font-size:10px}.stage-panel .uchart .v{font-size:12px}
.public-table{width:100%;border-collapse:collapse;font-size:13px;background:var(--surface)}
.public-table th{text-align:left;color:var(--muted);font-size:10px;letter-spacing:.04em;text-transform:uppercase;padding:14px 12px;background:var(--surface-2)}
.public-table td{padding:14px 12px;border-top:1px solid var(--line-2)}.public-table small{display:block;color:var(--muted);font-size:11px;margin-top:3px}
.result-badge{display:inline-block;border:1px solid var(--line);border-radius:6px;padding:3px 7px;font-size:11px;font-weight:700}
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

FILTER_JS = """<script>(function(){var bar=document.querySelector('[data-filter]');if(!bar)return;bar.hidden=false;
var filter='all';function update(){var c={all:0,tracked:0,upcoming:0,live:0,final:0};
document.querySelectorAll('.gcard').forEach(function(x){var s=x.dataset.state;c.all++;c[s]++;if(x.dataset.tracked==='1')c.tracked++;
x.hidden=filter==='tracked'?x.dataset.tracked!=='1':filter!=='all'&&s!==filter;});
bar.querySelectorAll('button').forEach(function(b){var k=b.dataset.show;b.querySelector('.ct').textContent=c[k];b.setAttribute('aria-pressed',k===filter?'true':'false');});
var e=document.querySelector('[data-filter-empty]');if(e)e.hidden=!!c[filter];}
window.nbaCount=update;bar.addEventListener('click',function(e){var b=e.target.closest('button');if(b){filter=b.dataset.show;update();}});update();})();</script>"""

RESULT_FILTER_JS = """<script>(function(){var controls=document.querySelectorAll('[data-result-filter]');
controls.forEach(function(c){c.addEventListener('change',function(){var n=0;document.querySelectorAll('[data-result-row]').forEach(function(r){
var show=true;controls.forEach(function(f){if(f.value!=='all'&&r.dataset[f.dataset.resultFilter]!==f.value)show=false;});r.hidden=!show;if(show)n++;});
document.querySelector('[data-results-empty]').hidden=n>0;});});})();</script>"""


class PublicSite:
    def __init__(self, ui, native, today, built, model_ok=True):
        self.ui, self.today, self.built, self.model_ok = ui, today, built, model_ok
        self.native = native.copy() if native is not None else ledger.empty()
        if len(self.native):
            self.native = self.native[self.native["basis"].eq("native")]
        self.season = ui.season_for(today)
        self.hypotheses = {h["key"]: h for h in analysis.active_hypotheses()}
        self.records = []
        src = analysis.on_book(self.native, "first", analysis.HYPOTHESIS_BOOK)
        done = ledger.graded(src)
        by_id = self.native.set_index("game_id", drop=False)
        for key, name, slug in RULES:
            for suffix in ("·E", ""):
                hyp = self.hypotheses[key + suffix]
                signals = analysis.hypothesis_signals(src, hyp, hyp["native"])
                grades = analysis.hypothesis_picks(done, hyp, hyp["native"])
                grades = {str(r.game_id): r for _, r in grades.iterrows()} if len(grades) else {}
                for _, s in signals.iterrows():
                    row = by_id.loc[s.game_id]
                    grade = grades.get(str(s.game_id))
                    self.records.append(dict(hyp=hyp, name=name, slug=slug, signal=s,
                        row=row, grade=grade, season=int(row.season), early=hyp["early"]))

    def page(self, title, active, body):
        return self.ui.page(title, active, body, self.built, extra_css=CSS)

    def head(self, title, desc, update=False):
        u = self.ui
        return (f"<div class='public-head'><div><div class='eyebrow'>{u.season_txt(self.season)} · NBA forecasts</div>"
                f"<h1>{u.esc(title)}</h1><p>{desc}</p></div>"
                + (f"<div class='update'>Forecast updated<time>{u.esc(self.built)}</time></div>" if update else "") + "</div>")

    def selection(self, rec):
        u, s = self.ui, rec["signal"]
        line = f" {u.line_txt(s.line)}" if rec["hyp"].get("market") == "ats" else ""
        return f"{u.esc(s.side)}{line} <span class='mono'>{u.ml_txt(s.ml)}</span>"

    def card(self, row):
        u = self.ui
        number = lambda k: pd.to_numeric(row.get(k), errors="coerce")
        p = number("p_home")
        dk = str(row.get("pre_book")) == "dk"
        recs = [r for r in self.records if str(r["row"].game_id) == str(row.game_id)]
        final = np.isfinite(number("home_won"))
        status = "Final" if final else "Pregame"
        teams = []
        for side in ("away", "home"):
            team = row[side]
            pts = number(f"{side}_pts")
            line = number("pre_spread") * (-1 if side == "away" else 1) if dk else np.nan
            teams.append(f"<div class='pub-team'><div class='tchip'><span>{u.esc(team)}</span>{u.logo_html(team)}</div>"
                f"<div class='tn'>{u.esc(u.TEAM_NAMES.get(team,team))}<small>{'@ ' if side == 'home' else ''}{u.esc(team)}</small></div>"
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
            tracking = "<div class='tracked-box'><span class='tracked-label'>Tracked picks · first snapshot</span>" + "".join(
                f"<div class='tracked-line'><span>{r['name']}</span><b>{self.selection(r)}</b></div>" for r in recs) + "</div>"
        else:
            tag = str(row.get("first_model_tag", ""))
            gp = np.fmin(number("gp_home"), number("gp_away"))
            why = ("Season openers are outside the tracked rules." if gp == 0 else
                   "The early-game first snapshot used an earlier model." if gp < 10 and tag and tag != "nan" and not tag.endswith("_v9") else
                   "Waiting for a priced first snapshot." if pd.isna(row.get("first_p_home")) else
                   "No tracked rule qualified at the first snapshot.")
            tracking = f"<div class='forecast-only'><b>Forecast only</b><br>{why}</div>"
        book = "DraftKings" if dk else "DraftKings unavailable; no market"
        q = number("pre_q_home") if dk else np.nan
        detail = (f"<h3>Forecast and market</h3><p>{u.esc(row.away)} @ {u.esc(row.home)} · {u.esc(book)} prices. "
                  f"Market home win probability: {u.pct(q)} with the bookmaker margin removed. "
                  "The forecast above is the latest saved pregame view. Prices and forecasts freeze at tip.</p>")
        if recs:
            detail += (f"<h3>Recorded before play</h3><p>Tracked at {u.tip_pt(row.get('first_snapshot_utc'))}. "
                       "The selections and prices above stay fixed even if the forecast or market changes later. "
                       "A game can qualify for multiple tests; each keeps its own record.</p>")
        if str(row.get("model_tag", "")) in u.AVAIL_TAGS:
            detail += "<p>The model includes the NBA injury report available when this forecast was saved.</p>"
        detail += "<p><a href='how-it-works.html'>How forecasts and tracked picks work →</a></p>"
        return (f"<details class='gcard tool-card' data-gid='{u.esc(row.game_id)}' data-tracked='{int(bool(recs))}' data-state='{'final' if final else 'upcoming'}'>"
                f"<summary><div class='g-meta'><span>{u.tip_pt(row.tip_utc)}</span><span class='badge st'>{status}</span></div>"
                + "".join(teams) + forecast + tracking
                + "<span class='details-cue'>Game details<i aria-hidden='true'>⌄</i></span></summary>"
                + f"<div class='g-detail'>{detail}</div></details>")

    def today_page(self):
        u = self.ui
        day = self.native[self.native.slate_date.astype(str).eq(self.today)]
        title = pd.Timestamp(self.today).strftime("%A, %B %-d")
        body = self.head(title, "Today’s model forecasts. Tracked picks are marked separately and locked at the first snapshot.", True)
        body += "<div class='public-intro'><p>DraftKings prices · All times Pacific (PT)<span data-scores-asof hidden> · Scores as of <time></time></span></p><a href='how-it-works.html'>How to read →</a></div>"
        if not len(day):
            body += "<div class='public-empty'><h2>No games to show</h2><p>There are no saved regular-season forecasts for this slate. Check back for the next update.</p></div>"
        else:
            body += "<div class='seg' data-filter role='group' aria-label='Filter games' hidden>" + "".join(
                f"<button type='button' data-show='{key}' aria-pressed='{'true' if key == 'all' else 'false'}'>{name}<span class='ct'>0</span></button>"
                for key,name in (("all","All"),("tracked","Tracked"),("upcoming","Upcoming"),("live","Live"),("final","Final"))) + "</div>"
            body += f"<div class='tool-grid' data-slate='{u.esc(self.today)}'>" + "".join(self.card(r) for _,r in day.sort_values("tip_utc").iterrows()) + "</div>"
            body += "<div class='public-empty' data-filter-empty hidden><p>No games match this filter.</p></div>" + FILTER_JS + u.SCORES_JS
        return self.page("Today · NBA Composite", "index.html", body)

    def results_page(self):
        u = self.ui
        body = self.head("Tracked results", "Every qualifying selection at its original price. Wins, losses and pending picks stay in the record.")
        body += "<p class='public-note'>Three separate tests. A game can appear in more than one; their returns are never added together. Early games (1–9) and established games (10+) remain separate.</p>"
        if not self.records:
            body += "<div class='public-empty'><h2>The forward record starts here</h2><p>No qualifying first-snapshot picks have been recorded yet. Today’s forecasts remain available while the tests build their records.</p></div><p><a href='index.html'>View today’s forecasts →</a></p>"
        else:
            seasons = sorted({r['season'] for r in self.records}, reverse=True)
            options = (("rule","Test",[("all","All tests")]+[(s,n) for _,n,s in RULES]),
                       ("stage","Stage",[("all","Both stages"),("early","Early · games 1–9"),("established","Established · games 10+")]),
                       ("season","Season",[("all","All seasons")]+[(str(s),u.season_txt(s)) for s in seasons]))
            body += "<div class='public-filters'>" + "".join(f"<label>{name}<select data-result-filter='{key}'>"+"".join(f"<option value='{v}'>{label}</option>" for v,label in opts)+"</select></label>" for key,name,opts in options) + "</div>"
            labels = ["Game", "Test / stage", "Recorded pick", "First snapshot", "Outcome", "Return"]
            rows = []
            for rec in sorted(self.records, key=lambda r:(str(r['row'].tip_utc),r['slug']), reverse=True):
                row, grade = rec['row'], rec['grade']
                if grade is None:
                    result, ret = "Pending", "—"
                else:
                    win = grade['result'] if rec['hyp'].get('market') == 'ats' else grade['won']
                    result = "Push" if win == .5 else "Win" if win == 1 else "Loss"
                    ret = f"{grade['units']:+.2f}u"
                stage = "Early · games 1–9" if rec['early'] else "Established · games 10+"
                cells = [f"{u.esc(row.away)} @ {u.esc(row.home)}<small>{u.esc(row.slate_date)} · {u.tip_pt(row.tip_utc)}</small>",
                         f"{rec['name']}<small>{stage} · {u.season_txt(rec['season'])}</small>", self.selection(rec),
                         f"{u.tip_pt(row.first_snapshot_utc)}<small>{u.esc(str(row.first_snapshot_utc)[:10])}</small>",
                         f"<span class='result-badge'>{result}</span>", ret]
                rows.append(f"<tr data-result-row data-rule='{rec['slug']}' data-stage='{'early' if rec['early'] else 'established'}' data-season='{rec['season']}'>"+"".join(f"<td data-label='{l}'>{v}</td>" for l,v in zip(labels,cells))+"</tr>")
            body += "<table class='public-table'><thead><tr>"+"".join(f"<th>{l}</th>" for l in labels)+"</tr></thead><tbody>"+"".join(rows)+"</tbody></table>"
            body += "<div class='public-empty' data-results-empty hidden><p>No picks match these filters.</p></div>"+RESULT_FILTER_JS
        return self.page("Results · NBA Composite", "results.html", body)

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
        r = analysis.ats_row(hyp['key'], d) if hyp.get('market') == 'ats' else analysis.roi_row(hyp['key'],d)
        pushes = r.get('push',0)
        record = f"{r['w']}–{r['l']}" + (f"–{pushes}" if pushes else "")
        models = sorted({m.group(0) for rec in records for m in [re.search(r'v\d+$',str(rec['row'].first_model_tag))] if m})
        z = analysis.z_vs_null(r)
        c = u.sig_cls(z)
        chart_records = sorted((rec for rec in records if rec['grade'] is not None),key=lambda rec:str(rec['row'].tip_utc))
        chart = u.units_svg([rec['grade']['units'] for rec in chart_records], "Cumulative return",width=560)
        return (start+f"<span class='record'>{record}</span><span class='sample'>{r['n']} graded · {pending} pending</span>"
                f"<div class='stage-stats'><div>Return<b>{r['units']:+.2f}u</b></div><div>ROI<b class='{c}'>{100*r['roi']:+.1f}%</b></div></div>"
                f"<p class='null'>ROI uncertainty ±{u.se_txt(r['roi_se'])} pp (1 SE). Market-price baseline {100*r['roi_null']:+.1f}%.</p>"
                f"<details><summary>Benchmarks and return chart</summary><p>Flat 1 unit per pick. Same-game market favourite ROI {u.fav_txt(r)}. "
                f"Model at first snapshot: {u.esc(' / '.join(models) or 'recorded model')}. "
                "Colour marks a gap of at least two standard errors from the market-price baseline.</p>"+chart+"</details></section>")

    def performance_page(self):
        u = self.ui
        body = self.head("The forward record", "Three registered tests, tracked from prices saved before play. Each test and season stage stands on its own.")
        body += "<p class='public-note'>Returns use a flat 1 unit per pick. ROI is return divided by stakes. The market-price baseline includes the bookmaker’s margin. A small sample can move sharply.</p>"
        seasons = sorted({self.season}|{r['season'] for r in self.records},reverse=True)
        for season in seasons:
            body += f"<h2 class='sec'>{u.season_txt(season)} · DraftKings · first snapshot</h2><div class='rule-grid'>"
            for key,name,slug in RULES:
                desc = {"H5":"Model margin versus the spread.","H2·F":"The value side when the market makes it favourite.","H4":"A model–market probability gap of at least 12 points."}[key]
                body += f"<article class='rule-panel'><h2>{name}</h2><p>{desc} <a href='how-it-works.html#{slug}'>Rule →</a></p>"
                body += self.stage(self.hypotheses[key+'·E'],season)+self.stage(self.hypotheses[key],season)+"</article>"
            body += "</div>"
        body += "<p class='public-note'>These tests can select the same game. There is no combined portfolio return. All qualifying results remain in <a href='results.html'>Results</a>, including losses. Forecast-only games do not enter these records.</p>"
        return self.page("Performance · NBA Composite", "performance.html", body)

    def guide_page(self):
        body = self.head("How it works", "A forecast describes the game. A tracked pick follows a fixed rule. Their timing and records are different.")
        body += """<div class='guide'><section><h2>Read a game in seconds</h2>
<p><b>Model win chance</b> is the model’s probability for its favoured team. <b>Predicted margin</b> is the expected points difference. Neither guarantees the outcome.</p>
<p>The prices beside the teams come from DraftKings. The small number below each moneyline is that team’s spread. All times are Pacific (PT).</p>
<p><b>Tracked pick</b> means a registered rule qualified at the first priced snapshot. <b>Forecast only</b> means the game has no qualifying tracked selection. The latest forecast can change before tip; a tracked selection and its recorded price cannot.</p></section>
<section><h2>One model, before the game</h2><p>The model combines prior-game team performance, roster talent, schedule and available NBA injury-report information. It predicts a points margin and converts it to a win probability. Market prices are a comparison, not a model input.</p>
<p>Only information saved before tip is used for a live prediction. Each team’s first game has a forecast but sits outside the tracked rules. Early-season forecasts have less current-season information, and injury-report timing can affect them.</p></section>
<section><h2>Three fixed tracking rules</h2>
<p>Rules use the probability and DraftKings prices saved at the first snapshot, which can be later than the market opening price. No rule is adjusted to fit later outcomes.</p>
<h3 id='spread'><span class='rule-number'>01</span>Spread</h3><p>Convert the first win probability to a home margin using 13.5 × Φ⁻¹(P). Track the model’s side when it differs from the market spread by at least 1 point, with both spread prices between −135 and +115. This fixed tracking conversion can differ from the model margin displayed on Today.</p>
<h3 id='favourite'><span class='rule-number'>02</span>Value favourite</h3><p>The value side is the team whose model probability exceeds the market probability after removing the bookmaker’s margin. Track it when the market also makes that side the favourite: market probability above 50%.</p>
<h3 id='gap'><span class='rule-number'>03</span>Large probability gap</h3><p>Track the value side when its model probability exceeds the market probability by at least 12 percentage points.</p>
<p>All three rules have separate early (games 1–9) and established (games 10+) records. Stage follows the team with fewer games played. Early selections require a v9 first snapshot; older saved predictions stay historical. These registrations began at different dates; the early segments were registered on October 23, 2026.</p></section>
<section><h2>Read the results fairly</h2><p>Each result uses the selection and price recorded before play. Wins, losses and pushes stay visible. Pending picks have no return until graded. A push returns the stake and produces zero profit.</p>
<p>One unit is staked on each selection. ROI is profit divided by total stakes. The market-price baseline is the expected ROI if the market probability is correct; it is usually negative because of the bookmaker’s margin. The market favourite benchmark uses the same games and prices.</p>
<p>Uncertainty is shown as one standard error (1 SE). One SE is not a 95% confidence interval. Small samples do not establish a lasting advantage. Tests may overlap, so their returns and the early/established stages are never pooled.</p>
<p>Public performance reports live, pregame-recorded selections. Rebuilt games, exploratory comparisons and historical model experiments are excluded from the public track record.</p></section></div>"""
        return self.page("How it works · NBA Composite", "how-it-works.html", body)

    def render(self):
        return {"index.html":self.today_page(),"results.html":self.results_page(),
                "performance.html":self.performance_page(),"how-it-works.html":self.guide_page()}
