"""Live win chance: an in-game home win probability for Today's live games.

Display only (owner request, 2026-11-07; CLAUDE.md "Live win chance"). It is
computed in the reader's browser from ESPN's live score and clock, the same
public scoreboard the Today page already fetches, and the last DraftKings
spread saved before tip (`pre_spread`; watch_slate.py dispatches a build
about 20 minutes before each tip to refresh it). Nothing here is written to data/, read
by the prediction model, graded, or used by a tracked test or record.

  P(home wins) = Phi( a(tau) * lead / sqrt(tau) + b(tau) * mu * sqrt(tau) )

lead = home - away points now; mu = the pregame expected home margin
(-pre_spread); tau = regulation time left / 48 minutes (in overtime, the
period's time left / 48 minutes). a and b are piecewise linear in tau on the
fit's knots. With a = b = 1/sigma this is the Brownian margin of
research/strength_vs_reaction.py; fitting them apart lets the pregame line
keep more weight than a lead of the same size, since NBA leads partly fade.
The coefficients are fitted on historical play-by-play by
research/live_wp.py and stored in model/live_wp.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PATH = Path(__file__).resolve().parent / "model" / "live_wp.json"
REG_SECONDS = 4 * 12 * 60          # regulation, seconds
QUARTER_SECONDS = 12 * 60
OT_SECONDS = 5 * 60
TAU_FLOOR = 1.0 / REG_SECONDS      # one second: never divide by zero


def load(path=PATH):
    """The fitted coefficients, or None when the file is missing or unreadable
    (the page then shows no live win chance)."""
    try:
        fit = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not _valid(fit):
        return None
    return fit


def _valid(fit):
    try:
        k, a, b = fit["knots"], fit["a"], fit["b"]
    except (TypeError, KeyError):
        return False
    return (isinstance(k, list) and len(k) >= 2 and len(k) == len(a) == len(b)
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in k + a + b)
            and all(k[i] < k[i + 1] for i in range(len(k) - 1)))


def tau(period, clock):
    """Share of regulation left: (4 - period) quarters plus the clock in
    regulation; the overtime period's clock in overtime. Both / 48 minutes."""
    period, clock = int(period), max(float(clock), 0.0)
    if period <= 4:
        return ((4 - period) * QUARTER_SECONDS + min(clock, QUARTER_SECONDS)) / REG_SECONDS
    return min(clock, OT_SECONDS) / REG_SECONDS


def interp(knots, values, t):
    """Piecewise-linear value at t, flat beyond the end knots."""
    if t <= knots[0]:
        return values[0]
    for i in range(1, len(knots)):
        if t <= knots[i]:
            w = (t - knots[i - 1]) / (knots[i] - knots[i - 1])
            return values[i - 1] + w * (values[i] - values[i - 1])
    return values[-1]


def Phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def win_prob(lead, mu, t, fit):
    """P(home wins) with home lead `lead`, pregame expected home margin `mu`
    and `t` = share of regulation left (tau). At no time left the leader has
    won; a tie there goes to overtime and is priced at overtime's start."""
    if t <= 0:
        if lead:
            return 1.0 if lead > 0 else 0.0
        t = OT_SECONDS / REG_SECONDS
    t = max(t, TAU_FLOOR)
    a = interp(fit["knots"], fit["a"], t)
    b = interp(fit["knots"], fit["b"], t)
    return Phi(a * lead / math.sqrt(t) + b * mu * math.sqrt(t))


def script(fit):
    """The browser's copy of win_prob, as window.nbaLive(card, type, status,
    scores), called by the live-score overlay (build_site.SCORES_JS) for each
    card it updates. It fills the card's [data-live-wp] block while a game is
    in progress and hides it otherwise. Empty without a fit."""
    if not fit:
        return ""
    coef = json.dumps({"k": fit["knots"], "a": fit["a"], "b": fit["b"]}, separators=(",", ":"))
    return ("<script>(function(){var F=" + coef + ";" + _JS + "})();</script>")


# Phi via Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7); the rest mirrors
# win_prob/tau/interp above line for line (tests/test_live_wp.py runs both).
_JS = """
function ip(v,t){var k=F.k;if(t<=k[0])return v[0];for(var i=1;i<k.length;i++)if(t<=k[i])
return v[i-1]+(t-k[i-1])/(k[i]-k[i-1])*(v[i]-v[i-1]);return v[v.length-1];}
function Phi(z){var x=Math.abs(z)/Math.SQRT2,t=1/(1+0.3275911*x),
y=1-((((1.061405429*t-1.453152027)*t+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);
return z>=0?(1+y)/2:(1-y)/2;}
function tau(p,c){c=Math.max(+c||0,0);return p<=4?((4-p)*720+Math.min(c,720))/2880:Math.min(c,300)/2880;}
function wp(lead,mu,t){if(t<=0){if(lead)return lead>0?1:0;t=300/2880;}t=Math.max(t,1/2880);
return Phi(ip(F.a,t)*lead/Math.sqrt(t)+ip(F.b,t)*mu*Math.sqrt(t));}
function clk(st){if(typeof st.clock==='number')return st.clock;var d=String(st.displayClock||'0'),m=d.split(':');
return m.length>1?(+m[0])*60+(+m[1]):+d||0;}
function fmt(p){return p>0.995?'>99%':p<0.005?'<1%':Math.round(100*p)+'%';}
window.nbaLiveWP=wp;window.nbaLiveTau=tau;
window.nbaLive=function(c,ty,st,s){var box=c.querySelector('[data-live-wp]');if(!box)return;
var sp=parseFloat(c.getAttribute('data-live-spread')),h=parseInt(s.home,10),a=parseInt(s.away,10),p=+st.period;
if(ty.state!=='in'||!isFinite(sp)||!isFinite(h)||!isFinite(a)||!(p>=1)){box.hidden=true;return;}
var ph=wp(h-a,-sp,tau(p,clk(st)));
box.querySelector('[data-wp=home]').textContent=fmt(ph);box.querySelector('[data-wp=away]').textContent=fmt(1-ph);
box.querySelector('.lwp-bar i').style.width=(100*(1-ph)).toFixed(1)+'%';box.hidden=false;};
"""
