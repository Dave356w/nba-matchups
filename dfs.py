"""DFS lineups page (CLAUDE.md, DFS lineup page amendment, 2026-10-29).

A Yahoo daily-fantasy lineup optimizer for the day's slate, a separate tool:
it reads no model probability, tracked pick or record and feeds nothing back.

  * Salaries, positions and the cap: Yahoo's public NBA players feed
    (YAHOO_FEED), snapshot to data/nba_yahoo_dfs.csv by the build. A row is
    replaced only before its game starts and frozen after, like the other
    pregame snapshots.
  * Projections: Sleeper's latest pregame line for the player
    (data/nba_sleeper.csv, kind "pre"; "first" if no "pre"), in Yahoo scoring
    (yahoo_points). Listed at 0 minutes, Out or IR = 0; no Sleeper match = none.
  * The page embeds the slate as JSON; the visitor's browser finds the best
    lineups exactly (SOLVER_JS: a knapsack per position, combined by
    salary, and Lawler's k-best partition for the next-best lineups) under
    Yahoo's classic rules: PG, SG, G, SF, PF, F, C, UTIL, the cap, at least
    MIN_TEAMS teams and at most MAX_PER_TEAM from one.

A Yahoo or Sleeper failure never stops a build; the page then says what is
missing.
"""
from __future__ import annotations

import html
import json
import os
import re
import tempfile
import time
import unicodedata
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import ledger
import market

YAHOO_FEED = "https://dfyql-ro.sports.yahoo.com/v2/external/playersFeed/nba"
YAHOO_PATH = os.path.join("data", "nba_yahoo_dfs.csv")
YAHOO_COLUMNS = ["slate_date", "snapshot_utc", "player_code", "game_code", "name", "team",
                 "position", "salary", "fppg", "home", "away", "start_utc", "cap"]
POSITIONS = ("PG", "SG", "SF", "PF", "C")
SLOTS = ("PG", "SG", "G", "SF", "PF", "F", "C", "UTIL")
# Yahoo's multi-game contest rules (legal.yahoo.com, daily fantasy basketball).
MIN_TEAMS = 3
MAX_PER_TEAM = 6
# Yahoo NBA scoring: reproduces Yahoo's season FPPG exactly for 55 of 74
# players on 2026-10-29 (the rest within a stat correction); no 3PM bonus.
SCORING = {"pts": 1.0, "reb": 1.2, "ast": 1.5, "stl": 3.0, "blk": 3.0, "to": -1.0}
# Yahoo codes that differ from Basketball-Reference's, beyond ESPN's.
YAHOO2BBR = {**market.ESPN2BBR, "PHO": "PHO", "UTH": "UTA", "WAS": "WAS"}
ET = ZoneInfo("America/New_York")
# Sleeper statuses that project 0 (GTD / DTD keep their projection).
OUT_STATUSES = ("out", "ir")


def yahoo_points(row):
    """Yahoo fantasy points of one Sleeper projected line (NaN if no line)."""
    vals = [pd.to_numeric(row.get(k), errors="coerce") for k in SCORING]
    if any(not np.isfinite(v) for v in vals):
        return float("nan")
    return float(sum(w * v for w, v in zip(SCORING.values(), vals)))


def norm_name(name):
    """A name for matching across feeds: no accents, case, punctuation or
    generational suffix ('Darius Acuff Jr.' == 'darius acuff')."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    words = [w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(words)


def team_code(yahoo_team):
    t = str(yahoo_team).upper()
    return YAHOO2BBR.get(t, t)


# ------------------------------------------------------------- Yahoo feed ---
def fetch_feed(url=YAHOO_FEED, timeout=20, attempts=3):
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 nba-matchups"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - retried, then raised
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def parse_feed(js, now=None):
    """Yahoo's players feed -> snapshot rows (YAHOO_COLUMNS), one per player,
    with the multi-game cap and each game's start in UTC."""
    now = now or ledger.utc_now()
    players = js.get("players") or {}
    players = players.get("result", []) if isinstance(players, dict) else players
    caps = (js.get("salaryCapInfo") or {}).get("result") or [{}]
    cap = caps[0].get("multiGameSalaryCap") if caps else None
    rows = []
    for p in players:
        try:
            start = parsedate_to_datetime(p["gameStartTime"]).astimezone(timezone.utc)
            salary = int(p["salary"])
        except (KeyError, TypeError, ValueError):
            continue
        pos = str(p.get("position", "")).upper()
        if pos not in POSITIONS or salary <= 0:
            continue
        rows.append(dict(
            slate_date=start.astimezone(ET).strftime("%Y-%m-%d"),
            snapshot_utc=ledger.fmt_utc(now), player_code=str(p.get("playerCode", "")),
            game_code=str(p.get("gameCode", "")), name=str(p.get("name", "")),
            team=team_code(p.get("team")), position=pos, salary=salary,
            fppg=p.get("fppg", ""), home=team_code(p.get("homeTeam")),
            away=team_code(p.get("awayTeam")), start_utc=ledger.fmt_utc(start),
            cap="" if cap is None else cap))
    return rows


def load(path=YAHOO_PATH):
    if not os.path.exists(path):
        return pd.DataFrame(columns=YAHOO_COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)[YAHOO_COLUMNS]


def save(df, path=YAHOO_PATH):
    """Atomic write, sorted by slate, start, game, salary (high first) and name."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=os.path.dirname(path) or ".")
    os.close(fd)
    df = df[YAHOO_COLUMNS].assign(_s=pd.to_numeric(df["salary"], errors="coerce"))
    df = df.sort_values(["slate_date", "start_utc", "game_code", "_s", "name"],
                        ascending=[True, True, True, False, True], kind="stable")
    df.drop(columns="_s").to_csv(tmp, index=False)
    os.replace(tmp, path)


def upsert(df, rows, now=None):
    """Replace each fetched player's row while his game has not started;
    rows of started games stay frozen. Returns (frame, written, frozen)."""
    now = now or ledger.utc_now()
    df = df.copy()
    keys = df["slate_date"].astype(str) + "|" + df["player_code"].astype(str)
    frozen = set(keys[[(ledger.parse_utc(s) or now) <= now for s in df["start_utc"]]])
    new = [r for r in rows if f"{r['slate_date']}|{r['player_code']}" not in frozen
           and (ledger.parse_utc(r["start_utc"]) or now) > now]
    drop = keys.isin({f"{r['slate_date']}|{r['player_code']}" for r in new})
    out = pd.concat([d for d in (df[~drop], pd.DataFrame(new, columns=YAHOO_COLUMNS))
                     if len(d)], ignore_index=True) if len(new) or len(df) else df
    return (out[YAHOO_COLUMNS] if len(out) else pd.DataFrame(columns=YAHOO_COLUMNS),
            len(new), len(rows) - len(new))


def snapshot(log=print, fetch=fetch_feed, now=None):
    """The build's step: fetch Yahoo's feed and record it. Never raises."""
    try:
        rows = parse_feed(fetch(), now)
        df, n, k = upsert(load(), rows, now)
        save(df)
        log(f"yahoo dfs: {n} salaries written, {k} frozen or started")
    except Exception as e:  # noqa: BLE001 - a separate tool never blocks the build
        log(f"yahoo dfs unavailable ({e!r})")


# --------------------------------------------------------------- the slate ---
def slate(today, yahoo=None, sleeper=None, native=None, now=None):
    """The page's data for ET date `today`: dict(cap, games, players, ...) or
    None when Yahoo has no salaries for it. Each player carries his Yahoo
    salary and position and his Sleeper projection in Yahoo points."""
    now = now or ledger.utc_now()
    yahoo = load() if yahoo is None else yahoo
    y = yahoo[yahoo["slate_date"].astype(str) == str(today)]
    if not len(y):
        return None
    sleeper = ledger.load_sleeper() if sleeper is None else sleeper
    native = ledger.load(ledger.NATIVE_PATH) if native is None else native
    games = native[native["slate_date"].astype(str) == str(today)] if len(native) else native
    gid = {(str(r.home), str(r.away)): str(r.game_id) for r in games.itertuples()}
    s = sleeper[sleeper["game_id"].astype(str).isin(set(gid.values()))].copy()
    # the latest snapshot per game: "pre", else "first"
    s["kind_order"] = s["kind"].map({"pre": 0, "first": 1})
    s = s.sort_values("kind_order", kind="stable").drop_duplicates(
        ["game_id", "team", "sleeper_id", "name"])
    s["match_name"] = s["name"].map(norm_name)
    by_team = {(str(r.game_id), str(r.team), r.match_name): r for r in s.itertuples()}
    players, unmatched = [], []
    for r in y.itertuples():
        g = gid.get((str(r.home), str(r.away)))
        hit = by_team.get((g, str(r.team), norm_name(r.name))) if g else None
        proj, minutes, inj = None, None, ""
        if hit is not None:
            line = hit._asdict()
            minutes = pd.to_numeric(line.get("minutes"), errors="coerce")
            inj = str(line.get("injury_status") or "")
            pts = yahoo_points(line)
            if np.isfinite(pts):
                proj = 0.0 if (minutes == 0 or inj.lower() in OUT_STATUSES) else round(pts, 2)
        else:
            unmatched.append(r.name)
        opp = r.away if r.team == r.home else r.home
        players.append(dict(id=r.player_code, name=r.name, team=r.team, opp=opp,
                            home=r.team == r.home, pos=r.position, sal=int(r.salary),
                            proj=proj, fppg=_num(r.fppg),
                            min=None if minutes is None or not np.isfinite(minutes) else float(minutes),
                            inj=inj, start=r.start_utc, game=r.game_code))
    snaps = s["snapshot_utc"].astype(str)
    games_out = sorted({(r.game_code, r.away, r.home, r.start_utc) for r in y.itertuples()},
                       key=lambda t: (t[3], t[0]))
    cap = _num(y["cap"].iloc[0])
    if cap is None:
        return None
    return dict(date=str(today), cap=cap, slots=list(SLOTS), min_teams=MIN_TEAMS,
                max_per_team=MAX_PER_TEAM, salaries_utc=max(y["snapshot_utc"]),
                sleeper_utc=max(snaps) if len(snaps) else None,
                games=[dict(code=c, away=a, home=h, start=t) for c, a, h, t in games_out],
                players=players, unmatched=unmatched, scoring=SCORING,
                built_utc=ledger.fmt_utc(now))


def _num(x):
    v = pd.to_numeric(x, errors="coerce")
    return None if not np.isfinite(v) else float(v)


# ------------------------------------------------------------------- page ---
CSS = """
.dfs-status{display:flex;flex-wrap:wrap;gap:8px 18px;padding:12px 14px;border:1px solid var(--line);border-radius:12px;background:var(--surface);font-size:13px;color:var(--muted);margin:0 0 18px}
.dfs-status b{color:var(--ink);font-family:var(--mono);font-weight:600}
.dfs-grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:18px;align-items:start}
.dfs-panel{border:1px solid var(--line);border-radius:14px;background:var(--surface);padding:16px;min-width:0}
.dfs-panel h2{font-size:16px;margin:0 0 10px;letter-spacing:-.01em;text-transform:none}
.dfs-controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin-bottom:12px}
.dfs-controls label{font-size:11px;color:var(--muted);font-weight:650}
.dfs-controls input,.dfs-controls select{display:block;margin-top:4px;min-height:38px;width:96px;border:1px solid var(--line);border-radius:8px;padding:6px 8px;background:var(--surface);color:var(--ink);font:inherit;font-size:13px}
.dfs-btn{min-height:38px;padding:0 14px;border-radius:9px;border:1px solid var(--line);background:var(--surface);color:var(--ink);font:650 13px var(--sans);cursor:pointer}
.dfs-btn.primary{background:rgb(var(--accent));border-color:rgb(var(--accent));color:var(--accent-on)}
.dfs-btn:disabled{opacity:.55;cursor:wait}
.dfs-pos{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 10px}
.dfs-pos button{min-height:32px;padding:0 10px;border:1px solid var(--line);border-radius:999px;background:var(--surface);color:var(--muted);font:650 12px var(--sans);cursor:pointer}
.dfs-pos button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.dfs-scroll{overflow:auto;max-height:640px;border:1px solid var(--line-2);border-radius:10px}
.dfs-table{width:100%;border-collapse:collapse;font-size:12.5px}
.dfs-table th{position:sticky;top:0;background:var(--surface-2);text-align:left;font-size:10px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);padding:8px;z-index:1;white-space:nowrap}
.dfs-table th,.dfs-table td{text-align:left;font-family:var(--sans);border-bottom:0}
.dfs-table td{padding:6px 8px;border-top:1px solid var(--line-2);white-space:nowrap;font-size:12.5px}
.dfs-table td.n{font-family:var(--mono);text-align:right}.dfs-table th.n{text-align:right}
.dfs-table small{display:block;color:var(--muted);font-size:10.5px}
.dfs-table input[type=number]{width:64px;min-height:30px;border:1px solid var(--line);border-radius:6px;padding:3px 5px;background:var(--surface);color:var(--ink);font:12px var(--mono);text-align:right}
.dfs-table tr.locked td{background:rgba(var(--accent),.08)}.dfs-table tr.out td{opacity:.45}
.dfs-inj{color:rgb(var(--bad));font-weight:700;font-size:10.5px;margin-left:4px}
.dfs-lineup{border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin:0 0 10px}
.dfs-lineup header{display:flex;justify-content:space-between;gap:8px;font-size:12px;color:var(--muted);margin-bottom:6px}
.dfs-lineup header b{color:var(--ink);font-family:var(--mono)}
.dfs-lineup ol{list-style:none;margin:0;padding:0;font-size:12.5px}
.dfs-lineup li{display:grid;grid-template-columns:44px minmax(0,1fr) 44px 52px;gap:6px;padding:3px 0;border-top:1px solid var(--line-2)}
.dfs-lineup li span:first-child{font:600 11px var(--mono);color:var(--muted)}
.dfs-lineup li .n{text-align:right;font-family:var(--mono)}
.dfs-lineup li em{font-style:normal;color:var(--muted);font-size:11px}
.dfs-msg{font-size:13px;color:var(--muted);margin:8px 0}
.dfs-exp{font-size:12px;columns:2;column-gap:16px;margin:8px 0 0;padding:0;list-style:none}
.dfs-exp li{display:flex;justify-content:space-between;gap:8px;break-inside:avoid;padding:2px 0}
@media(max-width:900px){.dfs-grid{grid-template-columns:1fr}.dfs-scroll{max-height:520px}.dfs-grid>.dfs-out{order:-1}}
.dfs-table .mpos{display:none}
@media(max-width:640px){.dfs-controls input,.dfs-controls select{width:84px}.dfs-exp{columns:1}.dfs-table .x{display:none}
.dfs-table th,.dfs-table td{padding:6px 4px}.dfs-table td.nm{white-space:normal;min-width:96px}.dfs-table .mpos{display:inline}
.dfs-table input[type=number]{width:56px}}
"""


def _esc(s):
    return html.escape(str(s), quote=True)


def body(data, ui):
    """The page body: header, notes and the optimizer's mount points; the
    pool and lineups are drawn by UI_JS (SOLVER_JS finds them) from the
    embedded JSON."""
    head = ("<div class='public-head'><div><div class='eyebrow'>Yahoo daily fantasy · NBA</div>"
            "<h1>DFS lineups</h1><p>Build Yahoo classic lineups for today’s slate from Sleeper’s "
            "projections and Yahoo’s salaries. The best lineups are found on your device; lock, "
            "exclude or re-project any player.</p></div></div>")
    note = ("<p class='public-note'>A separate tool: these projections are Sleeper’s (RotoWire) stat "
            "projections in Yahoo scoring (points 1, rebounds 1.2, assists 1.5, steals 3, blocks 3, "
            "turnovers −1), not this site’s model, and nothing here is a tracked pick. Players "
            "listed at 0 minutes or Out project 0. Check each contest’s rules and lock before "
            "entering. Not betting advice.</p>")
    if not data:
        return (head + "<div class='public-empty'><h2>No Yahoo slate today</h2><p>Yahoo has not "
                "published salaries for today’s games yet, or they could not be fetched. The page "
                "updates with each build.</p></div>" + note)
    teams = len({p["team"] for p in data["players"]})
    status = ("<div class='dfs-status'>"
              f"<span>Slate <b>{_esc(data['date'])}</b></span>"
              f"<span>Games <b>{len(data['games'])}</b> · teams <b>{teams}</b></span>"
              f"<span>Cap <b>${data['cap']:g}</b></span>"
              f"<span>Salaries <b>{_esc(ui.tip_pt(data['salaries_utc']))}</b></span>"
              f"<span>Projections <b>{_esc(ui.tip_pt(data['sleeper_utc']) if data['sleeper_utc'] else '—')}</b></span>"
              "</div>")
    games = " · ".join(f"{_esc(g['away'])} @ {_esc(g['home'])} {_esc(ui.tip_pt(g['start']))}"
                       for g in data["games"])
    miss = (f" {len(data['unmatched'])} Yahoo player(s) have no Sleeper projection; enter one "
            "to use them." if data["unmatched"] else "")
    controls = ("<div class='dfs-controls'>"
                "<label>Lineups<input type='number' id='dfs-n' min='1' max='20' value='10'></label>"
                "<label>Max exposure %<input type='number' id='dfs-exp' min='10' max='100' step='5' value='70'></label>"
                f"<label>Min salary $<input type='number' id='dfs-min' min='0' max='{data['cap']:g}' value='0'></label>"
                "<button type='button' class='dfs-btn primary' id='dfs-go'>Build lineups</button>"
                "<button type='button' class='dfs-btn' id='dfs-reset'>Reset</button></div>")
    pool = ("<section class='dfs-panel'><h2>Player pool</h2>"
            "<div class='dfs-pos' id='dfs-pos' role='group' aria-label='Filter by position'></div>"
            "<div class='dfs-scroll'><table class='dfs-table'><thead><tr><th>Lock</th><th>Out</th>"
            "<th>Player</th><th class='x'>Pos</th><th class='n'>Salary</th><th class='n'>Proj</th>"
            "<th class='n x'>Value</th><th class='n x'>FPPG</th><th class='n x'>Min</th></tr></thead>"
            "<tbody id='dfs-pool'></tbody></table></div></section>")
    out = ("<section class='dfs-panel dfs-out'><h2>Lineups</h2>" + controls +
           "<div id='dfs-msg' class='dfs-msg' aria-live='polite'>Building lineups…</div>"
           "<div id='dfs-out'></div><h2 style='margin-top:14px'>Exposure</h2>"
           "<ul class='dfs-exp' id='dfs-exposure'></ul></section>")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return (head + status + f"<p class='public-note'>{games}. Yahoo classic: PG, SG, G, SF, PF, F, "
            f"C, UTIL under the cap, players from at least {MIN_TEAMS} teams and at most "
            f"{MAX_PER_TEAM} from one. Value = projection per $10 of salary. Players whose game "
            f"has started are out.{miss}</p>"
            "<noscript><p class='public-note'>The optimizer needs JavaScript.</p></noscript>"
            "<div class='dfs-grid'>" + pool + out + "</div>" + note
            + f"<script type='application/json' id='dfs-data'>{payload}</script>"
            + f"<script>{SOLVER_JS}{UI_JS}</script>")


def page(ui, today, built, native=None, yahoo=None, sleeper=None, css=""):
    """The rendered dfs.html (ui: build_site, for the shared shell; css: the
    public pages' styles, which this page shares)."""
    try:
        data = slate(today, yahoo=yahoo, sleeper=sleeper, native=native)
    except Exception as e:  # noqa: BLE001 - the page says what is missing
        print(f"dfs slate unavailable ({e!r})", flush=True)
        data = None
    return ui.page("DFS lineups · NBA Composite", "dfs.html", body(data, ui), built,
                   extra_css=css + CSS)


# The browser half. solve(): for each position a 0/1 knapsack over exact
# salary for 1..3 players (C 1..2), forced players added; guards (PG+SG, 3
# or 4), forwards (SF+PF, 3 or 4) and centres (1 or 2) combined by max-plus
# convolution over salary for the three count patterns of 8 players that
# fill PG, SG, G, SF, PF, F, C, UTIL. kbest(): Lawler's partition (each
# child forbids one player of its parent's best and forces the ones before
# it), so lineups come out best-first and none twice; each is checked
# against the team rules and the exposure caps before it is kept.
SOLVER_JS = r"""
var DFS=(function(){
var POS=['PG','SG','SF','PF','C'],MAXK={PG:3,SG:3,SF:3,PF:3,C:2},NEG=-1e18;
function knap(items,K,S){var n=items.length,W=(K+1)*(S+1),dp=new Float64Array((n+1)*W).fill(NEG);dp[0]=0;
 for(var i=0;i<n;i++){var it=items[i],a=i*W,b=(i+1)*W;for(var j=0;j<W;j++)dp[b+j]=dp[a+j];
  for(var k=0;k<K;k++)for(var s=0;s+it.sal<=S;s++){var v=dp[a+k*(S+1)+s];if(v>NEG){var t=b+(k+1)*(S+1)+s+it.sal;if(v+it.v>dp[t])dp[t]=v+it.v;}}}
 return {dp:dp,n:n,K:K,S:S,items:items};}
function pick(T,k,s){var out=[],S=T.S,W=(T.K+1)*(S+1);for(var i=T.n;i>0&&k>0;i--){var cur=T.dp[i*W+k*(S+1)+s],prev=T.dp[(i-1)*W+k*(S+1)+s];
 if(cur!==prev){var it=T.items[i-1];out.push(it.id);k--;s-=it.sal;}}return out;}
function conv(A,B,S){var V=new Float64Array(S+1).fill(NEG),G=new Int16Array(S+1);for(var t=0;t<=S;t++){var a=A[t];if(a<=NEG)continue;
 for(var u=0;t+u<=S;u++){var b=B[u];if(b<=NEG)continue;if(a+b>V[t+u]){V[t+u]=a+b;G[t+u]=t;}}}return {v:V,g:G};}
// P: [{id,pos,sal,v}]; fin / fout: {id:1} forced in / out. The best lineup
// filling PG, SG, G, SF, PF, F, C, UTIL with salary in [minSal, cap], or null.
function solve(P,cap,minSal,fin,fout){var S=cap,tab={};
 for(var q=0;q<POS.length;q++){var pos=POS[q],forced=[],free=[],fs=0,fv=0;
  P.forEach(function(p){if(p.pos!==pos||fout[p.id])return;if(fin[p.id]){forced.push(p.id);fs+=p.sal;fv+=p.v;}else free.push(p);});
  if(forced.length>MAXK[pos]||fs>S)return null;var K=MAXK[pos]-forced.length,T=knap(free,K,S-fs),rows={};
  for(var c=1;c<=MAXK[pos];c++){var r=new Float64Array(S+1).fill(NEG),k=c-forced.length;if(k>=0&&k<=K){var W=(K+1)*(S-fs+1),base=T.n*W+k*(S-fs+1);
   for(var s=0;s+fs<=S;s++){var v=T.dp[base+s];if(v>NEG)r[s+fs]=v+fv;}}rows[c]=r;}
  tab[pos]={T:T,rows:rows,forced:forced,fs:fs};}
 if(Object.keys(fin).some(function(id){return !P.some(function(p){return p.id===id;});}))return null;
 function grp(a,b,n){var best=null;for(var x=1;x<n;x++){var y=n-x;if(x>MAXK[a]||y>MAXK[b])continue;var c=conv(tab[a].rows[x],tab[b].rows[y],S);
   if(!best){best={v:c.v,g:c.g,x:new Int8Array(S+1).fill(x)};continue;}for(var s=0;s<=S;s++)if(c.v[s]>best.v[s]){best.v[s]=c.v[s];best.g[s]=c.g[s];best.x[s]=x;}}return best;}
 var G={3:grp('PG','SG',3),4:grp('PG','SG',4)},F={3:grp('SF','PF',3),4:grp('SF','PF',4)},pats=[[3,3,2],[3,4,1],[4,3,1]],top=null;
 pats.forEach(function(pt){var X=conv(G[pt[0]].v,F[pt[1]].v,S),Y=conv(X.v,tab.C.rows[pt[2]],S);
  for(var s=Math.max(0,minSal);s<=S;s++)if(Y.v[s]>NEG&&(!top||Y.v[s]>top.v+1e-9))top={v:Y.v[s],s:s,pt:pt,X:X,Y:Y};});
 if(!top)return null;var sC=top.s-top.Y.g[top.s],sX=top.Y.g[top.s],sG=top.X.g[sX],sF=sX-sG,ids=[];
 function take(pos,c,s){var t=tab[pos];ids=ids.concat(t.forced,pick(t.T,c-t.forced.length,s-t.fs));}
 var g=G[top.pt[0]],a=g.x[sG],sPG=g.g[sG];take('PG',a,sPG);take('SG',top.pt[0]-a,sG-sPG);
 var f=F[top.pt[1]],c=f.x[sF],sSF=f.g[sF];take('SF',c,sSF);take('PF',top.pt[1]-c,sF-sSF);take('C',top.pt[2],sC);
 return {v:top.v,sal:top.s,ids:ids};}
function Heap(){this.a=[];}
Heap.prototype.push=function(x){var a=this.a;a.push(x);var i=a.length-1;while(i>0){var p=(i-1)>>1;if(a[p].sol.v>=a[i].sol.v)break;var t=a[p];a[p]=a[i];a[i]=t;i=p;}};
Heap.prototype.pop=function(){var a=this.a,top=a[0],last=a.pop();if(a.length){a[0]=last;var i=0;for(;;){var l=2*i+1,r=l+1,m=i;
 if(l<a.length&&a[l].sol.v>a[m].sol.v)m=l;if(r<a.length&&a[r].sol.v>a[m].sol.v)m=r;if(m===i)break;var t=a[m];a[m]=a[i];a[i]=t;i=m;}}return top;};
// Lineups best-first, none twice (Lawler): each child forbids one player of
// its parent's lineup and forces those before it. accept(sol) keeps one.
function kbest(P,cap,minSal,lock,N,accept,limit){var h=new Heap(),out=[],seen=0,root=solve(P,cap,minSal,lock,{});
 if(!root)return {lineups:out,seen:0};h.push({fin:lock,fout:{},sol:root});
 while(h.a.length&&out.length<N&&seen<limit){var nd=h.pop();seen++;if(accept(nd.sol))out.push(nd.sol);
  var fin=Object.assign({},nd.fin);nd.sol.ids.forEach(function(id){if(nd.fin[id])return;var fo=Object.assign({},nd.fout);fo[id]=1;
   var s=solve(P,cap,minSal,fin,fo);if(s)h.push({fin:Object.assign({},fin),fout:fo,sol:s});fin[id]=1;});}
 return {lineups:out,seen:seen};}
function teamsOk(ids,team,minTeams,maxPer){var c={};ids.forEach(function(i){var t=team[i];c[t]=(c[t]||0)+1;});var ks=Object.keys(c);
 return ks.length>=minTeams&&ks.every(function(k){return c[k]<=maxPer;});}
return {solve:solve,kbest:kbest,teamsOk:teamsOk};
})();
"""

# The page half: the pool table, controls and lineups (DOM only).
UI_JS = r"""
(function(){
var D=JSON.parse(document.getElementById('dfs-data').textContent);
var st={lock:{},out:{},proj:{},pos:'ALL'};
function $(id){return document.getElementById(id);}
function esc(s){return String(s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];});}
function proj(p){return st.proj[p.id]!==undefined?st.proj[p.id]:p.proj;}
function started(p){return Date.parse(p.start)<=Date.now();}
function pool(){return D.players.filter(function(p){var v=proj(p);return !st.out[p.id]&&!started(p)&&v!==null&&v!==undefined&&(v>0||st.lock[p.id]);})
 .map(function(p){return {id:p.id,pos:p.pos,sal:p.sal,v:proj(p)};});}
function slots(ids,byId){var ps=ids.map(function(i){return byId[i];}),used={},at={};
 function take(ok){for(var i=0;i<ps.length;i++){if(!used[i]&&ok(ps[i])){used[i]=1;return ps[i];}}return null;}
 [['PG',function(p){return p.pos==='PG';}],['SG',function(p){return p.pos==='SG';}],['SF',function(p){return p.pos==='SF';}],
  ['PF',function(p){return p.pos==='PF';}],['C',function(p){return p.pos==='C';}],['G',function(p){return p.pos==='PG'||p.pos==='SG';}],
  ['F',function(p){return p.pos==='SF'||p.pos==='PF';}],['UTIL',function(){return true;}]].forEach(function(x){at[x[0]]=take(x[1]);});
 return D.slots.map(function(s){return [s,at[s]];});}
function run(){var btn=$('dfs-go');btn.disabled=true;$('dfs-msg').textContent='Building lineups…';
 setTimeout(function(){try{build();}catch(e){$('dfs-msg').textContent='The optimizer failed: '+e.message;}btn.disabled=false;},20);}
// N lineups, each the best legal one not yet built from the players still
// under their exposure cap (locked players are exempt): a capped player
// leaves the pool, so the search never wades through lineups built on him.
function build(){var all=pool(),byId={},team={};D.players.forEach(function(p){byId[p.id]=p;team[p.id]=p.team;});
 var N=Math.max(1,Math.min(20,+$('dfs-n').value||10)),exp=Math.max(10,Math.min(100,+$('dfs-exp').value||100)),
  minSal=Math.max(0,+$('dfs-min').value||0),capN=Math.max(1,Math.floor(N*exp/100+1e-9)),cnt={},lock={},made={},L=[];
 Object.keys(st.lock).forEach(function(id){if(st.lock[id]&&byId[id]&&!started(byId[id]))lock[id]=1;});
 var t0=performance.now();
 while(L.length<N){var P=all.filter(function(p){return lock[p.id]||(cnt[p.id]||0)<capN;});
  var r=DFS.kbest(P,D.cap,minSal,lock,1,function(sol){return DFS.teamsOk(sol.ids,team,D.min_teams,D.max_per_team)&&!made[sol.ids.slice().sort().join()];},400);
  if(!r.lineups.length)break;var l=r.lineups[0];made[l.ids.slice().sort().join()]=1;L.push(l);l.ids.forEach(function(i){cnt[i]=(cnt[i]||0)+1;});}
 var el=$('dfs-out');
 if(!L.length){el.innerHTML='';$('dfs-exposure').innerHTML='';$('dfs-msg').textContent='No legal lineup: check the locks, exclusions and minimum salary.';return;}
 $('dfs-msg').textContent=L.length+' lineup'+(L.length>1?'s':'')+' · best '+L[0].v.toFixed(2)+' pts'+(L.length<N?' · fewer than asked under these caps':'')+' · '+Math.round(performance.now()-t0)+' ms';
 el.innerHTML=L.map(function(l,i){return "<article class='dfs-lineup'><header><span>#"+(i+1)+"</span><span>$<b>"+l.sal+"</b> · <b>"+l.v.toFixed(2)+"</b> pts</span></header><ol>"+
  slots(l.ids,byId).map(function(x){var p=x[1];return "<li><span>"+x[0]+"</span><span>"+esc(p.name)+" <em>"+esc(p.team)+"</em></span><span class='n'>$"+p.sal+"</span><span class='n'>"+proj(p).toFixed(1)+"</span></li>";}).join('')+"</ol></article>";}).join('');
 var ex=Object.keys(cnt).sort(function(a,b){return cnt[b]-cnt[a]||byId[a].name.localeCompare(byId[b].name);});
 $('dfs-exposure').innerHTML=ex.map(function(id){return "<li><span>"+esc(byId[id].name)+"</span><b class='mono'>"+Math.round(100*cnt[id]/L.length)+"%</b></li>";}).join('');}
function draw(){var rows=D.players.slice().sort(function(a,b){return (proj(b)||0)-(proj(a)||0)||b.sal-a.sal;});
 $('dfs-pool').innerHTML=rows.filter(function(p){return st.pos==='ALL'||p.pos===st.pos||(st.pos==='G'&&(p.pos==='PG'||p.pos==='SG'))||(st.pos==='F'&&(p.pos==='SF'||p.pos==='PF'));}).map(function(p){
  var v=proj(p),gone=started(p),cls=st.lock[p.id]?'locked':(st.out[p.id]||gone?'out':'');
  return "<tr class='"+cls+"' data-id='"+esc(p.id)+"'><td><input type='checkbox' data-act='lock' aria-label='Lock "+esc(p.name)+"'"+(st.lock[p.id]?' checked':'')+(gone?' disabled':'')+"></td>"+
   "<td><input type='checkbox' data-act='out' aria-label='Exclude "+esc(p.name)+"'"+(st.out[p.id]||gone?' checked':'')+(gone?' disabled':'')+"></td>"+
   "<td class='nm'>"+esc(p.name)+(p.inj?"<span class='dfs-inj'>"+esc(p.inj)+"</span>":"")+"<small><span class='mpos'>"+p.pos+" · </span>"+esc(p.team)+(p.home?' vs ':' @ ')+esc(p.opp)+(gone?' · started':'')+"</small></td>"+
   "<td class='x'>"+p.pos+"</td><td class='n'>$"+p.sal+"</td><td class='n'><input type='number' step='0.1' min='0' data-act='proj' aria-label='Projection for "+esc(p.name)+"' value='"+(v===null||v===undefined?'':v)+"'></td>"+
   "<td class='n x'>"+(v?(10*v/p.sal).toFixed(2):'—')+"</td><td class='n x'>"+(p.fppg===null?'—':p.fppg.toFixed(1))+"</td><td class='n x'>"+(p.min===null?'—':p.min)+"</td></tr>";}).join('');}
$('dfs-pos').innerHTML=['ALL','PG','SG','G','SF','PF','F','C'].map(function(k){return "<button type='button' data-pos='"+k+"' aria-pressed='"+(k===st.pos)+"'>"+(k==='ALL'?'All':k)+"</button>";}).join('');
$('dfs-pos').addEventListener('click',function(e){var b=e.target.closest('button');if(!b)return;st.pos=b.dataset.pos;
 this.querySelectorAll('button').forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false');});draw();});
$('dfs-pool').addEventListener('change',function(e){var t=e.target,tr=t.closest('tr');if(!tr)return;var id=tr.dataset.id,a=t.dataset.act;
 if(a==='lock'){st.lock[id]=t.checked;if(t.checked)st.out[id]=false;}else if(a==='out'){st.out[id]=t.checked;if(t.checked)st.lock[id]=false;}
 else if(a==='proj'){var v=parseFloat(t.value);if(isFinite(v)&&v>=0)st.proj[id]=v;else delete st.proj[id];}draw();});
$('dfs-go').addEventListener('click',run);
$('dfs-reset').addEventListener('click',function(){st.lock={};st.out={};st.proj={};draw();run();});
draw();run();
})();
"""
