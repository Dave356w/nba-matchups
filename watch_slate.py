"""Wake at each game's five-hour lock window and dispatch build.yml.

WHY, MEASURED. The five-hour lock (CLAUDE.md amendment, 2026-10-28) needs a
build inside each game's last five hours, and GitHub delivers a fraction of
the scheduled builds this repository asks for. From the 30-minute schedule's
merge (2026-10-28T21:44Z) to 2026-10-29T20:45Z, build.yml requested 33
scheduled runs and GitHub created 4 (23:59Z, 05:39Z, 10:54Z and 17:26Z).
On the lock's first slate, 2026-10-28, 3 of 12 games locked and 9 had no
snapshot in their window. On 2026-10-29, the first v11-family slate, no build
ran between 17:27Z and 20:45Z while two lock windows were open. A missed lock
is a game with no tracked bet, and it cannot be recovered after tip.

WHAT THIS DOES. One delivered run starts a shift (watch.yml). The shift reads
the day's slate from ESPN and the lock columns of data/nba_ledger.csv on main,
sleeps until the next game's window opens (tip - 5 h, plus OPEN_MARGIN), and
dispatches build.yml then. A game still unlocked RETRY_SECONDS after a build
was dispatched for it is dispatched again, so a build that fails or a price
that is missing is retried until tip. Before GitHub's 6-hour job cap, a shift
with games still to lock dispatches watch.yml again (RELAY_SECONDS): the
next shift waits in the site-watch queue and starts when this one ends. A
build from any path (a push, a dispatch, a delivered cron) also starts a
shift when none is running (build.yml, `--check`). So one delivered run
covers the rest of the slate, and the chain ends when every game has tipped.

WHAT IT DOES NOT CHANGE. The lock is still the first pregame snapshot within
five hours of tip (ledger.upsert_pregame), and builds are still safe at any
time. This only makes a build land in each window. build.yml keeps its own
schedule as an independent path; the watcher writes nothing under data/.
Stdlib only, so a shift costs no install and sleeps almost all of its time.
"""

import csv
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# ledger.LOCK_LEAD_HOURS; tests/test_watch_slate.py pins the two together.
LOCK_LEAD = timedelta(hours=5)
# Dispatch this long after a window opens, so the build's snapshot cannot
# land before it (a run takes about a minute to reach the snapshot).
OPEN_MARGIN = timedelta(minutes=2)
# Nothing is dispatched for a game this close to tip: the build would finish
# after it, when the row is frozen.
MIN_BEFORE = timedelta(minutes=10)
# A game still unlocked this long after its last dispatch is dispatched again.
RETRY_SECONDS = 30 * 60
# Longest sleep between slate re-reads (tip times and the ledger move).
RECHECK_SECONDS = 30 * 60
# After a failed dispatch, come back soon rather than at the next target.
FAILED_SECONDS = 2 * 60
# GitHub caps a job at 6 hours; watch.yml bounds the process just above this.
SHIFT_SECONDS = 5 * 3600 + 20 * 60
# Hand over to the next shift this long before the deadline.
RELAY_SECONDS = 2 * 60

# market.SCOREBOARD and market.REGULAR_SEASON (pinned by the tests): this
# module stays stdlib only, so it does not import market.
SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
              "scoreboard?dates={ds}&limit=100")
REGULAR_SEASON = 2
LEDGER_PATH = "data/nba_ledger.csv"
LOCK_MARKERS = ("lock_snapshot_utc", "lock_ou_snapshot_utc")

BUILD_WORKFLOW = "build.yml"
WATCH_WORKFLOW = "watch.yml"
REF = "main"
API = "https://api.github.com"


def log(msg):
    print(msg, flush=True)


def slate_date(now):
    """The ET calendar date, as build_site.et_today() scores it."""
    return now.astimezone(ET).strftime("%Y-%m-%d")


def parse_utc(s):
    if not s:
        return None
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t.astimezone(timezone.utc) if t.tzinfo else t.replace(tzinfo=timezone.utc)


def fetch_games(day):
    """Regular-season games on the ESPN scoreboard for an ET date:
    [{"game_id", "tip", "state"}]."""
    url = SCOREBOARD.format(ds=day.replace("-", ""))
    req = Request(url, headers={"User-Agent": "nba-matchups-lock-watch/1.0"})
    with urlopen(req, timeout=30) as r:
        js = json.load(r)
    out = []
    for ev in js.get("events", []):
        if (ev.get("season") or {}).get("type") != REGULAR_SEASON:
            continue
        comp = (ev.get("competitions") or [{}])[0]
        state = ((comp.get("status") or ev.get("status") or {}).get("type") or {}).get("state")
        tip = parse_utc(ev.get("date"))
        if tip is not None:
            out.append({"game_id": str(ev["id"]), "tip": tip, "state": state})
    return out


def read_csv_rows(text):
    return {r["game_id"]: r for r in csv.DictReader(io.StringIO(text))}


def fetch_ledger():
    """The ledger as committed on main now (the build commits it there)."""
    subprocess.run(["git", "fetch", "--quiet", "--depth=1", "origin", REF],
                   check=True, timeout=120)
    out = subprocess.run(["git", "show", f"FETCH_HEAD:{LEDGER_PATH}"],
                         check=True, timeout=60, capture_output=True, text=True)
    return read_csv_rows(out.stdout)


def local_ledger(path=LEDGER_PATH):
    with open(path, newline="", encoding="utf-8") as f:
        return read_csv_rows(f.read())


def locked(row):
    """Both write-once lock snapshots (moneyline/spread and O/U) are taken."""
    return row is not None and all((row.get(c) or "").strip() for c in LOCK_MARKERS)


def plan(games, rows, day, now):
    """Which games still need a build, and when.

    Returns (pending, due, openings): pending = game ids still more than
    MIN_BEFORE from tip and not locked; due = those a build should cover now
    (no ledger row yet, or inside their lock window); openings = future
    window openings (+ OPEN_MARGIN) of the rest. `games` None means the
    scoreboard failed: the ledger's own rows for `day` stand in for it.
    """
    if games is None:
        games = [{"game_id": gid, "tip": parse_utc(r.get("tip_utc")), "state": "pre"}
                 for gid, r in rows.items()
                 if r.get("slate_date") == day and not (r.get("home_won") or "").strip()]
    pending, due, openings = [], [], []
    for g in games:
        tip = g["tip"]
        if tip is None or g["state"] not in (None, "pre") or tip - now <= MIN_BEFORE:
            continue
        row = rows.get(g["game_id"])
        if locked(row):
            continue
        pending.append(g["game_id"])
        opens = tip - LOCK_LEAD + OPEN_MARGIN
        if row is None or now >= opens:
            due.append(g["game_id"])
        else:
            openings.append(opens)
    return pending, due, sorted(openings)


def _api(path, token, method="GET", body=None):
    req = Request(
        f"{API}{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "User-Agent": "nba-matchups-lock-watch/1.0",
                 "Content-Type": "application/json"})
    with urlopen(req, timeout=30) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def dispatch(repo, token, workflow):
    """Dispatch a workflow on main. GITHUB_TOKEN suffices: workflow_dispatch
    is one of the two events a GITHUB_TOKEN may trigger. Never raises;
    returns whether it landed."""
    try:
        _api(f"/repos/{repo}/actions/workflows/{workflow}/dispatches", token,
             method="POST", body={"ref": REF})
    except (HTTPError, URLError, OSError, ValueError) as e:
        log(f"dispatch {workflow} failed ({type(e).__name__}: {e})")
        return False
    return True


def watch(deadline, send, clock=time.time, sleep=time.sleep,
          now_utc=lambda: datetime.now(timezone.utc),
          games_fn=fetch_games, ledger_fn=fetch_ledger):
    """Dispatch builds at the slate's lock windows until `deadline` (a
    clock() value). `send(workflow)` dispatches and returns whether it
    landed. Returns the number of builds dispatched."""
    last = {}          # game id (or "slate") -> when a build was last sent for it
    sent = 0
    while True:
        now = now_utc()
        day = slate_date(now)
        stamp = now.strftime("%H:%M:%SZ")
        try:
            games = games_fn(day)
        except Exception as e:  # noqa: BLE001 - a scoreboard outage must not end the shift
            log(f"{stamp} scoreboard failed ({type(e).__name__}); reading the ledger's slate")
            games = None
        try:
            rows = ledger_fn()
        except Exception as e:  # noqa: BLE001 - fail open: every game looks unlocked
            log(f"{stamp} ledger read failed ({type(e).__name__}); treating games as unlocked")
            rows = {}
        pending, due, openings = plan(games, rows, day, now)
        unknown = games is None and not any(r.get("slate_date") == day
                                            for r in rows.values())
        if not pending and not unknown:
            log(f"{stamp} nothing left to lock on {day}; {sent} build(s) dispatched")
            return sent
        if unknown:
            due = ["slate"]   # neither source lists the slate; a build can
        retry = timedelta(seconds=RETRY_SECONDS)
        wait = RECHECK_SECONDS
        if any(g not in last or now - last[g] >= retry for g in due):
            if send(BUILD_WORKFLOW):
                sent += 1
                last.update((g, now) for g in due)
                log(f"{stamp} dispatched build for {len(due)} game(s): {', '.join(due)}")
            else:
                wait = FAILED_SECONDS
        wakes = openings + [last[g] + retry for g in due
                            if g in last and last[g] + retry > now]
        if wakes:
            wait = min(wait, max(1.0, (min(wakes) - now).total_seconds()))
        remaining = deadline - clock()
        if remaining <= RELAY_SECONDS + 1:
            relayed = False
            for _ in range(3 if pending else 0):
                relayed = send(WATCH_WORKFLOW)
                if relayed:
                    break
                sleep(RELAY_SECONDS / 4)
            log(f"{stamp} shift ends; {sent} build(s) dispatched"
                + (f"; next shift dispatched for {len(pending)} unlocked game(s)"
                   if relayed else ""))
            return sent
        nxt = (now + timedelta(seconds=wait)).strftime("%H:%MZ")
        log(f"{stamp} {len(pending)} game(s) unlocked; next check {nxt}")
        sleep(min(wait, remaining - RELAY_SECONDS))


def needs_watch(now=None, games_fn=fetch_games, ledger_fn=local_ledger):
    """For build.yml: does today's slate still have a game to lock?
    True when the scoreboard fails (the shift then decides)."""
    now = now or datetime.now(timezone.utc)
    day = slate_date(now)
    try:
        pending, _, _ = plan(games_fn(day), ledger_fn(), day, now)
    except Exception:  # noqa: BLE001
        return True
    return bool(pending)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--check"]:
        need = needs_watch()
        log("lock watch needed" if need else "no game left to lock today")
        return 0 if need else 3
    repo, token = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        log("GITHUB_REPOSITORY and GITHUB_TOKEN are required")
        return 2
    shift = float(os.environ.get("SHIFT_SECONDS", SHIFT_SECONDS))
    log(f"watching {repo} for {shift / 3600:.2f} h: a build at each lock window "
        f"(tip - {LOCK_LEAD} + {OPEN_MARGIN}), retried every {RETRY_SECONDS // 60} min")
    watch(time.time() + shift, lambda wf: dispatch(repo, token, wf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
