"""Lock watch (watch_slate.py, watch.yml): a build lands in every game's
five-hour lock window however few scheduled runs GitHub delivers."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ledger
import market
import watch_slate as ws

ROOT = Path(__file__).resolve().parent.parent
T0 = datetime(2026, 10, 29, 18, 0, tzinfo=timezone.utc)      # 2 pm ET
DAY = "2026-10-29"


def _game(gid, tip, state="pre"):
    return {"game_id": gid, "tip": tip, "state": state}


def _row(gid, tip, locked=False, ou_locked=None, slate=DAY):
    ou_locked = locked if ou_locked is None else ou_locked
    return {"game_id": gid, "slate_date": slate, "tip_utc": ledger.fmt_utc(tip),
            "home_won": "",
            "lock_snapshot_utc": "2026-10-29T19:30:00Z" if locked else "",
            "lock_ou_snapshot_utc": "2026-10-29T19:30:00Z" if ou_locked else ""}


class Sim:
    """A fake clock: sleep advances it, send records (time, workflow) and
    optionally locks the games a build was sent for."""

    def __init__(self, games, rows, start=T0, lock_on_build=True, fail=()):
        self.now, self.games, self.rows = start, games, rows
        self.start, self.sent = start, []
        self.lock_on_build, self.fail = lock_on_build, set(fail)

    def clock(self):
        return (self.now - self.start).total_seconds()

    def sleep(self, s):
        assert s > 0
        self.now += timedelta(seconds=s)

    def send(self, wf):
        if (len(self.sent), wf) in self.fail:
            self.sent.append((self.now, wf + " (failed)"))
            return False
        self.sent.append((self.now, wf))
        if wf == ws.BUILD_WORKFLOW and self.lock_on_build:
            for g in self.games or []:
                opens = g["tip"] - ws.LOCK_LEAD
                if opens <= self.now < g["tip"]:
                    self.rows[g["game_id"]] = _row(g["game_id"], g["tip"], locked=True)
                elif self.now < g["tip"]:
                    self.rows.setdefault(g["game_id"], _row(g["game_id"], g["tip"]))
        return True

    def run(self, shift=ws.SHIFT_SECONDS):
        return ws.watch(shift, self.send, clock=self.clock, sleep=self.sleep,
                        now_utc=lambda: self.now, games_fn=lambda day: self.games,
                        ledger_fn=lambda: dict(self.rows))


def test_constants_match_the_ledger_and_market():
    assert ws.LOCK_LEAD == timedelta(hours=ledger.LOCK_LEAD_HOURS)
    assert ws.SCOREBOARD == market.SCOREBOARD
    assert ws.REGULAR_SEASON == market.REGULAR_SEASON
    assert set(ws.LOCK_MARKERS) <= set(ledger.COLUMNS)
    assert ws.LOCK_MARKERS == (ledger.LOCK_COLUMNS[0], ledger.OU_LOCK_COLUMNS[0])


def test_slate_date_is_the_et_calendar_date_build_site_scores():
    late = datetime(2026, 10, 30, 3, 30, tzinfo=timezone.utc)   # 11:30 pm ET
    assert ws.slate_date(late) == DAY
    assert ws.slate_date(datetime(2026, 10, 30, 4, 1, tzinfo=timezone.utc)) == "2026-10-30"


def test_plan_dues_open_windows_and_rowless_games_and_skips_locked_ones():
    tip = T0 + timedelta(hours=4)                                # window open
    later = T0 + timedelta(hours=8)                              # opens in 3 h
    games = [_game("a", tip), _game("b", tip), _game("c", later), _game("d", later),
             _game("e", T0 + timedelta(minutes=5)),              # too close to tip
             _game("f", tip, state="in")]
    rows = {"a": _row("a", tip), "b": _row("b", tip, locked=True),
            "c": _row("c", later), "f": _row("f", tip)}
    pending, due, openings = ws.plan(games, rows, DAY, T0)
    assert pending == ["a", "c", "d"]
    assert due == ["a", "d"]                    # d has no ledger row yet
    assert openings == [later - ws.LOCK_LEAD + ws.OPEN_MARGIN]


def test_plan_needs_both_locks_and_falls_back_to_the_ledger_slate():
    tip = T0 + timedelta(hours=4)
    rows = {"a": _row("a", tip, locked=True, ou_locked=False),
            "z": _row("z", tip, slate="2026-10-28")}
    pending, due, _ = ws.plan(None, rows, DAY, T0)          # scoreboard failed
    assert pending == due == ["a"]


def test_one_build_per_window_at_each_opening_then_the_shift_ends():
    tips = [T0 + timedelta(hours=6), T0 + timedelta(hours=6),
            T0 + timedelta(hours=8)]
    games = [_game(g, t) for g, t in zip("abc", tips)]
    rows = {g["game_id"]: _row(g["game_id"], g["tip"]) for g in games}
    sim = Sim(games, rows)
    assert sim.run() == 2
    assert [t for t, _ in sim.sent] == [tips[0] - ws.LOCK_LEAD + ws.OPEN_MARGIN,
                                       tips[2] - ws.LOCK_LEAD + ws.OPEN_MARGIN]
    assert all(t >= g - ws.LOCK_LEAD for (t, _), g in zip(sim.sent, (tips[0], tips[2])))
    assert all(ws.locked(r) for r in rows.values())


def test_a_shift_starting_inside_a_window_builds_at_once():
    tip = T0 + timedelta(hours=2)
    sim = Sim([_game("a", tip)], {})
    assert sim.run() == 1
    assert sim.sent == [(T0, ws.BUILD_WORKFLOW)]


def test_an_unlocked_game_is_retried_until_tip_and_failed_dispatches_soon():
    tip = T0 + timedelta(hours=2)
    sim = Sim([_game("a", tip)], {"a": _row("a", tip)}, lock_on_build=False,
              fail={(1, ws.BUILD_WORKFLOW)})
    sim.run()
    builds = [t for t, wf in sim.sent if wf == ws.BUILD_WORKFLOW]
    assert builds[0] == T0
    assert sim.sent[1] == (T0 + timedelta(seconds=ws.RETRY_SECONDS), "build.yml (failed)")
    assert builds[1] == sim.sent[1][0] + timedelta(seconds=ws.FAILED_SECONDS)
    gaps = [(b - a).total_seconds() for a, b in zip(builds[1:], builds[2:])]
    assert gaps and all(g == ws.RETRY_SECONDS for g in gaps)
    assert builds[-1] < tip - ws.MIN_BEFORE


def test_the_shift_relays_before_its_deadline_only_while_games_remain():
    tip = T0 + timedelta(hours=11)                 # window opens after the shift
    sim = Sim([_game("a", tip)], {"a": _row("a", tip)})
    assert sim.run() == 0
    assert sim.sent == [(T0 + timedelta(seconds=ws.SHIFT_SECONDS - ws.RELAY_SECONDS),
                         ws.WATCH_WORKFLOW)]
    retried = Sim([_game("a", tip)], {"a": _row("a", tip)}, fail={(0, ws.WATCH_WORKFLOW)})
    retried.run()
    assert [wf for _, wf in retried.sent] == ["watch.yml (failed)", ws.WATCH_WORKFLOW]
    done = Sim([_game("a", tip)], {"a": _row("a", tip, locked=True)})
    assert done.run() == 0 and done.sent == []


def test_an_unknown_slate_fails_open_without_relaying():
    sim = Sim(None, {})
    calls = ws.watch(3 * 3600, sim.send, clock=sim.clock, sleep=sim.sleep,
                     now_utc=lambda: sim.now,
                     games_fn=lambda day: (_ for _ in ()).throw(OSError("espn")),
                     ledger_fn=lambda: {})
    assert calls == 6                               # every RETRY_SECONDS
    assert ws.WATCH_WORKFLOW not in {wf for _, wf in sim.sent}


def test_needs_watch_reads_the_local_ledger():
    tip = T0 + timedelta(hours=6)
    games = lambda day: [_game("a", tip)]           # noqa: E731
    assert ws.needs_watch(T0, games, lambda: {"a": _row("a", tip)})
    assert not ws.needs_watch(T0, games, lambda: {"a": _row("a", tip, locked=True)})
    assert ws.needs_watch(T0, lambda day: 1 / 0, lambda: {})


def test_watch_workflow_is_bounded_and_separate_from_site_build():
    wf = (ROOT / ".github/workflows/watch.yml").read_text()
    assert "group: site-watch" in wf and "site-build" not in wf.split("concurrency:")[1].split("\n")[1]
    assert "cancel-in-progress: false" in wf
    assert "actions: write" in wf and "contents: read" in wf
    assert "contents: write" not in wf
    kill = int(wf.split("timeout --signal=INT --kill-after=60 ")[1].split()[0])
    step = int(wf.split("timeout-minutes: ")[2].split()[0])
    job = int(wf.split("timeout-minutes: ")[1].split()[0])
    assert ws.SHIFT_SECONDS < kill < kill + 60 < step * 60 < job * 60 <= 6 * 3600
    assert "pip install" not in wf


def test_build_starts_a_shift_without_risking_the_ledger_commit():
    wf = (ROOT / ".github/workflows/build.yml").read_text()
    assert "actions: write" in wf
    assert wf.index("- name: Validate and commit ledger") < wf.index(
        "- name: Start the lock watcher")
    step = wf[wf.index("- name: Start the lock watcher"):]
    step = step[:step.index("\n      - ")]
    assert "if: ${{ !cancelled() }}" in step and "continue-on-error: true" in step
    assert "watch_slate.py --check" in step and "gh workflow run watch.yml" in step
    # the schedule stays an independent path
    assert "cron: '13,43 11-23 * * *'" in wf and "group: site-build" in wf
