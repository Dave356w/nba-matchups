"""Lock watch (watch_slate.py, watch.yml): a build lands in every game's
five-hour lock window, and one shortly before each tip (pre_spread for live
win chance), however few scheduled runs GitHub delivers."""
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


def _row(gid, tip, locked=False, ou_locked=None, slate=DAY, snapshot=None,
         book="dk", spread="-4.5"):
    ou_locked = locked if ou_locked is None else ou_locked
    return {"game_id": gid, "slate_date": slate, "tip_utc": ledger.fmt_utc(tip),
            "home_won": "", "snapshot_utc": ledger.fmt_utc(snapshot) if snapshot else "",
            "pre_book": book, "pre_spread": spread,
            "lock_snapshot_utc": "2026-10-29T19:30:00Z" if locked else "",
            "lock_ou_snapshot_utc": "2026-10-29T19:30:00Z" if ou_locked else ""}


class Sim:
    """A fake clock: sleep advances it, send records (time, workflow) and,
    like a build, refreshes every untipped game's snapshot (`refresh`) and
    locks those inside their window (`lock_on_build`)."""

    def __init__(self, games, rows, start=T0, lock_on_build=True, fail=(), refresh=True):
        self.now, self.games, self.rows = start, games, rows
        self.start, self.sent = start, []
        self.lock_on_build, self.fail, self.refresh = lock_on_build, set(fail), refresh

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
        if wf == ws.BUILD_WORKFLOW:
            for g in self.games or []:
                if self.now >= g["tip"]:
                    continue
                old = self.rows.get(g["game_id"]) or _row(g["game_id"], g["tip"])
                lock = ws.locked(old) or (self.lock_on_build and self.now >= g["tip"] - ws.LOCK_LEAD)
                snap = self.now if self.refresh else ws.parse_utc(old.get("snapshot_utc"))
                self.rows[g["game_id"]] = _row(g["game_id"], g["tip"], locked=lock, snapshot=snap)
        return True

    def run(self, shift=ws.SHIFT_SECONDS):
        return ws.watch(shift, self.send, clock=self.clock, sleep=self.sleep,
                        now_utc=lambda: self.now, games_fn=lambda day: self.games,
                        ledger_fn=lambda: dict(self.rows))


def test_pretip_timing_leaves_room_for_a_build_and_one_retry():
    # builds reach their snapshot in about a minute; the retry must still land
    assert ws.PRETIP_LEAD - timedelta(seconds=ws.PRETIP_RETRY_SECONDS) > ws.MIN_BEFORE
    assert "snapshot_utc" in ledger.COLUMNS


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
    # b is locked but still awaits its pre-tip build
    assert pending == ["a", "b", "c", "d"]
    assert due == ["a", "d"]                    # d has no ledger row yet
    pre = lambda t: t - ws.PRETIP_LEAD          # noqa: E731
    assert openings == sorted([pre(tip), pre(tip), later - ws.LOCK_LEAD + ws.OPEN_MARGIN,
                               pre(later), pre(later)])


def test_plan_needs_both_locks_and_falls_back_to_the_ledger_slate():
    tip = T0 + timedelta(hours=4)
    rows = {"a": _row("a", tip, locked=True, ou_locked=False),
            "z": _row("z", tip, slate="2026-10-28")}
    pending, due, _ = ws.plan(None, rows, DAY, T0)          # scoreboard failed
    assert pending == due == ["a"]


def test_plan_dues_the_pretip_build_until_a_recent_snapshot_is_saved():
    tip = T0 + timedelta(minutes=30)
    games = [_game("a", tip)]
    at = tip - ws.PRETIP_LEAD
    before = ws.plan(games, {"a": _row("a", tip, locked=True, snapshot=T0)}, DAY, T0)
    assert before == (["a"], [], [at])
    old = {"a": _row("a", tip, locked=True, snapshot=at - timedelta(minutes=1))}
    assert ws.plan(games, old, DAY, at) == (["a"], ["a" + ws.PRETIP], [])
    fresh = {"a": _row("a", tip, locked=True, snapshot=at)}
    assert ws.plan(games, fresh, DAY, at) == ([], [], [])
    # an unlocked game inside its window is due for both in one build
    assert ws.plan(games, {"a": _row("a", tip, snapshot=T0)}, DAY, at)[1] == ["a", "a" + ws.PRETIP]
    # never within MIN_BEFORE of tip
    assert ws.plan(games, old, DAY, tip - ws.MIN_BEFORE) == ([], [], [])


def test_one_build_per_window_at_each_opening_then_the_shift_ends():
    tips = [T0 + timedelta(hours=6), T0 + timedelta(hours=6),
            T0 + timedelta(hours=8)]
    games = [_game(g, t) for g, t in zip("abc", tips)]
    rows = {g["game_id"]: _row(g["game_id"], g["tip"]) for g in games}
    sim = Sim(games, rows)
    assert sim.run(shift=10 * 3600) == 4      # one long shift (relays are tested below)
    # one build per lock window and one per distinct tip: shared tips share builds
    assert [t for t, _ in sim.sent] == [tips[0] - ws.LOCK_LEAD + ws.OPEN_MARGIN,
                                       tips[2] - ws.LOCK_LEAD + ws.OPEN_MARGIN,
                                       tips[0] - ws.PRETIP_LEAD,
                                       tips[2] - ws.PRETIP_LEAD]
    assert all(ws.locked(r) for r in rows.values())
    assert all(ws.pretip_done(r, g["tip"]) for r, g in zip(rows.values(), games))


def test_a_shift_starting_inside_a_window_builds_at_once():
    tip = T0 + timedelta(hours=2)
    sim = Sim([_game("a", tip)], {})
    assert sim.run() == 2
    assert sim.sent == [(T0, ws.BUILD_WORKFLOW), (tip - ws.PRETIP_LEAD, ws.BUILD_WORKFLOW)]


def test_an_unlocked_game_is_retried_until_tip_and_failed_dispatches_soon():
    tip = T0 + timedelta(hours=2)
    sim = Sim([_game("a", tip)], {"a": _row("a", tip)}, lock_on_build=False,
              fail={(1, ws.BUILD_WORKFLOW)})
    sim.run()
    builds = [t for t, wf in sim.sent if wf == ws.BUILD_WORKFLOW]
    assert builds[0] == T0
    assert sim.sent[1] == (T0 + timedelta(seconds=ws.RETRY_SECONDS), "build.yml (failed)")
    assert builds[1] == sim.sent[1][0] + timedelta(seconds=ws.FAILED_SECONDS)
    lock_retries = [b for b in builds[1:] if b < tip - ws.PRETIP_LEAD]
    gaps = [(b - a).total_seconds() for a, b in zip(lock_retries, lock_retries[1:])]
    assert gaps and all(g == ws.RETRY_SECONDS for g in gaps)
    assert tip - ws.PRETIP_LEAD in builds       # the pre-tip build still lands
    assert builds[-1] < tip - ws.MIN_BEFORE


def test_a_recent_snapshot_without_a_draftkings_spread_is_not_done():
    tip = T0 + timedelta(minutes=30)
    at = tip - ws.PRETIP_LEAD
    good = _row("a", tip, locked=True, snapshot=at)
    assert ws.pretip_done(good, tip)
    for bad in (dict(spread=""), dict(spread="nan"), dict(spread="x"), dict(book="espnbet"),
                dict(book="")):
        row = _row("a", tip, locked=True, snapshot=at, **bad)
        assert not ws.pretip_done(row, tip)
        # still due, so the retry runs
        assert ws.plan([_game("a", tip)], {"a": row}, DAY, at)[1] == ["a" + ws.PRETIP]
    assert not ws.pretip_done(None, tip)


def test_a_failed_pretip_fetch_gets_its_retry():
    # the build runs (snapshot_utc advances) but saves no spread both times
    tip = T0 + timedelta(hours=1)
    rows = {"a": _row("a", tip, locked=True, snapshot=T0)}
    sim = Sim([_game("a", tip)], rows)
    send = sim.send

    def blank(wf):
        ok = send(wf)
        rows["a"] = dict(rows["a"], pre_spread="")
        return ok
    sim.send = blank
    assert sim.run() == 2
    assert [t for t, _ in sim.sent] == [tip - ws.PRETIP_LEAD,
                                       tip - ws.PRETIP_LEAD + timedelta(seconds=ws.PRETIP_RETRY_SECONDS)]


def test_a_pretip_build_that_saves_nothing_is_retried_once_before_tip():
    tip = T0 + timedelta(hours=1)
    sim = Sim([_game("a", tip)], {"a": _row("a", tip, locked=True, snapshot=T0)}, refresh=False)
    assert sim.run() == 2
    assert [t for t, _ in sim.sent] == [tip - ws.PRETIP_LEAD,
                                       tip - ws.PRETIP_LEAD + timedelta(seconds=ws.PRETIP_RETRY_SECONDS)]
    assert all(t < tip - ws.MIN_BEFORE for t, _ in sim.sent)


def test_the_shift_relays_before_its_deadline_only_while_games_remain():
    tip = T0 + timedelta(hours=11)                 # window opens after the shift
    sim = Sim([_game("a", tip)], {"a": _row("a", tip)})
    assert sim.run() == 0
    assert sim.sent == [(T0 + timedelta(seconds=ws.SHIFT_SECONDS - ws.RELAY_SECONDS),
                         ws.WATCH_WORKFLOW)]
    retried = Sim([_game("a", tip)], {"a": _row("a", tip)}, fail={(0, ws.WATCH_WORKFLOW)})
    retried.run()
    assert [wf for _, wf in retried.sent] == ["watch.yml (failed)", ws.WATCH_WORKFLOW]
    # a locked game still awaiting its pre-tip build keeps the chain going
    waiting = Sim([_game("a", tip)], {"a": _row("a", tip, locked=True)})
    assert waiting.run() == 0 and [wf for _, wf in waiting.sent] == [ws.WATCH_WORKFLOW]
    soon = T0 + timedelta(minutes=15)
    done = Sim([_game("a", soon)], {"a": _row("a", soon, locked=True, snapshot=soon - ws.PRETIP_LEAD)})
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
    # locked, but the pre-tip build is still to come
    assert ws.needs_watch(T0, games, lambda: {"a": _row("a", tip, locked=True)})
    fresh = _row("a", tip, locked=True, snapshot=tip - ws.PRETIP_LEAD)
    assert not ws.needs_watch(T0, games, lambda: {"a": fresh})
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
