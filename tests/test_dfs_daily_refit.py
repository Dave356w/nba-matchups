"""Daily DFS learning: grading order, one attempt per ET day, safe promotion."""
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

import dfs
from research import dfs_calibration as cal


def accepted(as_of="2026-10-30"):
    return dict(version=dfs.SIM["version"], accepted=True, as_of=as_of,
                trained_through="2026-10-27", validation_from="2026-10-28",
                validated_through="2026-10-29", bins={"high:starter": 1.1},
                factors={k: dfs.SIM[k] for k in
                         ("team", "starter", "game_star", "star_split", "center_pair")},
                test_rows=311, test_dates=2)


def history(graded=True):
    native = pd.DataFrame([dict(game_id="1", basis="native", home_won=1 if graded else float("nan"),
                                slate_date="2026-10-29", tip_utc="2026-10-30T02:00Z")])
    sleeper = pd.DataFrame([dict(game_id="1")])
    return native, sleeper


def test_daily_waits_for_morning_and_grading_and_retries_later(tmp_path):
    native, sleeper = history()
    report, model = tmp_path/"report.json", tmp_path/"fit.json"
    def due(hour, df=native, day="2026-10-30"):
        return cal.daily_due(df, sleeper, day, report, model,
                             now=datetime(2026, 10, 30, hour, tzinfo=timezone.utc))
    assert not due(11)[0]  # 07:00 ET
    assert due(12)[0]      # 08:00 ET
    pending, _ = history(graded=False)
    assert not due(13, pending)[0]
    assert due(14, native)[0]  # next delivered build after grading
    assert not due(14, day="2026-10-29")[0]  # historical build
    assert not report.exists() and not model.exists()


def test_daily_skips_completed_attempt_and_is_due_next_day(tmp_path):
    native, sleeper = history()
    report, model = tmp_path/"report.json", tmp_path/"fit.json"
    report.write_text(json.dumps(dict(as_of="2026-10-30", accepted=False)))
    assert not cal.daily_due(native, sleeper, "2026-10-30", report, model,
                            now=datetime(2026, 10, 30, 14, tzinfo=timezone.utc))[0]
    assert cal.daily_due(native, sleeper, "2026-10-31", report, model,
                         now=datetime(2026, 10, 31, 14, tzinfo=timezone.utc))[0]
    report.unlink()
    model.write_text(json.dumps(accepted()))
    assert not cal.daily_due(native, sleeper, "2026-10-30", report, model,
                            now=datetime(2026, 10, 30, 14, tzinfo=timezone.utc))[0]


def test_daily_no_native_history_defers_without_writes(tmp_path):
    native, sleeper = history()
    assert not cal.daily_due(native.iloc[:0], sleeper, "2026-10-30", tmp_path/"r", tmp_path/"m",
                            now=datetime(2026, 10, 30, 14, tzinfo=timezone.utc))[0]


def test_rejected_candidate_preserves_active_model_bytes_and_records_reason(tmp_path):
    output, report = tmp_path/"model.json", tmp_path/"report.json"
    output.write_text(json.dumps(accepted()))
    before = output.read_bytes()
    bad = dict(accepted("2026-10-31"), accepted=False, reason="holdout failed")
    assert not cal.save_candidate(bad, output, report)
    assert output.read_bytes() == before
    evidence = json.loads(report.read_text())
    assert evidence["reason"] == "holdout failed"
    assert not evidence["accepted"] and not evidence["model_updated"]
    assert evidence["active_model_as_of"] == "2026-10-30"
    assert dfs.simulation_model("2026-10-31", output)["calibration_status"] != "provisional"


def test_rejected_first_fit_leaves_provisional_model_and_no_active_file(tmp_path):
    output, report = tmp_path/"model.json", tmp_path/"report.json"
    assert not cal.save_candidate(dict(accepted(), accepted=False), output, report)
    assert not output.exists()
    assert json.loads(report.read_text())["active_model_status"] == "provisional"


def test_accepted_fit_is_promoted_and_diagnostics_identify_active_version(tmp_path):
    output, report = tmp_path/"model.json", tmp_path/"report.json"
    fit = accepted()
    assert cal.save_candidate(fit, output, report)
    assert json.loads(output.read_text()) == fit
    evidence = json.loads(report.read_text())
    assert evidence["model_updated"] and evidence["active_model_as_of"] == fit["as_of"]
    assert not cal.save_candidate(fit, output, report)  # exact same fit
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_failure_cannot_corrupt_active_model(tmp_path):
    output = tmp_path/"fit.json"
    output.write_text(json.dumps(accepted()))
    before = output.read_bytes()
    with pytest.raises(ValueError):
        cal.atomic_json(output, {"invalid": float("nan")})
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_cli_outputs_update_and_skips_second_attempt_without_fetching(tmp_path, monkeypatch):
    native, sleeper = history()
    monkeypatch.setattr(cal.ledger, "load_sleeper", lambda: sleeper)
    monkeypatch.setattr(cal.ledger, "load", lambda path: native)
    monkeypatch.setattr(cal.ledger, "utc_now", lambda: datetime(2026, 10, 31, 14, tzinfo=timezone.utc))
    calls = []
    def collect(*args):
        calls.append(args)
        return accepted("2026-10-31")
    monkeypatch.setattr(cal, "collect_fit", collect)
    output, report, flags = tmp_path/"model.json", tmp_path/"report.json", tmp_path/"outputs"
    args = ["--daily", "--as-of", "2026-10-31", "--output", str(output),
            "--report", str(report), "--github-output", str(flags)]
    assert cal.main(args) == 0 and flags.read_text().splitlines()[-1] == "updated=true"
    flags.write_text("")
    assert cal.main(args) == 0 and flags.read_text().splitlines()[-1] == "updated=false"
    assert len(calls) == 1


def test_cli_refit_failure_still_defaults_to_no_rerender(tmp_path, monkeypatch):
    native, sleeper = history()
    monkeypatch.setattr(cal.ledger, "load_sleeper", lambda: sleeper)
    monkeypatch.setattr(cal.ledger, "load", lambda path: native)
    def fail(*args):
        raise OSError("source unavailable")
    monkeypatch.setattr(cal, "collect_fit", fail)
    flags = tmp_path/"outputs"
    with pytest.raises(OSError):
        cal.main(["--github-output", str(flags), "--as-of", "2026-10-31"])
    assert flags.read_text() == "updated=false\n"


def test_missing_final_boxes_fail_matching_gate_without_replacing_good_fit(tmp_path):
    native, _ = history()
    sleeper = pd.DataFrame([dict(game_id="1", team="BOS", sleeper_id="p1", name="A",
                                kind="pre", snapshot_utc="2026-10-29T23:00Z", minutes="30",
                                pts="20", reb="5", ast="2", stl="1", blk="1", to="2",
                                injury_status="")])
    def unavailable(gid, cache):
        raise OSError("source unavailable")
    candidate = cal.collect_fit(sleeper, native, "2026-10-31", tmp_path/"cache", fetch=unavailable)
    assert not candidate["accepted"]
    assert candidate["projected_rows"] == 1 and candidate["matched_rows"] == 0
    assert "95%" in candidate["reason"]
    output, report = tmp_path/"model.json", tmp_path/"report.json"
    output.write_text(json.dumps(accepted()))
    before = output.read_bytes()
    assert not cal.save_candidate(candidate, output, report)
    assert output.read_bytes() == before


def test_workflow_refits_after_saved_grading_then_renders_only_accepted_updates():
    # Exercise the actual control contract; no additional writer/workflow is
    # introduced, and a historical date cannot change the live fit.
    text = (Path(dfs.__file__).parent/".github/workflows/build.yml").read_text()
    assert text.index("name: Grade, score slate, render") < text.index("name: Refit DFS uncertainty")
    assert text.index("name: Refit DFS uncertainty") < text.index("name: Render with accepted DFS update")
    assert text.index("name: Render with accepted DFS update") < text.index("name: Validate and commit ledger")
    assert "if: ${{ !inputs.date }}" in text
    assert "steps.dfs_refit.outputs.updated == 'true'" in text
    assert "timeout-minutes: 3" in text
    assert "git add data/ model/dfs_simulation.json diagnostics/dfs_calibration/RESULTS.json" in text
