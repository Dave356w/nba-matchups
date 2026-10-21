"""Signed handicaps, pushes vs no-edge passes, mirrored teams, and look-ahead labels."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import median_ats_comparison as ats  # noqa: E402


def test_signed_handicap_and_push_grading():
    side, result = ats.select_and_grade([8, 2, -4, -7, 3], [6, 1, -2, -9, 6], [-3, -3, 5, 5, -3])
    assert np.array_equal(side, [1, -1, 1, -1, 1])
    assert list(result) == ["cover", "cover", "cover", "cover", "push"]
    s = ats.summary(result)
    assert s["covers"] == s["graded"] == 4
    assert s["pushes"] == 1 and s["cover_rate"] == 1


def test_no_edge_pass_is_not_push_or_noncover():
    side, result = ats.select_and_grade([8, 3], [3, 3 + 1e-12], [-3, -3])
    assert np.array_equal(side, [0, 0])
    assert list(result) == ["pass", "pass"]
    s = ats.summary(result)
    assert s["passes"] == 2 and s["selected"] == s["graded"] == 0
    assert np.isnan(s["cover_rate"]) and np.isnan(s["ci_low"])


def test_team_mirroring_keeps_results_and_flips_selections():
    m, f, line = np.array([8, 2, -4, -7]), np.array([6, 1, -2, -9]), np.array([-3, -3, 5, 5])
    side, result = ats.select_and_grade(m, f, line)
    side2, result2 = ats.select_and_grade(-m, -f, -line)
    assert np.array_equal(side, -side2) and np.array_equal(result, result2)


def test_losing_selection_and_bad_inputs():
    side, result = ats.select_and_grade([2, 8], [6, 1], [-3, -3])
    assert list(result) == ["noncover", "noncover"]
    assert ats.summary(result)["cover_rate"] == 0
    with pytest.raises(ValueError, match="finite"):
        ats.select_and_grade([np.nan], [1], [2])
    with pytest.raises(ValueError, match="same-shaped"):
        ats.select_and_grade([1, 2], [1], [2])


def test_timing_labels_separate_future_lines_and_self_reference():
    assert ats.timing("dk_close", "open") == "lookahead_only"
    assert ats.timing("dk_open", "open") == ats.timing("dk_close", "close") == "same_line_all_pass"
    assert ats.timing("dk_open", "close") == "earlier_opening_line"
    assert "not_opening_time_aligned" in ats.timing("home_band_median", "open")
