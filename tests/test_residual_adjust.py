"""Rolling 10-day residual signal (research/residual_adjust.py): window,
no lookahead and the sides / totals formulas."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research"))
import residual_adjust as ra  # noqa: E402


def frame(rows):
    return pd.DataFrame(rows, columns=["season", "slate_date", "home", "away", "e"])


def test_window_is_d_minus_10_to_d_minus_1_same_season_only():
    df = frame([
        (2026, "2025-11-01", "A", "X", 50.0),   # D-11: outside
        (2026, "2025-11-02", "A", "X", 4.0),    # D-10: inside
        (2026, "2025-11-11", "Y", "A", -2.0),   # D-1, A away: +2 from A's side
        (2026, "2025-11-12", "A", "B", 99.0),   # the target: same day, never read
        (2026, "2025-11-12", "C", "A", 99.0),   # same day, other game: never read
        (2027, "2025-11-05", "B", "Z", 7.0),    # other season: never read
    ])
    rh, rw, nh, nw = ra.rolling_residual(df, df["e"], sides=True)
    i = 3
    assert nh[i] == 2 and rh[i] == pytest.approx((4.0 + 2.0) / 2)
    assert nw[i] == 0 and rw[i] == 0.0          # B: no same-season game in the window


def test_totals_residual_is_not_sign_flipped_and_nan_is_skipped():
    df = frame([
        (2026, "2025-11-05", "A", "X", 6.0),
        (2026, "2025-11-07", "Y", "A", 2.0),
        (2026, "2025-11-08", "A", "Z", np.nan),  # no result: skipped
        (2026, "2025-11-09", "A", "B", 0.0),
    ])
    rh, rw, nh, nw = ra.rolling_residual(df, df["e"], sides=False)
    assert nh[3] == 2 and rh[3] == pytest.approx(4.0)


def test_adjustments_halve_each_team_residual():
    m, p = ra.adjust_sides([2.0], [13.0], [4.0], [-2.0])
    assert m[0] == pytest.approx(2.0 + 0.5 * (4.0 - -2.0))
    assert p[0] == pytest.approx(ra.PHI(5.0 / 13.0))
    assert ra.adjust_totals([220.0], [4.0], [-2.0])[0] == pytest.approx(221.0)
