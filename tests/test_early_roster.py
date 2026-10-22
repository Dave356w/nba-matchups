import importlib.util
import os
import sys

import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "..", "research"))
_spec = importlib.util.spec_from_file_location(
    "early_roster", os.path.join(_here, "..", "research", "early_roster.py"))
er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(er)


def box(rows):
    return pd.DataFrame(rows, columns=["team", "date", "player_id", "minutes"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def test_returning_share_uses_last_season_minutes_and_the_first_box():
    prev = box([("BOS", "2025-03-01", "a", 2000.0), ("BOS", "2025-03-01", "b", 1000.0),
                ("BOS", "2025-03-01", "c", 1000.0), ("BOS", "2025-03-02", "x", 0.0),
                ("NYK", "2025-03-01", "n", 500.0)])
    cur = box([("BOS", "2025-10-21", "a", 30.0), ("BOS", "2025-10-21", "c", 0.0),
               ("BOS", "2025-10-21", "new", 25.0),
               ("BOS", "2025-10-23", "b", 30.0),          # back later: not on the first box
               ("NYK", "2025-10-22", "z", 40.0),
               ("MIA", "2025-10-22", "m", 40.0)])          # no last-season minutes
    s = er.returning_share(prev, cur)
    assert s["BOS"] == 3000.0 / 4000.0            # a and c (listed, DNP counts)
    assert s["NYK"] == 0.0 and "MIA" not in s
