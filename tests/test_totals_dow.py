import importlib.util
import os
import sys

import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))
sys.path.insert(0, os.path.join(_here, "..", "research"))

_spec = importlib.util.spec_from_file_location(
    "totals_dow", os.path.join(_here, "..", "research", "totals_dow.py"))
dow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dow)


def test_rest_and_day_terms():
    # 2025-11-03 Mon, 11-04 Tue, 11-05 Wed, 11-09 Sun
    rows = pd.DataFrame(dict(season=[2026] * 4,
                             slate_date=["2025-11-03", "2025-11-04", "2025-11-05", "2025-11-09"],
                             home=["AAA", "BBB", "AAA", "AAA"], away=["BBB", "CCC", "CCC", "BBB"]))
    t = dow.add_terms(rows)
    # AAA: first game 3, then 1 day off (11-05), then 3 (capped from 3); BBB: 3, 0 (b2b), 3 (cap)
    # CCC: 3, 0
    assert t["rest_tot"].tolist() == [6.0, 3.0, 1.0, 6.0]
    assert t["dow"].tolist() == ["Mon", "Tue", "Wed", "Sun"]
    assert t["midweek"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert t["sunmon"].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_rule_compares_dow_with_rest():
    assert dow.RULE == "DOW" and dow.CONTROL == "REST"
    assert dow.ARMS["DOW"] == dow.ARMS["REST"] + ["midweek", "sunmon"]
