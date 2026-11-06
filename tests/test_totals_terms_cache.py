"""The stored report terms (model/totals_report_terms.csv): Fit model reads
completed seasons from the file instead of refetching every NBA report."""
import pandas as pd
import pytest

import totals

Q = {"300": {"Questionable": 0.6, "Doubtful": 0.2, "Probable": 0.9}}


def _compute(calls):
    def f(y, lead, q):
        calls.append(y)
        return {(f"{y - 1}-11-01", "AAA", "BBB"): dict(av_o=1.0 * y, av_d=-0.5, av_m=2.0)}
    return f


def test_completed_seasons_are_computed_once_then_read(tmp_path):
    path, calls = tmp_path / "t.csv", []
    first = totals.report_terms([2025, 2026], Q, path=path, compute=_compute(calls),
                                today="2026-11-06", log=lambda m: None)
    assert calls == [2025, 2026] and path.exists()
    again = totals.report_terms([2025, 2026], Q, path=path, compute=_compute(calls),
                                today="2026-11-06", log=lambda m: None)
    assert calls == [2025, 2026]                             # nothing fetched the second time
    assert again == first == {("2024-11-01", "AAA", "BBB"): dict(av_o=2025.0, av_d=-0.5, av_m=2.0),
                              ("2025-11-01", "AAA", "BBB"): dict(av_o=2026.0, av_d=-0.5, av_m=2.0)}


def test_the_current_season_is_never_stored(tmp_path):
    path, calls = tmp_path / "t.csv", []
    totals.report_terms([2027], Q, path=path, compute=_compute(calls), today="2026-11-06",
                        log=lambda m: None)
    totals.report_terms([2027], Q, path=path, compute=_compute(calls), today="2026-11-06",
                        log=lambda m: None)
    assert calls == [2027, 2027] and not path.exists()


def test_changed_inputs_recompute_and_replace_the_season(tmp_path):
    path, calls = tmp_path / "t.csv", []
    totals.report_terms([2026], Q, path=path, compute=_compute(calls), today="2026-11-06",
                        log=lambda m: None)
    q2 = {"300": dict(Q["300"], Questionable=0.5)}           # a new play-rate table
    assert totals.terms_key(q2) != totals.terms_key(Q)
    totals.report_terms([2026], q2, path=path, compute=_compute(calls), today="2026-11-06",
                        log=lambda m: None)
    assert calls == [2026, 2026]
    stored = pd.read_csv(path, dtype={"key": str})
    assert len(stored) == 1 and stored["key"].iloc[0] == totals.terms_key(q2)   # stale row replaced


def test_the_committed_file_matches_the_deployed_play_rates():
    """The stored terms carry the fingerprint of model/logit_avail.json's
    q_rates at the 300-minute lead, for every report season of the fit."""
    import json
    if not totals.TERMS_PATH.exists():
        pytest.skip("no stored terms")
    q = json.loads((totals.MODEL_PATH.parent / "logit_avail.json").read_text())["q_rates"]
    d = pd.read_csv(totals.TERMS_PATH, dtype={"key": str})
    assert set(d["key"]) == {totals.terms_key(q)} and set(d["lead"]) == {totals.REPORT_LEAD}
    assert set(d["season"]) == set(totals._years(totals.REPORT_YEARS))
    assert not d.duplicated(["date", "home", "away"]).any()
