import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))
sys.path.insert(0, os.path.join(_here, "..", "research"))
sys.path.insert(0, _here)

import nba_composite as nc  # noqa: E402
from test_interaction_weight import _log, _weights  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "od_weights", os.path.join(_here, "..", "research", "od_weights.py"))
od = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(od)


def test_era_leaves_out_the_covid_seasons():
    assert 2020 not in od.ERA and 2021 not in od.ERA
    assert od.SEASONS[0] == 2018 and 2027 not in od.COMPLETED


def test_terms_split_delta_exactly():
    w = _weights(7)
    lh, la = _log(15, 8), _log(15, 9)
    od._vecs.clear()
    x = od.fade_inputs_vecs(lh, 12, None, la, 11, None, 0.36, w)
    want = nc.fade_inputs(lh, 12, None, la, 11, None, 0.36, w)
    assert np.isclose(x["delta"], want["delta"])
    g = pd.DataFrame({"phase": [0.5]})
    t = od.add_terms(g, list(od._vecs), w)
    z = t[od.Z].to_numpy(float)[0]
    assert np.isclose(z @ np.asarray(w["w"]), want["delta"])          # z . w = delta
    assert np.isclose(t["delta_off"].iloc[0], z[:4] @ np.asarray(w["w"][:4]))
    assert np.isclose(t["doff_phase"].iloc[0], 0.5 * t["delta_off"].iloc[0])


def test_arm_terms_and_offence_share():
    base = ["delta", "b2b_net", "d_phase"]
    assert od.arm_feats("OD2", base) == base + ["delta_off", "doff_phase"]
    assert od.arm_feats("GW8", base)[:8] == od.Z and "delta" not in od.arm_feats("GW8", base)
    f = od.arm_feats("OD2", base)
    out = od.implied(f, [2.0, 0.10, 2.0, 0.15, -0.08, 0.0])
    s, k = 0.10 + 0.15, -0.08
    assert np.isclose(out["off_share_1.0"], (s + k) / (2 * s + k))
    assert out["off_share_1.0"] < 0.5
