import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import nba_composite as nc  # noqa: E402
import kalshi  # noqa: E402
import player_availability as pav  # noqa: E402
import sleeper  # noqa: E402


@pytest.fixture(autouse=True)
def no_kalshi_network(monkeypatch):
    """Kalshi answers nothing unless a test supplies its own `get`: the build
    no longer fetches Kalshi, but research helpers still can, and tests never
    touch the network."""
    monkeypatch.setattr(kalshi, "get", lambda path, **params: {})


@pytest.fixture(autouse=True)
def no_sleeper_network(monkeypatch):
    """Sleeper projects nothing unless a test supplies its own: the daily
    build snapshots Sleeper's projections, and tests never touch the
    network."""
    monkeypatch.setattr(sleeper, "fetch_state",
                        lambda timeout=30: {"season": "2026", "week": 1})
    monkeypatch.setattr(sleeper, "fetch_projections",
                        lambda season, week, timeout=60: [])


@pytest.fixture(autouse=True)
def no_preseason_network(monkeypatch):
    """ESPN lists no preseason games unless a test supplies its own: the
    daily build fetches this season's preseason box scores (v8's game-0
    rosters), and tests never touch the network or write to data/ (an empty
    fetch is never saved)."""
    monkeypatch.setattr(pav, "preseason_rows",
                        lambda dates, sleep=0.15: (pd.DataFrame(columns=pav.PRE_COLUMNS), 0))


def make_log(n, start="2026-10-21", seed=0, gap_days=2, strength=0.0):
    """Synthetic regular-season game log with plausible raw totals."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq=f"{gap_days}D")
    rows = []
    for i, d in enumerate(dates):
        r = {"date": d, "home": bool(i % 2), "opp": "XXX"}
        for side, s in (("T", strength), ("O", -strength)):
            fga = rng.integers(84, 92)
            fg = int(fga * (0.46 + 0.02 * s + rng.normal(0, 0.02)))
            r[side + "FGA"] = fga
            r[side + "FG"] = fg
            r[side + "3P"] = rng.integers(10, 16)
            r[side + "FTA"] = rng.integers(18, 26)
            r[side + "FT"] = int(r[side + "FTA"] * 0.78)
            r[side + "ORB"] = rng.integers(8, 13)
            r[side + "DRB"] = rng.integers(30, 36)
            r[side + "TOV"] = rng.integers(11, 16)
        r["pts"] = 2 * r["TFG"] + r["T3P"] + r["TFT"]
        r["opp_pts"] = 2 * r["OFG"] + r["O3P"] + r["OFT"]
        rows.append(r)
    return pd.DataFrame(rows)[["date", "home", "opp", "pts", "opp_pts"] + nc.COLS]


@pytest.fixture
def weights():
    return {"sd": [2.0, 1.2, 1.5, 2.5, 2.0, 1.2, 1.5, 2.5],
            "w": [0.09, 0.05, 0.03, 0.02, 0.08, 0.04, 0.03, 0.02]}


@pytest.fixture
def model():
    return {"features": ["delta", "b2b_net"], "intercept": 0.241,
            "coef": [0.0374, 0.327], "half_life": 25.0}


def write_research(*args, **kwargs):
    """Render the backend workbench into the test's dedicated output folder."""
    import build_site
    return build_site.write_research(*args, **kwargs, out_dir=build_site.OUT_DIR)
