"""Shared decayed-feature helpers. Legacy fits live in research/legacy_cold_start.py."""
import numpy as np
import pandas as pd
import nba_composite as nc

RHO = 0.25

def cold_features(cur, i, half_life=nc.HALF_LIFE, prior=None, rho=0.0,
                  pre=None, kappa=0.0, orb_pct=False):
    """Decayed four-factor features from [prior season][preseason][current].

    Rows are ordered oldest -> newest and the decay counts games ago across
    the whole sequence, so last season fades as new games arrive. `rho` and
    `kappa` multiply the prior-season and preseason rows. With rho = kappa =
    0 this equals nc.decayed_features(cur, i). Returns None when there is
    nothing to weight.
    """
    parts, mult = [], []
    if prior is not None and rho > 0 and len(prior):
        parts.append(prior[nc.COLS].to_numpy(float))
        mult.append(np.full(len(prior), float(rho)))
    if pre is not None and kappa > 0 and len(pre):
        parts.append(pre[nc.COLS].to_numpy(float))
        mult.append(np.full(len(pre), float(kappa)))
    if i > 0:
        parts.append(cur[nc.COLS].to_numpy(float)[:i])
        mult.append(np.ones(i))
    if not parts:
        return None
    A = np.vstack(parts)
    m = np.concatenate(mult)
    w = 0.5 ** (np.arange(len(A))[::-1] / half_life) * m
    return nc.features_from_totals((A * w[:, None]).sum(0), orb_pct=orb_pct)


def carry_delta(cur_h, ih, prior_h, cur_a, ia, prior_a, weights,
                rho=RHO, half_life=nc.HALF_LIFE):
    """Composite delta (home - away) from carryover features, or None."""
    orb = nc.is_orb_pct(weights)          # features as the weights were fitted
    fh = cold_features(cur_h, ih, half_life, prior=prior_h, rho=rho, orb_pct=orb)
    fa = cold_features(cur_a, ia, half_life, prior=prior_a, rho=rho, orb_pct=orb)
    if fh is None or fa is None:
        return None
    return float(nc.composite(fh - fa, weights["sd"], weights["w"]))
