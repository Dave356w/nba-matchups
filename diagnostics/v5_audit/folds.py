#!/usr/bin/env python3
"""Build the v5 audit's walk-forward folds (common.build_folds) and check the
fast season builder against nc.build_games.

  python diagnostics/v5_audit/folds.py --seasons 2017 2018 2019 2021 ... 2026
"""
import argparse

import common as c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int,
                    default=[2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--verify", type=int, default=2026,
                    help="season to check against nc.build_games (0 = skip)")
    a = ap.parse_args(argv)
    if a.verify:
        v = c.verify(a.verify)
        c.log(f"verify {a.verify} vs nc.build_games (production weights): {v}")
        bad = [k for k in ("delta", "luck_def", "talent_diff", "d_phase", "b2b_net")
               if not (v[k] < 1e-6)]
        if bad or v["n"] != v["n_b"]:
            raise SystemExit(f"fast builder disagrees with nc.build_games: {bad} {v}")
    folds = c.build_folds(a.seasons)
    for y, f in folds.items():
        m = c.with_market(f.test)
        for lab, g in c.books(m):
            pv = g["p_v5"].to_numpy(float)
            ok = (pv == pv)
            c.log(f"fold {y} {lab}: n={int(ok.sum())} v5 logloss "
                  f"{c.ll(pv[ok], g['win'].to_numpy(float)[ok]).mean():.4f}  close "
                  f"{c.ll(g['close_q_home'].to_numpy(float)[ok], g['win'].to_numpy(float)[ok]).mean():.4f}")


if __name__ == "__main__":
    main()
