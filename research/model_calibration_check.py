#!/usr/bin/env python3
"""Model calibration check: is the margin model's probability too timid, too
confident or home-leaning, and does it add anything to the DraftKings close?

  python research/model_calibration_check.py

Owner request (2026-11-13), after a one-off sanity check found the v11 model's
2026-27 probabilities compressed toward 50% (calibration slope ~1.8 on games
1-9). This makes that check repeatable from the committed data, so later
readings are quoted from one fixed script.

Populations, never pooled (CLAUDE.md: native, reconstructed and rebuilt are
separate evidence; book x season x basis x stage separate):

  native lock     analysis.model_predictions: native v11 five-hour locks from
                  the family slate, void locks excluded (the public record)
  native first    native first snapshots tagged _v11
  rebuilt         this season's hindsight rebuild (p_home), never pregame
  reconstructed   v11 hindsight replay (p_home), by closing book x season

Each population is split by stage: games 1-9 and games 10+ (openers and
unknown stages are left out). Rows need a result and a usable closing pair
(no-vig q strictly between 0 and 1, both moneylines).

Per cell, on the same games:
  dispersion   SD of logit(P) beside SD of logit(q) (ratio < 1 = model
               probabilities closer to 50% than the market's)
  calibration  logistic slope and intercept of the result on logit(P)
               (slope 1 / intercept 0 = calibrated; slope > 1 = too timid);
               the market's own slope beside it
  joint        logistic weights on logit(P) and logit(q) fitted together: a
               model weight indistinguishable from 0 means the close already
               prices what the model knows
  scores       log loss and Brier, model minus close (negative = model better)
  home lean    mean P(home) and mean q(home) minus the home win rate

Descriptive monitoring only: no reading here selects a rule, a filter or a
recalibration. CLAUDE.md keeps early-season calibration and home lean as
"watch without fitting to thin data"; a change to sigma or P is the owner's
decision through an amendment, with a new model tag and the post-change
checks (research/calibration_stretch.py found a walk-forward stretch adds
nothing on games 10+). Nothing here touches the ledgers, the pages or
model/*.json. Writes diagnostics/model_calibration/RESULTS.txt. ± is 95%
(1.96 × SE): game-level paired SE for score differences, the logistic fit's
SE for slopes and weights.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import analysis  # noqa: E402
import ledger    # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "model_calibration", "RESULTS.txt")
Z95 = 1.96
MIN_N = 30          # below it a cell reports counts and scores, no fits
STAGES = (("games 1-9", "early"), ("games 10+", "established"))


def season_label(season):
    s = int(season)
    return f"{s - 1}-{str(s)[2:]}"


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def logistic_fit(X, y, iters=50):
    """(coefficients, SEs) of a logistic regression of y on [1, X] by
    Newton's method; NaNs if it fails to converge or separates."""
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    nan = np.full(X.shape[1], np.nan)
    if len(y) < MIN_N or y.min() == y.max():
        return nan, nan
    b = np.zeros(X.shape[1])
    try:
        for _ in range(iters):
            mu = 1 / (1 + np.exp(-X @ b))
            H = X.T @ (X * (mu * (1 - mu))[:, None])
            step = np.linalg.solve(H, X.T @ (y - mu))
            b += step
            if np.abs(step).max() < 1e-10:
                break
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return nan, nan
    if not np.isfinite(b).all() or np.abs(b).max() > 50:
        return nan, nan
    return b, np.sqrt(np.diag(cov))


def stage_of(g):
    """'opener', 'early', 'established' or 'unknown' by the fewer games
    either team has played (analysis.model_predictions' definition)."""
    gp = np.fmin(pd.to_numeric(g["gp_home"], errors="coerce"),
                 pd.to_numeric(g["gp_away"], errors="coerce"))
    return pd.Series(np.select([gp.eq(0), gp.between(1, 9), gp.ge(10)],
                               ["opener", "early", "established"], default="unknown"),
                     index=g.index)


def usable(g, p_col):
    """Graded rows with a model P and a usable DraftKings-style closing pair,
    as columns P (home), q (home), y (home won), stage."""
    if g is None or not len(g):
        return pd.DataFrame(columns=["P", "q", "y", "stage"])
    p = pd.to_numeric(g[p_col], errors="coerce")
    q = pd.to_numeric(g["close_q_home"], errors="coerce")
    y = pd.to_numeric(g["home_won"], errors="coerce")
    ok = (p.between(0, 1, inclusive="neither") & q.between(0, 1, inclusive="neither")
          & y.isin([0, 1]) & pd.to_numeric(g["close_home_ml"], errors="coerce").notna()
          & pd.to_numeric(g["close_away_ml"], errors="coerce").notna())
    return pd.DataFrame({"P": p[ok], "q": q[ok], "y": y[ok].astype(float),
                         "stage": stage_of(g.loc[ok])})


def cell_stats(c):
    """Every figure for one population x stage cell (see the module doc)."""
    P, q, y = c["P"].to_numpy(float), c["q"].to_numpy(float), c["y"].to_numpy(float)
    n = len(y)
    out = dict(n=n)
    if not n:
        return out
    lp, lq = logit(P), logit(q)

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d_ll, d_br = ll(P) - ll(q), (P - y) ** 2 - (q - y) ** 2
    se = (lambda d: float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"))
    out.update(
        sd_p=float(lp.std(ddof=1)) if n > 1 else float("nan"),
        sd_q=float(lq.std(ddof=1)) if n > 1 else float("nan"),
        d_ll=float(d_ll.mean()), d_ll_se=se(d_ll),
        d_br=float(d_br.mean()), d_br_se=se(d_br),
        home_rate=float(y.mean()),
        lean_p=float(P.mean() - y.mean()), lean_q=float(q.mean() - y.mean()),
        acc_p=float(((P >= 0.5) == (y == 1)).mean()),
        acc_q=float(((q >= 0.5) == (y == 1)).mean()),
    )
    b, s = logistic_fit(lp[:, None], y)
    out.update(int_p=b[0], int_p_se=s[0], slope_p=b[1], slope_p_se=s[1])
    b, s = logistic_fit(lq[:, None], y)
    out.update(slope_q=b[1], slope_q_se=s[1])
    b, s = logistic_fit(np.column_stack([lp, lq]), y)
    out.update(w_p=b[1], w_p_se=s[1], w_q=b[2], w_q_se=s[2])
    return out


def populations(native, recon, rebuilt):
    """[(label, usable rows)] in report order; each label is one evidence
    basis (and, for reconstructed, one closing book x season)."""
    out = []
    lock = analysis.model_predictions(native)
    for s, part in (lock.groupby("season") if len(lock) else []):
        out.append((f"native {season_label(s)}, 5 h lock (public record)",
                    usable(part, "lock_p_home")))
    if native is not None and len(native):
        first = native[native["basis"].astype(str).eq("native")
                       & native["first_model_tag"].astype(str).str.endswith("_v11")]
        for s, part in first.groupby("season"):
            out.append((f"native {season_label(s)}, first snapshot (_v11)",
                        usable(part, "first_p_home")))
    if rebuilt is not None and len(rebuilt):
        for s, part in rebuilt.groupby("season"):
            out.append((f"rebuilt {season_label(s)} (hindsight)", usable(part, "p_home")))
    if recon is not None and len(recon):
        for (s, book), part in recon.groupby(["season", "close_book"]):
            out.append((f"reconstructed {season_label(s)} {book} (hindsight)",
                        usable(part, "p_home")))
    return out


def _pm(v, se, fmt="{:+.2f}"):
    if not np.isfinite(v):
        return "--"
    return (fmt.format(v) + " ± " + fmt.format(Z95 * se).lstrip("+")) if np.isfinite(se) \
        else fmt.format(v)


def flags(s):
    """Plain-language notes where a 95% interval excludes the calibrated or
    market-equal value. Descriptive; they decide nothing."""
    notes = []
    if np.isfinite(s.get("slope_p", np.nan)) and np.isfinite(s.get("slope_p_se", np.nan)):
        lo, hi = s["slope_p"] - Z95 * s["slope_p_se"], s["slope_p"] + Z95 * s["slope_p_se"]
        if lo > 1:
            notes.append("too timid (slope > 1)")
        elif hi < 1:
            notes.append("too confident (slope < 1)")
    if s.get("n", 0) > 1 and np.isfinite(s.get("d_ll_se", np.nan)):
        if s["d_ll"] - Z95 * s["d_ll_se"] > 0:
            notes.append("worse than the close")
        elif s["d_ll"] + Z95 * s["d_ll_se"] < 0:
            notes.append("better than the close")
    if np.isfinite(s.get("w_p", np.nan)) and np.isfinite(s.get("w_p_se", np.nan)) \
            and s["w_p"] - Z95 * s["w_p_se"] > 0:
        notes.append("adds to the close")
    return notes


def render(pops):
    lines = [
        "Model calibration check (research/model_calibration_check.py). Descriptive monitoring;",
        "never a rule, filter or recalibration. ± is 95%. Bases, books, seasons and stages are never pooled.",
        "slope: result on logit(P), 1 = calibrated, > 1 = too timid. joint: weights on logit(P) and",
        "logit(close q) fitted together (model weight ~0 = the close already prices it).",
        "ΔLL / ΔBrier: model minus close, negative = model better. lean: mean P(home) − home win rate.",
        "",
    ]
    for label, rows in pops:
        lines.append(f"== {label}")
        for name, key in STAGES:
            s = cell_stats(rows[rows["stage"] == key])
            n = s["n"]
            if not n:
                lines.append(f"  {name:10s} n=    0")
                continue
            head = f"  {name:10s} n={n:5d}"
            if n > 1 and s["sd_q"] > 0:
                head += (f"  sd logit P {s['sd_p']:.2f} vs close {s['sd_q']:.2f}"
                         f" (ratio {s['sd_p'] / s['sd_q']:.2f})")
            lines.append(head)
            if np.isfinite(s.get("slope_p", np.nan)):
                lines.append(
                    f"             slope model {_pm(s['slope_p'], s['slope_p_se'], '{:.2f}')}"
                    f"  intercept {_pm(s['int_p'], s['int_p_se'])}"
                    f"  | slope close {_pm(s['slope_q'], s['slope_q_se'], '{:.2f}')}")
                lines.append(
                    f"             joint  model {_pm(s['w_p'], s['w_p_se'])}"
                    f"  close {_pm(s['w_q'], s['w_q_se'])}")
            else:
                lines.append(f"             fits need {MIN_N}+ games with both outcomes")
            lines.append(
                f"             ΔLL {_pm(s['d_ll'], s['d_ll_se'], '{:+.4f}')}"
                f"  ΔBrier {_pm(s['d_br'], s['d_br_se'], '{:+.4f}')}"
                f"  | accuracy model {s['acc_p']:.1%} close {s['acc_q']:.1%}")
            lines.append(
                f"             home won {s['home_rate']:.1%}; lean model {s['lean_p']:+.3f}"
                f" close {s['lean_q']:+.3f}")
            f = flags(s)
            if f:
                lines.append("             note: " + "; ".join(f))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    native = ledger.load(os.path.join(ROOT, ledger.NATIVE_PATH))
    recon = ledger.load(os.path.join(ROOT, ledger.RECON_PATH))
    rebuilt = ledger.load(os.path.join(ROOT, ledger.REBUILT_PATH))
    text = render(populations(native, recon, rebuilt))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
