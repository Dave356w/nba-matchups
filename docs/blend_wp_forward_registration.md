# Model–market blend shadow forward record — DRAFT

**Status: draft for the owner's review. Not registered.** The registration
timestamp, the ledger commit it cites and the first slate are filled in when
the owner approves. That commit holds this file alone, before any blend code.
Algorithm version: `blend_shrink40_v1`.

## Question

Does a fixed blend of the model's lock probability and the DraftKings lock
price predict winners better than the lock price alone? This tests the owner's
view that the model and the market together hold signal that neither holds
alone (2026-11-13). It is shadow research only. No model, lock, price, grade,
record, card or public page changes.

## Evidence before registration (hindsight; nothing here is forward evidence)

- Opening line, v11 replay, games 1+, blend weight fitted on one season and
  tested on the other: blend minus open log loss −0.0006 ± 0.0066 (2024-25
  ESPN BET → 2025-26 DraftKings) and −0.0048 ± 0.0034 (the reverse), ± 95%.
  Fitted weights 0.64 and 0.36; both seasons together 0.52 ± 0.25.
- Closing line, same rows: weight 0.08 ± 0.26; blend minus close +0.0005 ±
  0.0018 and +0.0002 ± 0.0003. By the close the market has priced the
  model's information.
- Binned matrix cells (Realized WP v2 method) were worse than the market at
  both the open and the close. Model × market interaction terms added nothing
  beyond the plain blend.
- Native 2026-27 at the lock, games 1+ (98 games, information only): fitted
  weight at its −1 bound, model log loss 0.039 ± 0.025 worse than the lock
  price.
- Caveats: the replay P comes from the current fit, whose seasons may include
  the test season. That flatters the model. No historical lock prices exist:
  the five-hour lock falls between the open and the close.

## Estimate (fixed)

    P_blend(home) = σ( logit q_lock + w · (logit P_lock − logit q_lock) ),  w = 0.40

This is `analysis.shrink_p` with `SHRINK_W = 0.40` unchanged. That weight was
fitted walk-forward on the opening line in `research/open_shrinkage.py`
(2026-10-22; +0.43 and +0.36 on the two priced seasons). It was set before
this season and is not re-selected here. It sits between the open's weight
(~0.5) and the close's (~0.1), and the lock falls between them too. P_lock is
the native lock's `lock_p_home`; q_lock is `lock_q_home`, the no-vig DraftKings
price at the same snapshot. There is no intercept, no refit, no per-stage
weight and no update during the record. Away is 1 − home.

## Population (fixed)

- The public model-prediction population, `analysis.model_predictions`:
  native v11-family five-hour locks with the lock tag ending `_v11` and a
  valid lock window. Stale-log void locks are excluded.
- Both lock moneylines and a usable `lock_q_home` (strictly between 0 and 1)
  are required. Pending, postponed and unpriced games do not enter.
- Starts with the first slate on which no game had locked at registration.
  Nothing earlier enters: no backfill.
- Games 1–9 and games 10+ are separate rows, per season. Openers and unknown
  stages are left out. Never pooled with Realized WP v1/v2, the ML record or
  any tracked test.

## Reproducibility

Every input is a write-once lock column of `data/nba_ledger.csv`, and w is a
constant. The blend is therefore fixed once a game locks, so no capture file is
needed. Lock values are never refreshed, rescored or replaced by later
forecasts or closing prices.

## Scoring (fixed)

On identical games, per season × stage:

- The blend, the lock price and the model at the lock: log loss, Brier and
  winner accuracy, plus paired differences (blend minus lock price, blend
  minus model) with ± 95% game-level intervals.
- The closing line beside them for context only. It is later information and
  decides nothing.
- The calibration slope of each, using `research/model_calibration_check.py`'s
  fit.

## Paper return (fixed)

Flat 1 unit at the lock moneyline on the side whose blend probability exceeds
that side's raw break-even probability at the lock price. If both sides do,
take the larger expected value; at most one bet per game. Push or void returns
the stake. Report W/L, units and ROI, each against its no-vig null and the
same-game favourite, ± 1 SE as on the report. Descriptive. The paper return
never becomes a rule without its own registration.

## Reading (fixed before any forward result)

Primary: games 10+, 2026-27 regular season, blend minus lock-price log loss.

- **Supports the blend** if the 95% upper bound is below 0.
- **Contradicts it** if the 95% lower bound is above 0.
- **Inconclusive** otherwise.

Games 1–9, Brier, accuracy and the paper return are reported beside it and
decide nothing. The hindsight gain at the open was about 0.001–0.005 in log
loss. One season of games 10+ (~1,000 games) gives an interval of about ±0.004
to ±0.007, so an inconclusive first season is the expected outcome and is not
evidence against the blend. Any adoption, display or rule needs a separate
owner amendment.

## Implementation (after this registration's own commit)

- `research/blend_forward.py` writes `diagnostics/blend_forward/RESULTS.md`
  from the ledger, using the population, estimate and scoring above.
- Tests: the estimate equals `shrink_p` at w = 0.40. The population matches
  `model_predictions` minus the start-slate cut. Void and off-family locks are
  excluded. Stages are never pooled. No later price replaces a lock value.
- It is not shown on any public page and adds no workflow, data file or ledger
  column.
