# Realized WP season-phase shrinkage test — registration

Registered 2026-11-16T21:30Z on the owner's instruction ("allow espn bet and
test for implementation over current method"), committed on its own before
any test code is written or run. Research only: nothing here changes the
card, Realized WP v4, its shadow record, the ROI matrix, the model, locks,
grades or any record. Implementation, if the test passes, needs its own
owner decision and its own registration.

## Question

Does a model-WP × market-WP cell estimate shrunk in three steps (the price,
then the previous season's same season phase, then this season) score better
than the production Realized WP v4 (shrunk latest-200 cell + pooled drift,
docs/realized_wp_forward_registration_v4.md)?

## Owner's book decision

The previous season has no DraftKings prices before about game 15 of 2025-26.
For this test the owner allows ESPN BET closing prices wherever the
DraftKings replay is missing: the 2024-25 season (ESPN BET only) and 2025-26
before DraftKings coverage begins. Each pick is measured against its own
book's no-vig price (gap = won − q), and each game appears once (the two
2025-26 books cover disjoint games). This is a research allowance; it does
not change which book any record, page or rule reads.

## Data

One stream of v11 picks, the model's selected team, ordered by slate date
then game ID, void locks excluded:

- 2024-25: reconstructed v11 replay at the ESPN BET close.
- 2025-26: reconstructed v11 replay at the ESPN BET close where it exists,
  otherwise at the DraftKings close.
- 2026-27: valid native v11 five-hour DraftKings locks.

Rows are built by `research/matrix_window_backtest.picks` (model pick,
no-vig q, won, cell). Cells are `wp_calibration.MODEL_BOUNDS` ×
`MARKET_BOUNDS`. Only strictly earlier slates enter any estimate.

## Phases

By the smaller of the two teams' games played before the game, fixed now:
**1–9, 10–29, 30–59, 60+**. Openers (game 0) are not scored by any arm.

## Estimators

Each estimate is clip(q_game + adjustment, 0, 1). With gap = won − q:

- **price**: adjustment 0.
- **v4** (production): as `research/shrunk_drift_backtest.py`, on this same
  stream (cells over the previous and current season, latest 200, × n/(n+200);
  pooled drift, half-life 100, 200 pseudo-games, latest 600; openers in their
  own cells and in the drift).
- **P1**: g1 = Σ gap / (n + k1) over the previous season's games in this
  game's phase, all cells.
- **P2**: g2 = g1 + Σ (gap − g1) / (n + k2) over the previous season's games
  in this phase and this cell.
- **PS** (primary candidate): g3 = g2 + Σ (gap − g2) / (n + k3) over this
  season's earlier slates in this phase and this cell.

## Shrinkage constants, fixed before scoring

Estimated once by method of moments on 2024-25 games 1+ (no game of a scored
season enters), then frozen for both scored seasons:

- σ² = mean q(1 − q).
- τ1² = mean over phases of (ḡ_phase² − σ²/n_phase): spread of phase gaps
  around the price. k1 = σ²/τ1².
- τ2² = [Σ_cells n_c (ḡ_c − ḡ_phase)² − C σ²] / Σ n_c over the C non-empty
  phase × cell groups: spread of cells around their phase. k2 = k3 = σ²/τ2².
- Each k is clipped to 50–2000; a non-positive τ² gives 2000.

## Scored cohorts

- **Primary:** 2025-26 games 1+ (previous season 2024-25).
- **Information only:** 2026-27 native locks, games 1+ (previous season
  2025-26), too few games for a verdict.

## Measures and pre-set reading

Brier and log loss (probabilities clipped to 0.01–0.99 for log loss) minus
the price on identical games, and the paired PS − v4 difference, game-level
± 95%. Mean and largest absolute adjustment.

**Pass:** on the primary cohort, the PS − v4 paired Brier difference has its
95% upper bound below 0, and the PS − v4 log-loss point estimate is ≤ 0.
Otherwise the test fails and v4 stays. P1 and P2 are reported as diagnostics
and cannot pass in PS's place. A pass is grounds for the owner to consider an
implementation amendment, not an implementation.

## Stated in advance

- v4's constants were chosen in-sample on 2025-26, which favours v4 here.
- The phase idea was prompted by the 2026-27 early-season and 2025-26
  phase figures seen on 2026-11-16 (outcome-prompted). Exploratory phase
  variants tried on 2026-11-13 did not beat v4 (v4 registration).
- Replay P comes from leave-one-season-out fits; closing prices are later
  than a five-hour lock; the prior crosses books where noted.

Script: `research/phase_shrink_backtest.py`. Output:
`diagnostics/phase_shrink/RESULTS.txt`.
