# Realized WP band-adjusted shadow forward record — v3 (latest 200)

Registered 2026-11-13T17:02:13Z, on the owner's instruction to "update realized WP estimate
with best researched n", committed on its own before any v3 code or capture.
Algorithm version: `realized_wp_rolling200_bandadj_v3`. It supersedes
`realized_wp_rolling50_bandadj_v2` (docs/realized_wp_forward_registration_v2.md)
for new captures and for the card display. Everything not restated here is
unchanged from v2: the band-adjusted estimate, cells, seeding, pooling,
exclusions, ordering, capture, the official observation, frozen fields,
scoring, the +EV flat 1u paper rule, benchmarks, scope and persistence.

## Why

The matrix window backtest (`research/matrix_window_backtest.py`,
diagnostics/matrix_window/RESULTS.txt, reading fixed before its run) scored
windows of 10 to 300 and all, walk-forward, as Realized WP v2 against the
price alone. On its primary cohort (2025-26 DraftKings replay at the close,
963 games) the lowest Brier difference was window **200** (+0.0024 ± 0.0022,
95%); latest 50 was +0.0051 ± 0.0035. No window beat the price at 95%: the
larger window is chosen as the least-bad and steadiest (slate-to-slate jitter
0.4 points against 0.8 at 50), not as evidence that the cell predicts. This
selection reads replay outcomes, so it is in-sample to that backtest; the v3
forward record is its out-of-sample test.

## Estimate

    Realized WP = clip( q_game + wins/games − mean(q_member), 0, 1 )

over the **latest 200** results in the joint model-WP × market-WP cell
(previously 50); all available results when a cell has fewer than 200; an
empty cell has no estimate. No shrinkage, pseudo-games, thresholds or tuning.

## Record

- v2 closes when v3 is deployed (production keeps saving v2 until then). At
  registration the shadow file holds v2 captures for the 2026-11-12 slate and
  54 pre-lock captures for the 2026-11-13 slate; the native ledger at
  `d568a65` has no lock on the 2026-11-13 slate (first lock window opens
  2026-11-13T19:00Z). v2 and v1 captures stay unchanged, graded under their
  own versions and never pooled with v3. Nothing is rescored or converted.
- v3 captures start with the first production build after deployment. No
  backfill, including for games that already hold a v2 observation; such a
  game may also receive its own v3 observation from a valid capture before tip.
- The snapshot schema is unchanged; each capture's evidence records window 200.

Shadow research only. The public card and guide show the v3 estimate under the
same name. The ROI matrix's own window switch (Latest 50 default), the model,
locks, prices, grades and public records are unchanged.
