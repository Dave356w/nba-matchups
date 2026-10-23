# Public UI review — 2026-10-23

The four public pages are Today, Results, Performance and How it works. Retired hypotheses, reconstructed results and research reports remain in the internal workbench; the Pages upload accepts only the four HTML files and `.nojekyll`.

## Verification

- [Corrected browser and full-suite run](https://github.com/Dave356w/nba-matchups/actions/runs/40026354120): 368 tests passed, data validation passed, desktop and mobile checks passed.
- The previous browser failure exposed an ambiguous Results filter label. Each filter now has an explicit label and matching select ID.
- Screenshots cover all four pages at 1440 × 1000 and 390 × 844. Files containing `fixture` visibly identify synthetic verification data and demonstrate wins, losses, pushes, pending outcomes and overlapping rules. Fixture data never enters the published site.
- Filters, details, both themes and cumulative return charts were exercised. The live-score overlay changed scores and status without changing frozen selections.
- A final CSS adjustment left-aligns Results cells consistently on desktop and mobile; the PR workflow repeats the complete browser review and uploads fresh screenshots.

The native snapshot sample currently has no qualifying registered early-game decisions: its frozen early snapshots use the earlier model. Latest forecast probabilities do not retrospectively turn these snapshots into registered decisions. Zero tracked results is therefore intentional. Forecasts remain visible.

No model coefficients, registration thresholds, snapshot timing or prediction mathematics changed. Native, reconstructed and hindsight results remain separate.
