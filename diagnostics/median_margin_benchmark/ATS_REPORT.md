# ATS cover-rate comparison: frozen forecasts, 2025–26

The same 964 matched DraftKings games as the median-margin benchmark. Median bands and mean-conversion coefficients are fitted using 8,438 OOS WP/outcome pairs from seasons before 2025–26. These cover rates are retrospective diagnostics, not an untouched prospective test.

Selection rule: compare the predicted signed home margin with minus the quoted home handicap. Forecast above that margin selects home ATS; below selects away ATS; exact equality passes. Numerical equality tolerance is 1e-10 points. Cover rate = covers / (covers + noncovers); pushes and passes are reported separately. No minimum edge is added or optimized.


## At DraftKings opening lines

| Forecast | Covers–noncovers–pushes | Passes | Cover rate | Wilson 95% interval |
|---|---:|---:|---:|---:|
| Home-band median | 490–473–0 | 1 | 50.88% | 47.73–54.03% |
| Favorite-band median | 493–471–0 | 0 | 51.14% | 47.99–54.29% |
| Fixed 13.5 | 504–460–0 | 0 | 52.28% | 49.13–55.42% |
| Fitted mean scale | 495–469–0 | 0 | 51.35% | 48.19–54.49% |
| Fitted mean intercept + scale | 500–464–0 | 0 | 51.87% | 48.71–55.01% |
| DK opening forecast | 0–0–0 | 964 | — | — |
| DK closing forecast **(LOOK-AHEAD ONLY)** | 427–304–0 | 233 | 58.41% | 54.81–61.93% |

## At DraftKings closing lines

| Forecast | Covers–noncovers–pushes | Passes | Cover rate | Wilson 95% interval |
|---|---:|---:|---:|---:|
| Home-band median | 466–498–0 | 0 | 48.34% | 45.20–51.50% |
| Favorite-band median | 470–494–0 | 0 | 48.76% | 45.61–51.91% |
| Fixed 13.5 | 475–489–0 | 0 | 49.27% | 46.13–52.43% |
| Fitted mean scale | 482–482–0 | 0 | 50.00% | 46.85–53.15% |
| Fitted mean intercept + scale | 489–475–0 | 0 | 50.73% | 47.57–53.87% |
| DK opening forecast | 340–391–0 | 233 | 46.51% | 42.92–50.14% |
| DK closing forecast | 0–0–0 | 964 | — | — |

## Timing and interpretation

- **The closing forecast at the opening line uses future information.** Its result is a look-ahead diagnostic only and must not be ranked as an available opening-time strategy.
- The opening forecast at its own line, and the closing forecast at its own line, have zero forecast disagreement on every game. All 964 pass; a 50% rate must not be invented for them.
- Opening and closing lines differ on 731 games and match on 233. The opening-line forecast evaluated at the closing line therefore has 731 selections and 233 passes.
- Reconstructed model WP is pregame but has no archived opening-time snapshot. Opening-line model rates are not timestamp-aligned opening-time performance. Closing-line comparisons also remain reconstructed historical research, not native live records.
- All recorded lines in this sample are half points and final margins are integer points; there are no outcome pushes. The grader nevertheless handles pushes and passes independently.
- Lower margin MAE and higher ATS cover rate are different objectives. The median forecasts' lower average home bias does not establish higher ATS accuracy.
- Wilson intervals treat graded games as independent Bernoulli observations; they do not account for same-date/serial dependence, model development, multiple comparisons, or varying selection samples. Cover rate alone does not establish profitability; prices are not evaluated.

## Frozen mean-conversion coefficients

The two fitted mean mappings supplement the five forecasts in the earlier median benchmark. They use the same prior-season-only protocol from the original WP-to-MOV study.

| Mapping | Training games | Last training BBR year | Intercept | Scale |
|---|---:|---:|---:|---:|
| Fitted mean scale | 8438 | 2025 | +0.0000 | 11.1796 |
| Fitted mean intercept + scale | 8438 | 2025 | -0.4331 | 11.4588 |

## Reproduction

```bash
python research/median_ats_comparison.py --out /tmp/median-ats-comparison
```

Outputs: `ATS_REPORT.md`, `ats_cover_rates.csv`, `ats_mean_coefficients.csv`, and `ats_predictions.csv.gz`. The per-game output includes target line, forecast, selected side, result, and timing label. The underlying median lookup values and data audit are in [REPORT.md](REPORT.md). Production code, models and ledgers are unchanged.

Source commit: `bd05c099f7b3a6ce62e3e1ec4cde59828ff87a70`.
