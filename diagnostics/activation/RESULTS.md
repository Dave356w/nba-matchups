# Simplification activation — 2026-10-23

Items 2, 1 and 3 are integrated into main through
[PR 105](https://github.com/Dave356w/nba-matchups/pull/105), following the
separate registration and implementation PRs 102–104. The final native
snapshot eligibility correction is
[PR 106](https://github.com/Dave356w/nba-matchups/pull/106).

The registration timestamp remains 2026-10-23T19:46:58Z. First snapshots,
including the older v8 snapshots captured for today's slate, remain
immutable. Only v9 snapshots qualify for the new games-1–9 segments:
`first_model_tag` for native grades, `model_tag` for reconstructed hindsight.
The games-10+ rules, thresholds and price filters are unchanged. Today has
no qualifying native v9 early snapshots; the forward segments start with
the first qualifying v9 snapshots, without retroactively relabeling v8.

## Completed checks

| Check | Result | Evidence |
|---|---|---|
| Integration CI | Passed | [Tests 259](https://github.com/Dave356w/nba-matchups/actions/runs/40020639016) |
| Eligibility correction | 362 tests passed; data files OK; 8 existing warnings | [PR CI](https://github.com/Dave356w/nba-matchups/actions/runs/40021342464), [main CI](https://github.com/Dave356w/nba-matchups/actions/runs/40021496437) |
| Historical rescore | 1,225 rows each in 2025 and 2026; all 2,450 scored with v9; validation passed | [Backfill 16](https://github.com/Dave356w/nba-matchups/actions/runs/40020696684) |
| Model review and DraftKings check | Both completed successfully after rescore | Same backfill run; [model review](model_review.txt), [DraftKings](dk_hypotheses.txt) |
| Current-season rebuild | 16/16 games scored by v9; validation passed; output identical to existing rebuilt file | [Rebuild](https://github.com/Dave356w/nba-matchups/actions/runs/40021096202) |
| Retained history backtest | Completed all modern and out-of-era sections | [History backtest 6](https://github.com/Dave356w/nba-matchups/actions/runs/40020759249), [output](history_backtest.txt) |
| Production build and deployment | Render, ledger validation and deployment passed with the final correction | [Build](https://github.com/Dave356w/nba-matchups/actions/runs/40021496409) |
| Live visual checks | Picks, Ledger, Results and Performance inspected | [Screenshots](screenshots/) |

The rescore and rebuild made no changes to their generated CSVs. No native
first snapshot or bot-generated data/public output was hand-committed.

## Review of results and limits

The neutral-input inventory found no fallback cases in the 2,450 historical
and 16 rebuilt scored games ([test-first result](../neutral_fill/BEFORE.txt)).
The paired empirical comparison therefore has n=0: it cannot estimate a
benefit or penalty for the newly supported missing-input cases. Complete
input probabilities keep the existing v9 maths and tags. Missing optional
inputs have distinct neutral v9 tags, and missing core ratings abstain.

Current v9 Model review remains worse than the close on the two large
games-10+ book-seasons: model-minus-close log loss +0.0170 on DraftKings
2025–26 (n=964) and +0.0130 on ESPN BET 2024–25 (n=1,071). Brier differences
are +0.0066 and +0.0054. These are comparisons with the close, not a
regression caused by the neutral-input change. No new edge is claimed.

The DraftKings opening-price hindsight remains H4 n=112, H2·F n=324 and H5
n=696. The new early segments have no DraftKings hindsight games. The 276
ESPN BET early games remain separately labeled information-only hindsight,
with the injury-timing caveat. All stages and evidence bases remain apart.

The retained History backtest explicitly exercises the archived v6 routing,
v4 out-of-era fit and carryover research. It completed with the relocated
research helpers and model files. It is a historical compatibility check,
not a new v9 walk-forward benchmark; v9's current priced outputs are the
Model review and DraftKings checks above. The full output is retained with
its original research labels.

Exactly eight workflow files remain. The old branch-only “Export v6 direct
research inputs” workflow was also disabled in Actions; its runs and
research branch remain available. The deleted closed workflows, research
scripts/results, Kalshi history and preseason CSV history remain preserved
as specified by the plan.

## Live screenshots

Captured after the final deployment on 2026-10-23. Picks correctly show
descriptive spread leans for the older v8 first snapshots, with one primary
probability and margin. Ledger shows six separate active stage rows and
the frozen retired records. Results show separate H5·E and H5 spread tiles;
Performance has no active early spread record before a qualifying grade.

- [Picks cards](screenshots/picks.jpg)
- [Ledger stage rows](screenshots/ledger.jpg)
- [Results](screenshots/results.jpg)
- [Performance](screenshots/performance.jpg)
