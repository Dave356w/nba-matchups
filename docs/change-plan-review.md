# Prepared changes for review

All changes are local commits in Dave356w/nba-matchups. No branches have
been published, no PRs opened, no merges performed and no production
activation attempted. Automatic approval review blocked git push because
it requires explicit authorization to publish project documentation to the
public repository, even after the connected owner/admin identity was verified.

| Item | Head branch | Proposed base | Full suite |
|---|---|---|---|
| 2 | codex/first-snapshot-forward-tests | main | 371 passed |
| 1 | codex/v9-neutral-inputs | codex/first-snapshot-forward-tests | 366 passed |
| 3 (optional) | codex/retire-side-processes | codex/v9-neutral-inputs | 361 passed |

Each ran data validation and has screenshots. No bot-generated data/public
files changed. Item 2's registration is its own first commit, `2e301d3`,
2026-10-23T19:46:58Z (12:46:58 PM Pacific), before CHI @ MIL's scheduled tip.

The branches are stacked because removing the carryover model depends on
archiving H1/H6. Each item's diff is separate and each has a prepared PR body:

- [Item 2](prs/item2-first-snapshot-tests.md)
- [Item 1](prs/item1-v9-neutral-inputs.md)
- [Item 3](prs/item3-retire-side-processes.md)

Pending after publication: cached historical rescore, current-season rebuild,
build and History backtest; review their model checks before activation.
The Model review and DraftKings backtest already ran on committed v9 rows.
The neutral-input empirical comparison has zero observed fallback cases;
controlled tests cover the changed paths, without claiming improvement.

Publication target for the three branches and draft PRs:
https://github.com/Dave356w/nba-matchups
