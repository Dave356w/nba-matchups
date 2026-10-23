# Retire exhibition/Kalshi production and fifteen closed workflows

The build no longer fetches/grades Kalshi or scores exhibitions. The Preseason
page/link is removed, including stale output cleanup. Existing price columns
and the preseason CSV remain validated history; preseason box scores still
supply regular-season game-0 roster inputs. Research scripts/results remain.

Exactly eight workflows remain: build, tests, fit-model, backfill, rebuild,
box-history, history-backtest, margin-model. The live working agreement and
model guidelines lead with v9, three tests and separate stage rows; dated
research/registrations remain in archived docs with result links.

Validation: data files OK; 361 tests passed; all eight YAML files parsed;
page screenshots in `diagnostics/process_retirement/screenshots/`. Tests
prove historical files/columns are preserved, retired sources are not read,
and no exhibition page is published. No data/public files changed.

Head: codex/retire-side-processes. Base: codex/v9-neutral-inputs.
Optional cleanup, intended draft PR; model activation remains a separate gate.
