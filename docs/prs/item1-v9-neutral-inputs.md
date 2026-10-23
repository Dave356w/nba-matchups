# Keep v9 when optional inputs are missing

The primary v9 scorer and historical reconstruction now neutral-fill missing
optional terms instead of loading carryover/v4 models. A distinct tag identifies
only the changed cases. Complete-input predictions keep their old tags/maths;
core missing ratings still abstain. New rows no longer write carryover P, while
historical columns remain readable and unchanged. Legacy research keeps its
archived fits/helpers outside the deployed model directory.

The fallback inventory was committed first (`19af1d9`): zero fallback cases
among 2,450 reconstructed and 16 rebuilt scored rows. Paired empirical n=0;
no accuracy improvement is claimed. Controlled regressions cover stage handling,
live/backfill agreement, neutral injuries, CLI agreement and history preservation.
Implementation: `ddf9ebb`.

Validation: data files OK; 366 tests passed; Model review and DraftKings
backtest run on the committed v9 rows. Screenshots/activation details in
`diagnostics/neutral_fill/`. Full historical rescore, season rebuild and
History backtest are pending in the cached repository workflow environment.
This must remain a draft until activation checks pass. No native snapshots
or bot-generated data were rescored locally.

Head: codex/v9-neutral-inputs. Base: codex/first-snapshot-forward-tests.
Depends on item 2's retirement of H1/H6. Intended draft PR.
