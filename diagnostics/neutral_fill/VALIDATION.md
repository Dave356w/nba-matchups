# Item 1 validation and activation

The pre-change inventory was committed first (`19af1d9`): 2,450 reconstructed
and 16 rebuilt scored rows, zero fallback cases. The paired empirical comparison
has n=0, so no improvement is claimed. The deployed v9 fits are unchanged.

Controlled tests cover neutral talent at game 0, games 1–9 and games 10+;
jointly neutral missing injury terms; core-rating abstention; live/backfill
agreement; no extra fallback fits; CLI/site agreement; and preservation of
historical carryover columns. Obsolete write/fallback tests were removed,
not skipped. Legacy research code and frozen fits remain under research/.

- Data validation: passed.
- Full suite: 366 passed, 8 existing warnings.
- Model review and DraftKings backtest: ran on the committed v9 rows;
  updated results in their existing diagnostics directories.
- Screenshots: Model and Picks under screenshots/; formula verified as
  margin fitted by least squares, P = Phi(margin / sigma).

Not activated: no production data or model files were hand-rewritten, and
no main branch merge or workflow dispatch occurred. The complete historical
rescore/rebuild and History backtest require the repository's cached Actions
environment (local BBR/injury-report caches are absent). Required sequence
on this branch after publication: Backfill history, seasons 2025 2026,
rescore=true; Rebuild season for 2027; build; History backtest. Backfill now
runs Model review and the DraftKings backtest after rescore. Check all
results before activation. Native first snapshots must never be rescored.

This branch depends on item 2 because H1/H6 must be archived before their
supporting production model is removed. Push/PR creation remains blocked
by automatic approval review.
