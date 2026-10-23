# Item 3 validation

- `python validate_data_files.py`: passed, including historical Kalshi and preseason data.
- `python -m pytest tests/ -q`: 361 passed, 8 existing warnings; no skips added.
- Eight workflow YAML files parsed successfully; exactly the requested eight remain.
- Tests cover an exhibition-only build that never reads or rewrites the historical
  preseason file, never calls Kalshi, removes a stale Preseason page, and publishes
  no retired navigation link. Regular-season game-0 preseason-box inputs remain.
- Historical exchange columns survive refresh and grading; new exchange columns
  stay blank. Kalshi grading/pregame columns are outside production write sets.
- Screenshots reviewed: Picks, Ledger (active and retired sections), Results,
  Performance, Model and Calibration. External font/logo requests were blocked
  for deterministic offline rendering.
- All closed research scripts/results remain. Retired fit/backfill helpers moved
  under research; fifteen workflow files are recoverable from git history.
- Live rules and open questions lead CLAUDE.md. Dated full research and guidelines
  are preserved under docs/; the original registration timestamp remains intact.
- No changes under data/ or public/. No production activation or merge.
