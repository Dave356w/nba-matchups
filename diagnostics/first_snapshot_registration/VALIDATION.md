# Item 2 validation

Registration commit: `2e301d3`, 2026-10-23T19:46:58Z, before CHI @ MIL tip.
Implementation is a later commit. No bot-generated data or public files changed.

- `python validate_data_files.py`: data files OK.
- `python -m pytest tests/ -q`: 371 passed (8 existing warnings).
- `python research/early_forward_segments.py`: 276 ESPN BET early v9 rows;
  separate early rule results in `early_hindsight.txt`, information only.
- Browser screenshots: Picks, Ledger, Results, Performance in `screenshots/`.
  External fonts/logos were blocked for deterministic offline rendering.
- Frozen retirement archive: 4 H1 and 11 H6 native bets, all ungraded;
  no H2/H3 native bets. Registered hindsight is retained separately from
  current-model hindsight. Never graded into an active record.
- Existing first model tags/P remain immutable, including CHI @ MIL's v8 snapshot.

Push and PR creation are blocked by automatic approval review; no production
activation has happened.
