# Realized WP band-adjusted shadow forward record — v2

Registered 2026-11-12T22:35:12Z, committed on its own before any v2 code or capture.
Algorithm version: `realized_wp_rolling50_bandadj_v2`. It supersedes
`realized_wp_rolling50_v1` (docs/realized_wp_forward_registration.md) for new
captures and for the card display. Everything not restated here is unchanged
from v1: capture, the official observation, frozen fields, scoring, the +EV
flat 1u paper rule, benchmarks, scope and persistence.

## Why

v1 compares a cell's raw win rate with one game's break-even price. A market
band is 10 to 20 points wide, so the cell's games carry a different average
market probability from the game being priced: a game at the bottom of its band
looks like value and one at the top looks overpriced even when the market is
perfectly calibrated. Found by code and data inspection on 2026-11-12, before
any v1 forward result was graded (0 of 4 official v1 observations graded).

## Estimate

For the model's selected team, with the same cell, window, order, seeding,
pooling and exclusions as v1 (latest 50 results in the joint model-WP ×
market-WP cell):

    Realized WP = clip( q_game + wins/games − mean(q_member), 0, 1 )

q_game is this game's current no-vig DraftKings probability for the selected
team; q_member is each retained cell observation's no-vig DraftKings
probability for its own selected team at its source price (five-hour lock for
live, close for previous-season replay). The other side is the complement. An
empty cell has no estimate. No shrinkage, pseudo-games, thresholds or tuning;
small cells stay noisy and the clip can still produce 0% or 100%.

## Record

- v1 closes when v2 is deployed (production keeps saving v1 until then). Its
  saved captures stay unchanged in the same files, are graded and reported
  under their own version, and are never pooled with v2. No v1 capture is
  rescored or converted.
- v2 captures start with the first production build after deployment. No
  backfill, including for games that already hold a v1 official observation;
  such a game may also receive its own v2 observation if a valid capture is
  saved before tip.
- The snapshot schema is unchanged. `realized_p_home` and
  `displayed_realized_wp` hold the v2 estimate; `cell_games`, `cell_wins`
  and the content-addressed evidence (which carries each member's market
  probability) reproduce it.
- The retrospective replay reports v1 and v2 side by side as exploratory
  research only; it selects nothing and enters no forward file.

Shadow research only. No model, lock, price, grade, public record or pick
changes. The public card and guide show the v2 estimate under the same name.
