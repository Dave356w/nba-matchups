# Realized WP rolling-50 shadow forward record — v1

Registered 2026-11-12T21:50:58Z, after the owner's explicit instruction to
save Realized WP before tip and keep this record as shadow research.
Algorithm version: `realized_wp_rolling50_v1`.

## Capture and official observation

Save each distinct valid pre-tip forecast prepared for the public card during
the production build, including forecasts outside the five-hour window.
Captures are append-only and never reconstructed from completed games.
Capture time is the actual recording time; the native forecast's timestamp is
saved separately and cannot substitute for recording time. Rendering a page
alone does not create a historical forward capture.

One official observation per game and algorithm version: the first valid saved
capture whose recording time is within five hours of its saved tip and strictly
before tip. The source forecast must also be timestamped within that five-hour
window, so a stale forecast cannot acquire a fresh official timestamp. Earlier captures remain audit history; later forecasts never replace
the official observation. A game with no valid capture in that window is not
scored. Activation does not reconstruct locks that predate this logger.

Freeze native model P, no-vig DraftKings Q, Realized WP, both quoted moneylines,
model-selected team, forecast/capture/tip timestamps, season, experience stage,
model tag, algorithm version, source revision, cell W/L/count/season composition,
the displayed rounded percentage alongside the full-precision probabilities,
and a content-addressed copy of the matrix evidence. The displayed probability
is the existing raw wins/games in the latest 50 joint-cell observations; early
and 10+ calibration evidence is pooled, with openers separate. None of that
prediction math changes in this registration.

## Scoring and fixed paper-return rule

Join saved forecasts to the native game's actual outcome only. Never replace
saved probabilities or prices with refreshed or closing values. Report model,
market and Realized WP probability scores on identical official graded games,
including games on which the return rule abstains. Use Brier score, log loss and
winner accuracy. Exact 0%/100% forecasts are preserved; a contradicted certainty
has infinite log loss, not a silently clipped score.

For each estimator separately, compute each side's expected net return at the
saved DraftKings price. Take a flat one-unit paper position only if its expected
net return is strictly positive. Choose the side with the greater expected net
return; at an exact tie choose home. At most one side per game. Underdogs and
sides opposite the native model pick are allowed. Net units and ROI use actual
outcomes at those saved quoted prices. No thresholds, sizing or window tuning.
Also report the same-game market favorite's flat-one-unit return as a benchmark.

Keep algorithm versions distinct. Show pooled and separate experience-stage
research breakdowns; these never alter the existing public ML records. All
pre-tip captures stay available for later sensitivity research, but are not
counted as independent games in the official record. No retrospective replay
enters this forward record. An algorithm or rule change needs a new version.

## Scope and persistence

Shadow research only: append-only snapshot and evidence files under `data/`,
with internal Markdown/JSON reports under `diagnostics/realized_wp_forward/`.
Nothing is added to the published Pages allowlist. Capture is enabled explicitly
by the production workflow. Native prices, locks, predictions and grading are
unchanged. Production build captures are persisted even if a later render fails;
they are saved forward forecasts, not proof that a visitor viewed a page or that
a quoted price was executable. No captured forecasts are hand-backfilled.
