# DFS simulation v2

The DFS tool uses Sleeper projections in Yahoo scoring. It is separate
from the tracked game model, its tags, snapshots, bets and records.

## Score distribution

For positive projection μ and modeled standard deviation σ, set
τ² = log(1 + σ²/μ²), then score = μ exp(τZ − τ²/2). This preserves both
the projected mean and the modeled standard deviation. A zero projection
is identically zero. It removes the old positive bias from clipping
μ + σZ at zero, particularly for cheap, low-minute players.

The provisional SD is (7.48 + 0.102μ) sqrt(min(minutes, 30)/30).
Unknown minutes use 30. This scaling is a provisional prior, not an
empirical claim about bench players. Accepted native-error fits multiply
variance by a shrunk ratio for projected minutes (<12, 12–23, ≥24) and
starter/bench role. Starter means the top five projected minutes on the
team; stars are the top two projections. Editing a projection refreshes
star and centre ranks; excluding a player from DFS does not change the
team's basketball roles.

Shared Gaussian factors define valid latent dependence, with independent
variance filling the remainder to one. A nonlinear score transform changes
the resulting Pearson correlations; factor coefficients are not advertised
as measured fantasy-score correlations. Overcommitted factor variance
raises an error instead of silently inflating variance.

Positive lognormal marginals omit negative fantasy scores and do not
explicitly model DNP, injury availability, minutes, overtime or game script.
These are limitations of the current distribution, not calibrated processes.

## Generation and evaluation

Generation draws produce legal winning candidates. Add the best 20 cash
lineups and cash lineups diversified under requested exposure caps, and
deduplicate. Freeze this set before drawing 5,000 fresh evaluation slates
with a separate deterministic seed (generation seed XOR 0x9E3779B9).
Rank candidates only on evaluation scores. A New draws action changes
both streams; Reset reproduces the default seed.

Candidate top 5% is the frequency of scoring in the top ceil(0.05 ×
candidate count) among this fixed set. Ties count. The reported ±1 SE is
sqrt(p(1−p)/5000), in percentage points, for Monte Carlo sampling only.
It excludes projection, distribution and selection uncertainty. p90 also
uses evaluation draws. Per-player example scores come from the lineup's
median evaluation slate, rather than its best or winning generation draw.

There is no opponent field, ownership, duplication or payout model. These
are relative candidate scores, not contest finish probabilities or ROI.

## Legal lineups and exposure

The salary/position dynamic program supplies upper bounds to a team-rule
branch-and-bound search. A max-team violation branches on dropping a
selected player from that team. A min-team violation branches on adding
an outside-team player. Prefix forcing/exclusion makes the branches
disjoint and exhaustive. Team-rule search has no cutoff. The outer k-best
enumerator reports `truncated` when its separate enumeration budget is
exhausted; callers raise an explicit search-limit message rather than
misreporting infeasibility.

Exposure counts are floor(returned lineup count × percentage). Zero
appearances are allowed; locked players alone are exempt. If a greedy
attempt returns fewer than requested, retry complete smaller portfolios
with recomputed counts. Recount actual selected IDs before publication.
If no complete portfolio is found, publish no partial set. One lineup
requires 100% exposure for each unlocked player. Greedy portfolio
selection is not a global portfolio optimum; the tournament candidate
set is also a sample, so failure to find a portfolio is not a proof that
no feasible portfolio exists in the full lineup universe.

## Native calibration and temporal check

Run `python research/dfs_calibration.py --as-of YYYY-MM-DD`. The existing
Fit model workflow also runs it. Outcomes are cached under
`bbr_cache/dfs_final_boxes/`; this cache is already retained by the workflow.

Only graded native games strictly before the cutoff enter. Independently
check each projection timestamp is before the native recorded tip. Prefer
the latest pregame `pre` row, falling back to `first`. Match names and
teams to final ESPN boxes. Explicit DNPs count as zero; missing or
unfinished box rows are never silently treated as zero. The untimed
historical Sleeper archive and rebuilt/reconstructed forecasts are excluded.

Fit shrunk variance ratios to actual minus projection errors. Estimate
latent dependence from standardized, Gaussianized projection errors,
averaging products within each game and shrinking toward zero by game
count. No negative centre-pair effect is imported from season-mean research.

The last two slate dates are held out; all earlier dates train. At least
500 matched training rows on five dates and 200 held-out rows on two dates
are required. Activation requires at least 95% result matching, no worse
held-out variance loss (mean(error²/variance + log variance)), and p90
coverage within max(3 percentage points, 3 × date-level SE) of 90%.
The holdout does not refit parameters. The loader rejects failed, missing,
malformed, overlapping or future-dated fits.

`model/dfs_simulation.json` is the versioned fit and decision;
`diagnostics/dfs_calibration/RESULTS.json` records its evidence. The page
states whether the model is provisional or native-holdout checked, with
the small held-out sample size. Passing two dates is an early diagnostic,
not evidence of season-long calibration, ownership accuracy or profitability.

The October 30 run trained on 881 player-games from October 22–27, then
checked 311 player-games on October 28–29. p90 coverage was 90.35%, versus
88.10% for the provisional model; variance loss fell from 6.0731 to 5.6420.
It matched 1,192 of 1,200 positive projections across 57 final games.
Dependence still has only a short native history; this report does not
validate joint lineup tails or contest returns.
