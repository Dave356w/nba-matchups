# Pre-registered holdout: late-game Kalshi pricing

Fixed 2026-10-22, owner's request ("keep digging"), and committed **before
any 2025 or 2026 play-in or playoff Kalshi candle or ESPN play-by-play was
fetched** for this work. Nothing below may be changed after the holdout is
read. Source of the hypotheses: `research/strength_vs_reaction.py` §3 and
§10, run on 1,161 2025-26 regular-season games (`RESULTS.txt` at this
commit). Many looks were taken at those games (§1b–§10); these three are the
only ones that held in both halves of the season.

## Holdout

Every 2025 and 2026 NBA play-in and playoff game (ESPN season types 5 and 3)
with the home team's Kalshi KXNBAGAME 1-minute candles and ESPN
play-by-play with wallclock times. The data, cleaning, state matching and
quote rules are those of the regular-season run at this commit. No model
probability is used. None of these games entered any fit or scan.

## Hypotheses

**H-L1 (favourite-longshot bias; primary).** In each game, take the first
live regulation minute with τ ≤ 0.25 (4th quarter) whose raw favourite mid
is in (0.93, 0.995]. Count the favourite's actual losses k against the
expected λ = Σ(1 − mid). Prediction: k < λ, with a one-sided exact Poisson
p = P(X ≤ k | λ) < 0.05. Regular season: 27 lost against 41.0 expected
(ratio 0.66, 95% 0.43–0.96; Oct–Jan 0.70, Feb–Apr 0.58).

**H-L2 (the score beyond the mid).** Use 4th-quarter minutes (τ ≤ 0.25,
mids within [0.03, 0.97]) and fit y ~ logit(mid) [offset] + a·Δ/√τ +
b·z0√τ, each game weighted equally, with game-clustered SE. Prediction:
a > 0 at one-sided 95% (z > 1.645). Regular season: +0.021 ± 0.011
(Oct–Jan) and +0.022 ± 0.016 (Feb–Apr), ± 95%.

**H-L3 (the trade).** Buy the favourite at the first 4th-quarter minute its
raw ask is in (0.93, 0.99]. Fill at its worst quote 2 minutes later (YES at
the high ask, the away side at 1 − the low bid), pay the taker fee
(`market.kalshi_cost`), and hold to settlement. Prediction: pnl − null > 0
at one-sided 95%, where the null is the fill minute's mid less price and
fee. pnl itself is reported beside it. Regular season: pnl −0.36c ± 1.25,
pnl − null +1.48c ± 1.25 per $1 contract.

## Power, stated in advance

The holdout is about 170 games, with perhaps 130 reaching a 4th-quarter
favourite at 93c+. For H-L1 that gives λ ≈ 6, and at the regular-season
ratio about 4 losses, which is P ≈ 0.29: only a ratio near 0.2 or below
would reach p < 0.05. H-L2's and H-L3's SEs will be about twice the
regular season's. **So the holdout can confirm only a large effect. A
miss is not a refutation.** Playoff basketball also differs (rotations,
pace, stakes), so a difference may be regime, not noise. The decisive
test is the 2026-27 regular season, read from Kalshi's archive after the
season with these rules unchanged.

## Report

Each hypothesis's statistic, interval and pass/fail as defined above, win
or lose, with n; written to `diagnostics/strength_vs_reaction/HOLDOUT.txt`.
