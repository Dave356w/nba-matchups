# Model v8

**v8** (owner's decision 2026-10-22, active since 2026-10-22: refit
`567753d`, reconstructed rows re-scored `3f10659`; "v8" below): one model for every game,
opening night included. Each team's rating carries last season in at 0.25 ×
(1 − games played ÷ 10), so it fades out by game 10; the base and
availability logits keep v6's terms and are fitted on every game
(availability from game 1); game 0's roster is the preseason box scores.
Tags: `fourfactors_hl25_b2b_fade10_phase_luck_talent_ft_avail_v8` (the
injury report covers the game, from game 1) and
`fourfactors_hl25_b2b_fade10_phase_luck_talent_ft_v8` (every other game);
a game whose terms are missing keeps the carryover logit before game 10
(`fourfactors_hl25_b2b_carry25_v2`) or the frozen v4 logit after
(`fourfactors_hl25_b2b_carry25_phase_v4`). H1 and H6 stay on the carryover
model (`carry_p_home`, games 1–9). §1–§4 below describe the v6/v7 pieces v8
reuses; "v8" at the end gives the changes and the test.

v6: active since 2026-10-01 (owner's decision after the gate, "v6" below;
refit `f299ac6`, from which the daily build scores v6; reconstructed rows
re-scored `f85e3ff`). Tags:
`fourfactors_hl25_b2b_carry25_phase_luck_talent_ft_avail_v6` (games 10+ with
a covering injury report) and
`fourfactors_hl25_b2b_carry25_phase_luck_talent_ft_v6` (all other scored
games); games 10+ whose v6 terms cannot be computed keep
`fourfactors_hl25_b2b_carry25_phase_v4` (the frozen v4 logit). v6 is v5 plus
one term, the own free-throw percentage gap (§2). Full technical report of the v1 core:
[`docs/nba_composite_model_report.pdf`](docs/nba_composite_model_report.pdf).

| Module | Role |
|---|---|
| `nba_composite.py` | Four-factor composite, decayed game features, `logit_inputs` (every logit feature), base logit fit |
| `cold_start.py` | The carryover rating and logit (H1 / H6's P, v8's fallback before game 10, preseason) |
| `player_availability.py` | Injury-report parsing, report coverage, availability terms and their logit |
| `build_site.py` | `score_game`: routes each game to one formula and records it; changes no math |
| `backfill_history.py` | `reconstruct_season`: the same routing on completed seasons (leave-one-season-out, or `walk_forward=True`) |

Every path (fit, daily build, CLI `score`, backfill, walk-forward) builds
its features with `nba_composite.logit_inputs`, so a feature means the same
thing everywhere.

## 1. Routing

v8 (active): for a game on date D, with g = min(games played before D by
either team):

| Condition | Formula | Model file | Tag |
|---|---|---|---|
| g ≥ 1, report covers the game | availability logit on the faded rating (v8) | `model/logit_avail.json` | `..._fade10_..._avail_v8` |
| every other game, g = 0 included | base logit on the faded rating (v8) | `model/logit.json` | `..._fade10_..._ft_v8` |
| a term missing, g ≤ 9 | carryover logit (§3) | `model/logit_early_v6.json` | `..._carry25_v2` |
| a term missing, g ≥ 10 | frozen v4 base logit | `model/logit_v4.json` | `..._phase_v4` |

Games 1–9 also record the carryover logit's P (`carry_p_home`) for H1 / H6.

Before v8 (v7, until 2026-10-22):

| Condition | Formula | Model file | Tag |
|---|---|---|---|
| g = 0 | abstain (no P) | — | — |
| 1 ≤ g ≤ 9 | early logit: carryover + roster talent (§3, v7) | `model/logit_early.json` | `..._carry25_talent_early_v7` |
| 1 ≤ g ≤ 9, talent missing (no box score) | carryover logit (§3) | `model/logit_early_v6.json` | `..._luck_talent_ft_v6` |
| g ≥ 10, report covers the game | availability logit (§4) | `model/logit_avail.json` | `..._luck_talent_ft_avail_v6` |
| g ≥ 10, otherwise | base logit (§2) | `model/logit.json` | `..._luck_talent_ft_v6` |
| g ≥ 10, a v6 term missing (no box score, no 3PA) | frozen v4 base logit | `model/logit_v4.json` | `..._phase_v4` |

Preseason (exhibition) games are not regular-season rows: they are scored
separately to `data/nba_preseason.csv` (tag `..._carry25_preseason_g0`) by
the §3 logit on last season's games alone, with b2b_net from the previous
day's scoreboard, priced from Kalshi (YES asks with the taker fee; close
= the last 1-minute candle at or before tip; `kalshi.py`), and shown only
on the Preseason page. No new fit; the
logit was trained on regular-season games, so those rows measure how far
the offseason view travels, not regular-season skill.

The CLI `score` has no box scores, so it always uses `model/logit_v4.json`.

**Report coverage** (`player_availability.covers`): the latest NBA injury
report lists the game's matchup (or both teams) for date D, and neither
team is marked NOT YET SUBMITTED. No report, an empty or unparsed one,
another slate or an unfiled team is *not* coverage; such a game uses the
base logit and is never read as "nobody is out". Fitting and backfill apply
the same rule. In 2022-23 to 2025-26, 4,922 of 4,923 games were covered.

## 2. Composite and base logit (games 10+)

- **Team features** (higher = better): off eFG%, −off TOV/100, off ORB/100,
  off FTA/FGA, −opp eFG%, opp TOV forced/100, −opp ORB/100, −opp FTA/FGA.
- **Decay**: each prior game's 16 raw totals (8 team, 8 opponent) are
  weighted by 0.5^(games ago / 25) and summed *before* rates are computed.
  Only games strictly before D count.
- **Composite** Δ = 100 · ((f_home − f_away) / sd) · w, in win-% points.
  Ridge weights (λ = 0.1) on team-season win%, features centred within
  season; `model/weights.json`, seasons 2015–19 and 2021–26.
- **b2b_net** = [away played yesterday] − [home played yesterday].
- **phase** = days since opening night / 175, capped at 1 (known before tip).
- **luck_def** = home − away `nba_composite.opp_luck`: how much a team's
  composite rises if the 3P% its opponents shot over its rating window
  (same decay; 3PA from the BBR logs) were the league's 3P% before D.
- **talent_diff** = home − away `player_availability.talent_fn`: minutes
  share (mean minutes / 48 this season before D, else last season's) ×
  last-season BPM value above replacement, summed over the players on the
  team's previous box score. Trades and returns count at once.
- **ft_diff** (v6) = home − away own FT% (100 × FTM / FTA), each team's
  free throws over its games before D, decayed like the rating window
  (`nba_composite.own_ft_pct`). The four factors read free throws only as
  FTA/FGA, so free-throw skill was missing.
- **P(home)** = σ(a + b·Δ + c·b2b_net + e·Δ·phase + l·luck_def +
  t·talent_diff + f·ft_diff). Committed v6 fit (refit `f299ac6`; 2016–19,
  2021–26, games with every term, n = 10,192): a = 0.297, b = 0.0161,
  c = 0.315, e = 0.0212, l = 0.0126, t = 0.0559, f = 0.0151 (a 10-point FT%
  gap ≈ 0.15 logit). The other terms barely move from v5 (t 0.0578 →
  0.0559). Talent takes about a third of Δ's weight (v4: b = 0.0228). The effective
  slope on Δ still roughly doubles from opening night to April.
- **Fallback** `model/logit_v4.json` (frozen, not refitted): σ(a + b·Δ +
  c·b2b_net + e·Δ·phase), n = 10,582: a = 0.291, b = 0.0228, c = 0.306,
  e = 0.0274.

## 3. Early season (games 1–9)

Each team's totals are [last season's full log × ρ][this season so far],
with the same decay running across the offseason (ρ = 0.25).

- **Carryover logit** (v2–v6; frozen in `model/logit_early_v6.json`):
  P(home) = σ(a + b·Δ + c·b2b_net), fitted on every game in the first 20
  games of 2023–26 (n = 1,233): a = 0.392, b = 0.0348, c = 0.306. No phase
  term.
- **v7**: σ(a + b·Δ + c·b2b_net + t·talent_diff), talent_diff as in §2
  (from game 1 every team has a box score), fitted by "Fit model" on the
  games with talent in the first 20 games of the box seasons 2016–19,
  2021–26 (`cold_start.py fit --talent`; `model/logit_early.json`). A game
  whose talent cannot be computed uses the carryover logit (v6 tag).
- Every early row records the carryover logit's P as `carry_p_home` (and
  `first_carry_p_home` with the first snapshot): H1 and H6 were registered
  on it and read it (`analysis.carry_view`). Preseason rows use the
  carryover logit (no box scores).

## 4. Player availability (games 10+, covered)

For each player with history for the team: a = his share of the decayed
rating window (half-life 25), role = minutes/48 when playing, value = last
season's BBR BPM shrunk by MP/(MP+500), minus replacement (−2; unmatched
players get replacement). p = 0 if the latest report lists him Out or
Doubtful, else 1 if he was on the team's previous box score, else 0.

- av_min = Σ role·(p − a), av_bpm = Σ role·value·(p − a), home − away.
- **P(home)** = σ(a + b·Δ + c·b2b_net + d1·av_min + d2·av_bpm + e·Δ·phase
  + l·luck_def + t·talent_diff + f·ft_diff), luck_def, talent_diff and
  ft_diff as in §2. Committed v6 fit (refit `f299ac6`; report seasons
  2022-23 to 2025-26, last report ≥ 30 min before tip, covered games 10+,
  n = 4,289): a = 0.247, b = 0.0199, c = 0.278, d1 = 0.135, d2 = 0.0382,
  e = 0.0212, l = 0.0159, t = 0.0398, f = 0.0175.
  Every term is refit on the four report seasons, so talent is estimated on
  less than half of §2's games (t = 0.041 vs 0.058); see "v5" below.

## 5. Lean and displays

- **Lean**: the side with P ≥ 0.5 (a pick'em goes home). The research pages'
  model − market is the lean side's P minus its no-vig market probability.
- **Card edge and EV (display only, 2026-10-22)**: the Picks card shows the
  model's P as recorded, but its edge and EV count only `analysis.SHRINK_W`
  = 0.40 of the model's disagreement with the price: P* = σ(logit q + 0.40 ·
  (logit P − logit q)), edge P* − q, EV P* × decimal odds − 1. 0.40 is the
  share outcomes bore out against the opening line (+0.43 / +0.36 fitted on
  the other season, `research/open_shrinkage.py`); at the close it is ~0, so
  at the card's later price it is generous. P, the lean, the ledgers and
  the hypotheses are unchanged; no new tag.
- **ATS (display only)**: the ledger maps p_home to an implied margin with a
  fixed σ = 13.5 points (`analysis.ATS_SIGMA`). This is an unvalidated
  assumption used to pick an ATS side; it is not a calibrated spread model
  and changes no prediction.

## 6. Evidence

All comparisons are on the same games as the close, one book at a time,
log loss with 95% intervals; positive = the market is better. An interval
crossing zero has not resolved the effect.

### 6.1 Production routing, walk-forward (headline)

`research/walk_forward.py --avail --v5` (workflow "Walk-forward backtest",
2026-09-30, the v5 gate): every fit on earlier seasons only, the exact
routing of §1, v5 beside v4 on the same games.

| Games | n | v5 | v4 | Close | v5 − close | v5 − v4 |
|---|---:|---:|---:|---:|---:|---:|
| 2024-25 ESPN BET, games 10+ | 1,071 | 0.5939 | 0.5940 | 0.5776 | +0.0163 ± 0.0099 | −0.0001 ± 0.0086 |
| 2025-26 DraftKings, games 10+ | 964 | 0.5896 | 0.5918 | 0.5702 | +0.0194 ± 0.0112 | −0.0022 ± 0.0068 |
| 2025-26 ESPN BET, games 10+ | 107 | 0.5481 | 0.5400 | 0.5551 | −0.0070 ± 0.0335 | +0.0081 ± 0.0247 |
| 2024-25 ESPN BET, games 1–9 | 138 | 0.6101 | same | 0.6169 | −0.0067 ± 0.0381 | 0 |
| 2025-26 ESPN BET, games 1–9 | 138 | 0.5785 | same | 0.5693 | +0.0092 ± 0.0420 | 0 |

- The model trails the close wherever the sample resolves it.
- v5 vs v4 is unresolved in every cell; v5 is not worse on the two large
  books, which was the shipping criterion. All games 10+ were covered, so
  every one was scored by the availability logit (§4).
- v4 vs the v1-style base (Δ, b2b only), same games: −0.0090 ± 0.0079
  (2024-25) and −0.0137 ± 0.0089 (2025-26).
- Walk-forward vs leave-one-season-out: +0.0010 ± 0.0030 (2024-25);
  identical in 2025-26 (every other season is earlier). The reconstructed
  rows are therefore not materially flattered by seeing later seasons.
- A possession-based pace arm adds nothing (±0.0003), and its interaction
  changes sign between seasons.

### 6.2 Components

Season phase (`research/calibration_shape.py`, walk-forward, logits
2016–24/25, rerun 2026-09-30 with production's opening-night clock; the
result is unchanged to four decimals):

| Test | 2024-25 ESPN BET | 2025-26 DraftKings |
|---|---:|---:|
| base + Δ·phase vs base | −0.0036 ± 0.0036 | −0.0075 ± 0.0032 |
| base + Δ·abs(Δ) vs base | +0.0005 ± 0.0014 | −0.0007 ± 0.0009 |
| od + Δ·phase vs od (`research/pregame_availability.py`) | −0.0008 ± 0.0006 | −0.0046 ± 0.0020 |

March–April slope of the outcome on the model logit (1 = calibrated
shape): base 1.42 → phase 1.16 (2024-25), 1.88 → 1.52 (2025-26); the close
is 1.23 and 1.37. The phase coefficient is stable (+0.023 to +0.025) with
two or more training seasons.

Availability (`research/pregame_availability.py`, walk-forward, games 10+):

| Rows | v2 vs close | v3 vs close | v3 vs v2 |
|---|---:|---:|---:|
| 2024-25 ESPN BET (1,071) | +0.025 ± 0.014 | +0.016 ± 0.011 | −0.008 ± 0.010 |
| 2025-26 DraftKings (964) | +0.036 ± 0.014 | +0.028 ± 0.012 | −0.008 ± 0.010 |

Each season's interval crosses zero, but both agree in size and sign; the
terms explain about a third of the market-minus-model logit gap (hindsight
"who played": 43–46%).

Early season (`research/cold_start_probe.py`, 2023-24 … 2025-26, ESPN BET
close): carryover trails by about +0.02 per game, like mid-season; the
"this season only" arm trailed by +0.03 to +0.09. Game 0 trails by +0.07
on 45 games, so it abstains.

### 6.3 Original report (v1, leave-one-season-out, 2023–26, 4,289 games)

| Model | Accuracy | Log loss |
|---|---:|---:|
| Decayed composite + B2B | 65.9% | 0.6156 |
| Decayed composite | 66.1% | 0.6182 |
| Season-to-date composite | 65.7% | 0.6193 |
| Home court only | 55.1% | — |

The composite correlates 0.98 with season-to-date margin: an interpretable
decomposition, not a stronger rating.

### 6.4 Outcomes only, every season (history backtest)

`research/history_backtest.py` (workflow "History backtest", run
39693643467, 2026-10-21, owner's question; `diagnostics/history_backtest/
RESULTS.txt`). With no prices it cannot say whether the model beats the
market. It tests whether the logic and calibration hold on every season the
game logs reach. Games 10+. Source: Basketball-Reference and ESPN, the
model's own feeds; stats.nba.com does not answer from the runners.

Walk-forward, the production v6 routing, 2016-17 to 2025-26 (9,509 games;
every fit on earlier seasons; the availability logit from 2023-24):

| Arm | Log loss | v6 − arm |
|---|---:|---:|
| Home edge only | 0.6858 | −0.0753 ± 0.0076 |
| v1 core (Δ, b2b) | 0.6226 | — |
| v4 core (+ Δ·phase) | 0.6205 | −0.0100 ± 0.0033 (better in 9 of 9 seasons; Brier −0.0045 ± 0.0015) |
| v6 | 0.6105 | |

Each v6 term dropped and refitted, on the games it reaches (positive = the
term helps): roster talent +0.0059 ± 0.0026, back-to-back +0.0024 ± 0.0012,
Δ·phase +0.0009 ± 0.0006 (each 8 of 9 seasons); availability +0.0029 ±
0.0033 (3 of 3); own FT% +0.0004 ± 0.0005 (8 of 9); opponent 3P% luck
+0.0002 ± 0.0009 (6 of 9, the least supported term pooled). That figure is
mostly the base route of 2016-17 to 2022-23: on the priced seasons, where
every game 10+ runs through the availability logit, dropping luck fails the
gate (v7, walk-forward run 39698425564: v7 − v6 +0.0018 ± 0.0017 2024-25
ESPN BET, +0.0013 ± 0.0028 2025-26 DK; owner kept v6, 2026-10-22).

Calibration against outcomes: slope 0.99 ± 0.06 from the second fold on
(0.95 ± 0.05 with 2016-17, whose fit had one season: 0.70); 0.90 to 1.17 by
season. By season third the v1 core runs 0.81 / 0.90 / 1.24 and v6 0.92 /
0.88 / 1.04, so Δ·phase removes the late-season flatness out of sample.
Reliability is within Wilson 95% in 8 of 10 favourite bins (65–70%: 67.5
predicted, 64.8 won; 85–90%: 87.3, 84.4).

The home edge follows the training seasons. Home teams won 55.9% against
57.4% predicted from 2017-18 on (intercept −0.07 ± 0.05); 2020-21, 2021-22
and 2023-24 were 3–4 pts below the prediction, 2024-25 and 2025-26 within
1.3. The recency-weighted intercept was tested and bought nothing
([`docs/model_guidelines.md`](docs/model_guidelines.md) §4).

Out of era, 2004-05 to 2013-14 (10,480 games no fit has seen), the frozen
core as committed (weights and `logit_v4.json`): slope 1.03 ± 0.06
(opening third 1.17 ± 0.13, then 1.01 and 0.99); home won 59.9% against
57.1% (intercept +0.12 ± 0.04, a larger home edge then). Frozen − logit
refit on the era's earlier seasons +0.0016 ± 0.0015 (mostly the larger home
edge); frozen − full refit (weights too) +0.0005 ± 0.0018. Weights fitted
on 2004–14 correlate +0.96 with the committed ones (more weight on
opponents' eFG% and both FTA/FGA terms then). The four-factor composite and the logit's shape
carry across eras; only the home edge moves.

Follow-ups (2026-10-22, the same run; `RESULTS.txt` A8, A9, C):

- In-season home edge (A8): each game's logit shifted by this season's own
  residual on earlier dates, empirical-Bayes prior. The true season-to-season
  spread of the offset is ~0.04 logit (tau), mostly noise: −0.0001 ± 0.0002
  pooled (helps in 2 of 9), +0.0000 / +0.0001 against the close; out of era
  −0.0005 ± 0.0002. Not worth a change.
- Rookies by draft slot (A9): drafted rookies at the rookie BPM their pick
  predicts (about −1.26 − 0.45 log pick, earlier seasons only) instead of at
  replacement: −0.0002 ± 0.0002, better in 9 of 9 seasons; against the
  close −0.0001 ± 0.0006 / −0.0002 ± 0.0006. Real but small.
- Games 1–9 (C): the carryover, walk-forward 2016–26 (n = 1,258): home lean
  +0.2 ± 2.6 pts, slope 0.88 ± 0.18, −0.040 ± 0.017 vs the home edge; the
  frozen early logit on 2005–14 (n = 1,257): −0.1 ± 2.6 pts, slope 0.90 ±
  0.17. 2020-21 (no crowds) is the outlier (+10.7 pts).

Back-to-backs: in 2016–26 home teams on a back-to-back won 48.3% (45.2–51.4)
against 51.7% predicted, about 2 pts below the season's general home
over-prediction; in 2004–14 they won 54.5% against 49.7%, about 2 pts above
it. The residual flips sign between eras, so a separate home back-to-back
term is not a lead.

## 7. Known limitations

The aggregate review of every result, with the rules for using and
changing the model, is [`docs/model_guidelines.md`](docs/model_guidelines.md)
(2026-10-21; numbers from `research/model_review.py`).

- **No demonstrated edge.** v6 trails the close by +0.0184 ± 0.0110
  (2025-26 DraftKings) and +0.0143 ± 0.0098 (2024-25 ESPN BET) log loss per
  game, games 10+, and most of the gap sits in the games where it disagrees
  most with the close. Native (pregame-locked) rows are the forward test;
  none is graded yet.
- **Only the open is close to break-even.** At the close the outcome puts
  no weight on the model's disagreement (w −0.06 ± 0.40 / +0.01 ± 0.37); at
  the open about a third (+0.33 ± 0.37 / +0.44 ± 0.35), and lines move
  toward it (CLAUDE.md, open vs close). Part of that may be injury-report
  timing: `research/open_price.py`.
- **Not flat against outcomes.** The calibration slope is 1.04 ± 0.17
  (2025-26 DK) and 0.97 ± 0.15 (2024-25 ESPN BET); the earlier "heavy
  favourites too flat" reading was against the close or an older model (v5
  audit, Task C). Heavy favourites (≥ 0.90 at the close) trail the DK close
  in 2025-26 (+0.050 ± 0.013, n = 39) but not the ESPN BET close in 2024-25.
  Against outcomes alone the slope is 0.99 ± 0.06 walk-forward (2017-18 to
  2025-26) and 1.03 ± 0.06 for the frozen core on 2004–14 (§6.4).
- **The home edge is the training seasons' edge.** It ran 1.5 pts above
  the actual home rate walk-forward, 2017-18 to 2025-26 (3–4 pts in 2020-21,
  2021-22 and 2023-24) and 2.8 pts below it on 2004–14 (§6.4); within 1.3
  pts in the last two seasons.
- **Reconstructed rows are hindsight**: priced at the close, with design
  choices (half-life, B2B, phase) made on overlapping seasons.
- **Not modelled**: travel, altitude, lineups beyond the report's
  Out/Doubtful, in-game or late news after the last report.
- **ATS** uses the fixed σ = 13.5 mapping (§5).

## 8. Reproducing

| Step | Workflow | Command |
|---|---|---|
| Weights, base and availability logits (v8) | Fit model | `nba_composite.py fit-weights`, `fit-logit --v8`, `player_availability.py fit --v8` |
| Availability logit alone | Fit availability | `python player_availability.py fit --v8 --years 2023-2026` |
| Box scores and preseason box scores | Box-score history | `python research/box_history.py --seasons 2016-2019 2021-2026 --commit` |
| v8 test | Unified model (v8) | `python research/unified.py` |
| Reconstructed rows | Backfill history | `python backfill_history.py --seasons 2025 2026 --rescore` |
| §6.1 | Walk-forward backtest | `python research/walk_forward.py --seasons 2025 2026 --avail --v5` (v6 gate: `--v6`) |
| §6.2 phase | Calibration shape | `python research/calibration_shape.py --seasons 2025 2026` |
| Open vs close, H1–H3 | Open vs close | `python research/open_price.py --seasons 2025 2026` |
| H1–H6 on DraftKings | — (repo data) | `python research/dk_hypotheses.py [--open-price research/output/open_price.csv]` |
| Model review, §7 and the guidelines | — (repo data) | `python research/model_review.py` |
| §6.4 outcomes only | History backtest | `python research/history_backtest.py --seasons 2017 2018 2019 2021 2022 2023 2024 2025 2026 --era 2005-2014` |

The 2026-09-30 refit and re-score after the coverage fix reproduced the
committed `logit_avail.json` and reconstructed rows exactly (nothing to
commit), because every historical test game was covered.

## Version history

- `fourfactors_hl25_b2b_v1`: games 10+ only.
- `fourfactors_hl25_b2b_carry25_v2`: v1 from game 10; carryover for games 1–9.
- `fourfactors_hl25_b2b_carry25_avail_v3`: v2 plus availability terms for
  games 10+ (rows without a report keep the v2 tag).
- `fourfactors_hl25_b2b_carry25_phase_v4` /
  `fourfactors_hl25_b2b_carry25_phase_avail_v4`: v3 plus Δ·phase in both
  games-10+ logits. Later fix (same tags, since each row's tag still names
  the formula that scored it): availability only on report-covered games.
- `fourfactors_hl25_b2b_carry25_phase_luck_talent_v5` /
  `..._phase_luck_talent_avail_v5`: v4 plus opponent 3-point luck and
  roster talent in both games-10+ logits (next section); v4 base logit
  kept as the fallback when a v5 term is missing.
- `fourfactors_hl25_b2b_carry25_phase_luck_talent_ft_v6` /
  `..._phase_luck_talent_ft_avail_v6`: v5 plus the own FT% gap in both
  games-10+ logits ("v6" below); same v4 fallback.
- `fourfactors_hl25_b2b_carry25_talent_early_v7`: games 1–9 add roster
  talent to the carryover logit ("v7" below). Games 10+ keep the v6 tags;
  early games without talent keep the carryover logit and the v6 tag.
- `fourfactors_hl25_b2b_fade10_phase_luck_talent_ft_v8` /
  `..._fade10_phase_luck_talent_ft_avail_v8`: one faded rating and one pair
  of logits for every game, game 0 included ("v8" below). Missing terms
  fall back to the carryover logit (v2 tag) before game 10 and the v4 logit
  after; v7's `model/logit_early.json` is retired.

## v5 (active since 2026-09-30)

`fourfactors_hl25_b2b_carry25_phase_luck_talent_v5` /
`..._phase_luck_talent_avail_v5` add luck_def and talent_diff (§2) to both
games-10+ logits (`nba_composite.V5_FEATURES`,
`player_availability.FEATURES_V5`). Fitted walk-forward, 71–78% of
opponents' 3P% deviation is noise the v4 rating counted as defence.

Evidence (research/team_quality.py, walk-forward, box seasons 2015-16 on,
games 10+, vs base v4 on the same games): −0.0036 ± 0.0098 (2024-25 ESPN
BET) and −0.0054 ± 0.0089 (2025-26 DK) log loss; luck alone ≈ −0.002 per
season (~2 SE pooled), talent alone −0.003 per season (unresolved). Routing:
a game whose v5 terms cannot be computed (no box scores or 3PA) is scored by
the frozen v4 base logit `model/logit_v4.json` and tagged v4; the CLI
`score` always uses it (no box scores). Games 1–9 are unchanged.

Activation (2026-09-30), in the pre-set order: (1) the gate, §6.1: v5 not
worse than v4 on the same games (−0.0001 ± 0.0086 2024-25 ESPN BET,
−0.0022 ± 0.0068 2025-26 DK); (2) "Fit model" refit (`d765264`,
coefficients in §2 and §4); (3) "Backfill history" `--rescore`
(`a3b5415`): all 2,418 reconstructed rows re-tagged v5, leave-one-season-
out v5 − v4 −0.0009 ± 0.0074 (2024-25 ESPN BET, games 10+) and
−0.0022 ± 0.0068 (2025-26 DK); games 1–9 unchanged.

The gain is smaller than the team-quality arms (about −0.003 to −0.005),
which were measured against base v4 without the availability terms. v5 is
measured against v4 *with* them, and both av_bpm (the BPM of players out
relative to the rating window) and talent_diff are built from last season's
BPM of the roster, so part of talent's gain was already in v4.

Tested and rejected (walk-forward, 2026-09-30, `prod_fixed` in
`research/walk_forward.py --avail --v5`): the availability logit with luck
and talent fixed at the base fit's coefficients (`player_availability.
fit_fixed`, an offset) and only its other terms refit. Against the shipped
v5 on the same games, games 10+: +0.0008 ± 0.0006 (2024-25 ESPN BET),
+0.0016 ± 0.0018 (2025-26 DK), −0.0013 ± 0.0065 (2025-26 ESPN BET, n =
107). Holding talent at 0.060 pulls d_phase from 0.021 to 0.013 and av_min
from 0.14 to 0.07. The availability logit's lower talent coefficient (0.041
vs 0.058) reflects the overlap with av_bpm, not a noisy estimate, so the
free fit stays.

## v6 (activated 2026-10-01)

Origin: the v5 audit (`diagnostics/v5_audit/RESULTS.md`, Task D). The
market's correction to v5, logit(close) − logit(v5), loads on each team's
own FT% in both seasons (t = 3.0 2024-25 ESPN BET, 9.7 2025-26 DK; week
block bootstrap), and the factor gaps plus FT% explain 46–63% of the
persistent team-level part of that correction. The term is fitted on
outcomes only; the market residual only pointed at it.

Gate (`research/walk_forward.py --avail --v6`, run 36940663654, the v5
gate's protocol: every fit on earlier seasons, production routing, v6 vs v5
on the same games, games 10+, log loss ± 95%): −0.0003 ± 0.0008 (2024-25
ESPN BET, n = 1,071), −0.0010 ± 0.0011 (2025-26 DK, n = 964), −0.0018 ±
0.0042 (2025-26 ESPN BET, n = 107); Brier agrees. Not worse on either large
book (the shipping criterion); better in all three, resolved in none. v6 −
close +0.0160 / +0.0184: no demonstrated edge. Games 1–9 unchanged.

Activation (2026-10-01/02, as v5): (1) the gate above; (2) "Fit model"
refit `f299ac6` (§2, §4: ft_diff 0.0151 base, 0.0175 availability; other
terms within 0.002 of v5); (3) "Backfill history" `--rescore` `f85e3ff`:
all 2,418 reconstructed rows re-tagged v6 (2,142 availability, 276 early).
Leave-one-season-out v6 − v5 on the same rows, games 10+, ± 95%:
−0.0002 ± 0.0016 (2024-25 ESPN BET, n = 1,071), −0.0010 ± 0.0011 (2025-26
DK, n = 964), −0.0018 ± 0.0042 (2025-26 ESPN BET, n = 107); v6 − close
+0.0143 / +0.0184; games 1–9 unchanged. The build follows the model files.

## v7 (owner's decision, 2026-10-22)

Origin: the model's weaknesses review (2026-10-22): the early route saw no
players, so offseason roster changes entered only through results, while
the market prices them from opening night.

Test (`research/early_roster.py`, workflow "Early roster", run 39830317895,
`diagnostics/early_roster/RESULTS.txt`): the early logit + talent_diff vs
the carryover features, walk-forward on games 1–9 of 2016-17 to 2025-26
(n = 1,219), ± 95%: −0.0105 ± 0.0087 log loss vs outcomes (Brier −0.0045 ±
0.0039), better in 8 of 9 seasons, talent coefficient 0.072–0.078 from
2018-19 on. Against the close (ESPN BET 2024-25 / 2025-26, n = 138 each;
DraftKings priced no game 1–9): carryover − close −0.0080 / +0.0125, v7 −
close −0.0124 / +0.0068 (v7 − carryover −0.0045 ± 0.0220 / −0.0057 ±
0.0248): not worse on either. The gain is above G7's 0.008 leakage
tripwire; checked (walk-forward fits, the box score strictly before the
date, and v7 narrows but does not beat the close). The market's correction
of the carryover model follows talent_diff (corr +0.42 ± 0.12, n = 276).

H1 and H6 (games 1–9) were registered on the carryover model. The owner
kept them there: the build and the backfill record the carryover P beside
the v7 P (`carry_p_home`, `first_carry_p_home`), and the hypotheses read it.
In hindsight H1's profit sat in bets roster talent was against (n 77,
+32.4% ± 33.2) more than those it backed (n 38, −11.8% ± 33.3), unresolved.

Activation: (1) the test above; (2) "Fit model" refits
`model/logit_early.json` with talent; (3) "Backfill history" `--rescore`
re-tags the reconstructed games 1–9 v7 and fills `carry_p_home` (H1 / H6
hindsight unchanged). The build follows the model files.

## v8 (owner's decision, 2026-10-22)

Origin: the owner's request for one model from day 1 to the end of the
season. Under v7 game 0 abstained (~16 games a season) and at game 10 a
team's rating dropped last season at once (sd of the game 9 → 10 change in
a team's composite 7.4 win-% points, against ~2.9 from one game to the next
later on).

Changes:
- **Rating** (`nba_composite.fade_v8_games`, `fade_inputs`): each team's 16
  raw totals are [last season × ρ(i)][this season], one decay (half-life
  25) running across the offseason, ρ(i) = 0.25 × max(0, 1 − i ÷ 10), i =
  the team's games played. Game 0 is last season's rating; from game 10
  the rating is this season's alone, as in v6. Opponent 3-point luck
  (league 3P% before the date, else last season's) and the own FT% gap read
  the same totals.
- **Logits**: v6's base terms (Δ, b2b_net, Δ·phase, luck_def, talent_diff,
  ft_diff) fitted on every game of the game-log seasons (`fit-logit
  --v8`); the availability logit (+ av_min, av_bpm) on every covered game
  from game 1 (`player_availability.py fit --v8`). At game 0 no player has a
  share of the rating window, so those games use the base logit.
- **Game-0 roster** (`player_availability.roster_talent`): talent before a
  team's first game sums its preseason players (ESPN box scores,
  `data/nba_box_pre_<season>.csv`): last season's minutes role × shrunk BPM
  above replacement, the pregame analogue of "the previous box score".
  Against the roster each team actually opened with: corr +0.94 (285
  team-seasons, 2015-16 to 2025-26), mean +0.10 pts/48.
- **Fallbacks**: a game whose terms are missing gets the carryover logit
  before game 10 and the frozen v4 logit after. H1 and H6 keep the
  carryover P (`carry_p_home`, games 1–9).

Test (`research/unified.py`, workflow "Unified model (v8)", run 39867001658,
`diagnostics/unified/RESULTS.txt`), walk-forward 2016-17 to 2025-26 against
v7's routing rebuilt on the same games (rebuilt inputs match the production
builders to 1e-14); the fade point (10, 20, 40 or none) was chosen by a rule
fixed before the run, the lowest pooled log loss on games 1+. ± is 95%:

| Games | n | v8 − v7 (log loss) | Brier | Seasons better |
|---|---:|---:|---:|---|
| 1+ | 10,767 | −0.0003 ± 0.0008 | −0.0001 ± 0.0003 | 6 of 9 |
| 1–9 | 1,258 | −0.0015 ± 0.0056 | −0.0007 ± 0.0025 | 6 of 9 |
| 10–19 | 1,375 | −0.0011 ± 0.0017 | −0.0003 ± 0.0007 | 6 of 9 |
| 20+ | 8,134 | +0.0001 ± 0.0004 | +0.0000 ± 0.0002 | 4 of 9 |

Against the close, same games: ESPN BET 2024-25 (n = 1,209) −0.0011 ±
0.0025 (Brier −0.0004), DraftKings 2025-26 (n = 964) −0.0009 ± 0.0010
(Brier −0.0003): not worse on either large book-season (G7); ESPN BET
2025-26 games 10+ (n = 107) +0.0057 ± 0.0051, its games 1–9 −0.0077 ±
0.0199. Game 0 (n = 143, v7 abstained): −0.0145 ± 0.0214 vs the carryover
logit without a roster, −0.067 ± 0.047 vs the home edge; no game-0 row is
priced. No season gains more than 0.008. The game 9 → 10 jump falls from
sd 7.4 to 4.1 points. Equal accuracy, one formula, every game projected:
a simplification and a coverage fix, not an edge.

Activation: (1) "Box-score history" commits each finished season's
preseason box scores; (2) "Fit model" fits the v8 logits (and retires
`model/logit_early.json`); (3) "Backfill history" `--rescore` re-scores the
reconstructed rows with v8 and adds the game-0 rows v7 never wrote (with
their prices); H1 / H6 hindsight unchanged. The build follows the model
files.

Done 2026-10-22: preseason boxes `9f9ac3f` (2015-16 to 2025-26 but 2019-20,
0 failed requests, 30 teams each); refit `567753d` (base n = 11,687 games,
availability n = 4,847); rescore `3f10659` after the pandas 3.1 fix
(Dave356w/nba-matchups#95): 2,450 rows, all v8 (1,209 a season on the
availability logit, the rest base), 32 game-0 rows added (all priced by
ESPN BET, so off the pages), `carry_p_home` identical on all 276 games 1–9
rows (H1 n 115 +17.8%, H6 n 201 +3.6%, unchanged). Game 0 is in no
pre-registered rule (`analysis.stage`). On the re-scored rows, games 10+,
model − close +0.0174 ± 0.0112 (2025-26 DK) and +0.0152 ± 0.0101 (2024-25
ESPN BET), log loss, ± 95% (`diagnostics/model_review/RESULTS.txt`).

## Version rule

Any change to prediction math (features, weights protocol, half-life, logit
terms, abstention) needs a new `MODEL_TAG` in `build_site.py`. Rows
keep the tag they were written under. A refit with the same protocol keeps
the tag; the fit's seasons and n are in `model/*.json`.
