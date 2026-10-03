# Model v6

Active since 2026-10-01 (owner's decision after the gate, "v6" below;
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
| `cold_start.py` | Games 1–9: last-season carryover and its logit |
| `player_availability.py` | Injury-report parsing, report coverage, availability terms and their logit |
| `build_site.py` | `score_game`: routes each game to one formula and records it; changes no math |
| `backfill_history.py` | `reconstruct_season`: the same routing on completed seasons (leave-one-season-out, or `walk_forward=True`) |

Every path (fit, daily build, CLI `score`, backfill, walk-forward) builds
its features with `nba_composite.logit_inputs`, so a feature means the same
thing everywhere.

## 1. Routing

For a game on date D, with g = min(games played before D by either team):

| Condition | Formula | Model file | Tag |
|---|---|---|---|
| g = 0 | abstain (no P) | — | — |
| 1 ≤ g ≤ 9 | carryover logit (§3) | `model/logit_early.json` | `..._luck_talent_ft_v6` |
| g ≥ 10, report covers the game | availability logit (§4) | `model/logit_avail.json` | `..._luck_talent_ft_avail_v6` |
| g ≥ 10, otherwise | base logit (§2) | `model/logit.json` | `..._luck_talent_ft_v6` |
| g ≥ 10, a v6 term missing (no box score, no 3PA) | frozen v4 base logit | `model/logit_v4.json` | `..._phase_v4` |

Preseason (exhibition) games are not regular-season rows: they are scored
separately to `data/nba_preseason.csv` (tag `..._carry25_preseason_g0`) by
the §3 logit on last season's games alone, with b2b_net from the previous
day's scoreboard, and shown only on the Preseason page. No new fit; the
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
P(home) = σ(a + b·Δ + c·b2b_net), fitted on every game in the first 20
games of 2023–26 (n = 1,233): a = 0.392, b = 0.0348, c = 0.306. No phase
term.

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

- **Lean**: the side with P ≥ 0.5 (a pick'em goes home). Model − market is
  the lean side's P minus its no-vig market probability.
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

## 7. Known limitations

- **No demonstrated edge.** v5 trails the close by about 0.016–0.019 log
  loss per game on the resolved samples. Native (pregame-locked) rows are
  the forward test; there are none yet.
- **Only the open is close to break-even.** At the close the model adds
  nothing to the price; at the open its disagreement is about the size a
  value bet needs, and lines move toward it (CLAUDE.md, open vs close).
  Part of that may be injury-report timing: `research/open_price.py`.
- **Heavy favourites are too flat.** Market favourites of 80–90% won
  83–86%; the base + phase model's mean P was 77–78% (§6.2 runs).
- **Reconstructed rows are hindsight**: priced at the close, with design
  choices (half-life, B2B, phase) made on overlapping seasons.
- **Not modelled**: travel, altitude, lineups beyond the report's
  Out/Doubtful, in-game or late news after the last report.
- **ATS** uses the fixed σ = 13.5 mapping (§5).

## 8. Reproducing

| Step | Workflow | Command |
|---|---|---|
| Weights, base, early and availability logits | Fit model | `nba_composite.py fit-weights`, `fit-logit --v6`, `cold_start.py fit`, `player_availability.py fit` |
| Availability logit alone | Fit availability | `python player_availability.py fit --years 2023-2026` |
| Reconstructed rows | Backfill history | `python backfill_history.py --seasons 2025 2026 --rescore` |
| §6.1 | Walk-forward backtest | `python research/walk_forward.py --seasons 2025 2026 --avail --v5` (v6 gate: `--v6`) |
| §6.2 phase | Calibration shape | `python research/calibration_shape.py --seasons 2025 2026` |
| Open vs close, H1–H3 | Open vs close | `python research/open_price.py --seasons 2025 2026` |

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

## Version rule

Any change to prediction math (features, weights protocol, half-life, logit
terms, abstention) needs a new `MODEL_TAG` in `build_site.py`. Rows
keep the tag they were written under. A refit with the same protocol keeps
the tag; the fit's seasons and n are in `model/*.json`.
