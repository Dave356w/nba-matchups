# Model v4

Tags: `fourfactors_hl25_b2b_carry25_phase_avail_v4` (games 10+ with a
covering injury report) and `fourfactors_hl25_b2b_carry25_phase_v4` (all
other scored games). Full technical report of the v1 core:
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
| 1 ≤ g ≤ 9 | carryover logit (§3) | `model/logit_early.json` | `..._phase_v4` |
| g ≥ 10, report covers the game | availability logit (§4) | `model/logit_avail.json` | `..._phase_avail_v4` |
| g ≥ 10, otherwise | base logit (§2) | `model/logit.json` | `..._phase_v4` |

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
- **P(home)** = σ(a + b·Δ + c·b2b_net + e·Δ·phase).
  Committed fit (2016–19, 2021–26, n = 10,582): a = 0.291, b = 0.0228,
  c = 0.306, e = 0.0274. The effective slope on Δ roughly doubles from
  opening night (0.023) to April (0.050).

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
- **P(home)** = σ(a + b·Δ + c·b2b_net + d1·av_min + d2·av_bpm + e·Δ·phase).
  Committed fit (report seasons 2022-23 to 2025-26, last report ≥ 30 min
  before tip, covered games 10+, n = 4,289): a = 0.237, b = 0.0255,
  c = 0.266, d1 = 0.143, d2 = 0.0548, e = 0.0233.

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

`research/walk_forward.py --avail` (workflow "Walk-forward backtest",
2026-09-30): every fit on earlier seasons only, the exact routing of §1.

| Games | n | Model | Close | Model − close |
|---|---:|---:|---:|---:|
| 2024-25 ESPN BET, games 10+ | 1,071 | 0.5940 | 0.5776 | +0.0163 ± 0.0106 |
| 2025-26 DraftKings, games 10+ | 964 | 0.5918 | 0.5702 | +0.0216 ± 0.0114 |
| 2025-26 ESPN BET, games 10+ | 107 | 0.5400 | 0.5551 | −0.0151 ± 0.0399 |
| 2024-25 ESPN BET, games 1–9 | 138 | 0.6101 | 0.6169 | −0.0067 ± 0.0381 |
| 2025-26 ESPN BET, games 1–9 | 138 | 0.5785 | 0.5693 | +0.0092 ± 0.0420 |

- The model trails the close wherever the sample resolves it.
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

- **No demonstrated edge.** v4 trails the close by about 0.016–0.022 log
  loss per game on the resolved samples. Native (pregame-locked) rows are
  the forward test; there are none yet.
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
| Weights, base and early logits | Fit model | `nba_composite.py fit-weights`, `fit-logit --phase`, `cold_start.py fit` |
| Availability logit | Fit availability | `python player_availability.py fit --years 2023-2026` |
| Reconstructed rows | Backfill history | `python backfill_history.py --seasons 2025 2026 --rescore` |
| §6.1 | Walk-forward backtest | `python research/walk_forward.py --seasons 2025 2026 --avail` |
| §6.2 phase | Calibration shape | `python research/calibration_shape.py --seasons 2025 2026` |

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

## Version rule

Any change to prediction math (features, weights protocol, half-life, logit
terms, abstention) needs a new `MODEL_TAG` in `build_site.py`. Rows
keep the tag they were written under. A refit with the same protocol keeps
the tag; the fit's seasons and n are in `model/*.json`.
