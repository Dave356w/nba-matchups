# Model: `fourfactors_hl25_b2b_carry25_phase_avail_v4`

Full technical report: [`docs/nba_composite_model_report.pdf`](docs/nba_composite_model_report.pdf).
The implementation is `nba_composite.py` (games 10+), `cold_start.py`
(games 1–9) and `player_availability.py` (availability terms, games 10+). `build_site.py` supplies them with pregame game logs and records
their output; it does not change the math.

## Specification

- **Features** (higher = better): off eFG%, −off TOV/100, off ORB/100, off
  FTA/FGA, −opp eFG%, opp TOV forced/100, −opp ORB/100, −opp FTA/FGA.
- **Composite**: features centered within season and divided by pooled SD,
  with ridge weights (λ = 0.1) fitted to team-season win%, expressed in
  win-% points.
- **Game features**: each team's 16 raw totals from prior games, weighted by
  0.5^(games ago / 25), are summed *before* rates are computed.
- **P(home win)** = σ(a + b·Δ + c·b2b_net), where Δ = home − away composite
  and b2b_net = away on a back-to-back − home on a back-to-back. The report's
  pooled 2023–26 fit is a = 0.241, b = 0.0374, c = 0.327 (`model/logit.json`).
- **Early season (new in v2)**: when min(games played) is 1–9, each team's
  totals are [last season's full log × ρ][this season so far], with the same
  decay running across the offseason (ρ = 0.25). P(home) uses its own logit,
  fitted on every game in the first 20 games of the logit seasons
  (`model/logit_early.json`, `python cold_start.py fit`).
- **Player availability (new in v3, games 10+)**: P(home) = σ(a + b·Δ +
  c·b2b_net + d1·av_min + d2·av_bpm) (`model/logit_avail.json`,
  `python player_availability.py fit`). For each player with history for
  the team: a = share of the decayed rating window (half-life 25) he played
  in, role = his minutes/48 when playing, value = last season's BBR BPM
  shrunk by MP/(MP+500), minus replacement (−2). p = 0 if the latest NBA
  injury report lists him Out or Doubtful, else 1 if he was on the team's
  previous box score, else 0. av_bpm = Σ role·value·(p − a) and
  av_min = Σ role·(p − a), home − away. When the report or box history is
  missing, the row uses the base logit.
- **Season phase (new in v4, games 10+)**: both logits add e·Δ·phase, where
  phase = days since opening night / 175, capped at 1. A given composite gap
  counts for more as the season goes on (about double by April). The base
  logit (`model/logit.json`, `fit-logit --phase`) is fitted on the 2016–19
  and 2021–26 game logs; the availability logit on the 2022-23 to 2025-26
  report seasons. Rows with a report are tagged
  `fourfactors_hl25_b2b_carry25_phase_avail_v4`; rows without one, and games
  1–9 (carryover unchanged), `fourfactors_hl25_b2b_carry25_phase_v4`.
- **Abstention**: only while a team has played no games (v1 abstained until
  both teams had 10).
- **Lean**: the side with P ≥ 0.5 (a pick'em goes to the home side). Model −
  market is the lean side's P minus its no-vig market probability.

## What the report established (leave-one-season-out, 2023–26, 4,289 games)

| Model | Accuracy | Log loss |
|---|---:|---:|
| Decayed composite + B2B | 65.9% | 0.6156 |
| Decayed composite | 66.1% | 0.6182 |
| Season-to-date composite | 65.7% | 0.6193 |
| Home court only | 55.1% | — |

Calibration by quintile was close to exact. The composite correlates 0.98
with season-to-date margin, so it is an interpretable decomposition rather
than a stronger rating.

## Against the market (reconstructed, hindsight)

- Games 10+: the model trails the close by +0.035 ± 0.007 log loss per game
  (DraftKings, 2025-26, n = 964) and +0.026 ± 0.007 (ESPN BET, 2024-25,
  n = 1,071).
- Games 1–9, `research/cold_start_probe.py` (2023-24 … 2025-26, ESPN BET
  close): the carryover arm trails by about +0.02 per game, similar to
  mid-season, where the v1-style "this season only" arm trailed by +0.03
  to +0.09. Game 0 trails by +0.07 on 45 games, so it still abstains.

The reconstructed fits hold out the test season but keep later seasons (the
2024-25 rows saw 2025-26), so they flatter the model slightly;
`research/walk_forward.py` re-scores them with earlier seasons only.

No arm or bucket beats the close. The model does not see injuries, lineups,
or travel; the market does. Native rows are the forward test.

## v3 evidence (research/pregame_availability.py, hindsight-free inputs)

Walk-forward (every fit on earlier seasons), games 10+, the last injury
report at least 30 min before tip, same games as the close, one book at a
time (log loss; 95% intervals; negative = better):

| Rows | v2 vs close | v3 vs close | v3 vs v2 |
|---|---:|---:|---:|
| 2024-25 ESPN BET (1,071) | +0.025 ± 0.014 | +0.016 ± 0.011 | −0.008 ± 0.010 |
| 2025-26 DraftKings (964) | +0.036 ± 0.014 | +0.028 ± 0.012 | −0.008 ± 0.010 |

Each season's v3 − v2 interval crosses zero, but the two agree in size and
sign; the terms explain about a third of the market-minus-model logit gap
(hindsight "who played": 43–46%). v3 still trails the close in both books.
Close to tip, Questionable players have almost all been resolved to Out or
Available, so the Out/Doubtful rule loses little. Native rows are the test.

## v4 evidence (research/calibration_shape.py, research/pregame_availability.py)

In-sample, the outcome's slope on the model logit was ~1 overall but
1.4–1.9 in March–April (too flat) and ~0.8 before March. Walk-forward, same
games as the close, one book at a time (log loss; 95% intervals):

| Test | 2024-25 ESPN BET | 2025-26 DraftKings |
|---|---:|---:|
| base + Δ·phase vs base (logits 2016–24/25) | −0.0036 ± 0.0036 | −0.0075 ± 0.0033 |
| base + Δ·abs(Δ) vs base | +0.0005 ± 0.0014 | −0.0007 ± 0.0009 |
| od + Δ·phase vs od (report seasons from 2022-23) | −0.0008 ± 0.0006 | −0.0046 ± 0.0020 |

The phase coefficient is stable (+0.02 to +0.025) whenever two or more
training seasons are available; 2023-24 alone showed little of it, which is
why the 2024-25 gain on top of the report is small. Mar–Apr slope in
2025-26 DraftKings: od 1.79 → od + phase 1.61 (the close: 1.37). The 107
early-season 2025-26 ESPN BET games were slightly worse with phase
(+0.005 ± 0.013, unresolved); watch native games 10–25.

## Version history

- `fourfactors_hl25_b2b_v1`: games 10+ only.
- `fourfactors_hl25_b2b_carry25_v2`: v1 unchanged from game 10; carryover
  model for games 1–9.
- `fourfactors_hl25_b2b_carry25_avail_v3`: v2 plus the pregame player
  availability terms for games 10+ (rows without a report keep the v2 tag).
- `fourfactors_hl25_b2b_carry25_phase_v4` /
  `fourfactors_hl25_b2b_carry25_phase_avail_v4`: v3 plus Δ·season phase in
  both logits; base logit on 2016–19, 2021–26; availability logit on
  2023–26. The reconstructed rows are re-scored with v4 (leave-one-season-out
  for weights, both logits and the carryover logit).

## Version rule

Any change to prediction math (features, weights protocol, half-life,
logit terms, abstention) needs a new `MODEL_TAG` in `build_site.py`. Rows
keep the tag they were written under. A preseason refit that keeps the same
protocol keeps the tag, and the refit date appears in `model/*.json` history.
