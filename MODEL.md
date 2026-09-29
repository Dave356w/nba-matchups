# Model: `fourfactors_hl25_b2b_carry25_v2`

Full technical report: [`docs/nba_composite_model_report.pdf`](docs/nba_composite_model_report.pdf).
The implementation is `nba_composite.py` (games 10+) and `cold_start.py`
(games 1–9). `build_site.py` supplies them with pregame game logs and records
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

## Version history

- `fourfactors_hl25_b2b_v1`: games 10+ only.
- `fourfactors_hl25_b2b_carry25_v2`: v1 unchanged from game 10; carryover
  model for games 1–9.

## Version rule

Any change to prediction math (features, weights protocol, half-life,
logit terms, abstention) needs a new `MODEL_TAG` in `build_site.py`. Rows
keep the tag they were written under. A preseason refit that keeps the same
protocol keeps the tag, and the refit date appears in `model/*.json` history.
