# Model: `fourfactors_hl25_b2b_v1`

Full technical report: [`docs/nba_composite_model_report.pdf`](docs/nba_composite_model_report.pdf).
The implementation is `nba_composite.py`. `build_site.py` supplies it with
pregame game logs and records its output; it does not change the math.

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
  pooled 2023–26 fit is a = 0.241, b = 0.0374, c = 0.327.
- **Abstention**: no prediction until both teams have 10 games.
- **Lean**: the side with P ≥ 0.5 (a pick'em goes to the home side). Model −
  market is the lean side's P minus its no-vig DraftKings probability.

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

## Not yet established

The model has **no market benchmark** yet. Answering that is the job of this
repository's calibration page: Brier and log loss against the DraftKings close
on the same games, calibration against actual results, and CLV on native
rows. The model does not see injuries, lineups, or travel; the market does.

## Version rule

Any change to prediction math (features, weights protocol, half-life,
logit terms, abstention) needs a new `MODEL_TAG` in `build_site.py`. Rows
keep the tag they were written under. A preseason refit that keeps the same
protocol keeps the tag, and the refit date appears in `model/*.json` history.
