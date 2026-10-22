# Model guidelines (v6): strengths, weaknesses and rules of use

Owner's request, 2026-10-21: step back from the individual studies and say,
from all of them together, what the model is good for, what it is not, and
how to use and change it. Every figure comes from a committed result:
`MODEL.md` §6, the research log in `CLAUDE.md`, `diagnostics/` (v5 audit,
ATS signal, Kalshi benchmark, DraftKings backtest) and a new aggregate check,
`research/model_review.py` → `diagnostics/model_review/RESULTS.txt`.

**Evidence base.** All of it is hindsight on two priced seasons: 2024-25
(ESPN BET, 1,209 games) and 2025-26 (964 DraftKings games from 25 Nov, 245
ESPN BET before). Reconstructed rows are leave-one-season-out; for 2025-26
every fit uses earlier seasons only, and walk-forward costs only +0.0010 ±
0.0030 log loss against leave-one-season-out, so the rows are not flattered.
No native (pregame-locked) row has been graded with a model probability yet.
Against outcomes alone (no prices) the evidence is wider: nine walk-forward
seasons (2016-17 to 2025-26) and ten seasons no fit has seen (2004-05 to
2013-14), `research/history_backtest.py` (MODEL.md §6.4).
± is 95% throughout; log loss differences are per game, model − market
(positive = the market is better).

## 1. Bottom line

1. **A calibrated forecaster that is less informed than the close.** Against
   outcomes the model is as well shaped as the market (calibration slope 1.04
   ± 0.17 on DraftKings 2025-26, 0.97 ± 0.15 on ESPN BET 2024-25; the close
   1.06). It still trails the close by +0.0184 ± 0.0110 (DraftKings) and
   +0.0143 ± 0.0098 (ESPN BET) log loss in games 10+, and at the close its
   disagreement with the price carries no information: the outcome's weight
   on it is w = −0.06 ± 0.40 and +0.01 ± 0.37 (0 = adds nothing).
2. **Its information is early.** Against the opening line w is +0.33 ± 0.37
   and +0.44 ± 0.35: the same sign and size in both seasons, about a third of
   the model's disagreement. Lines then move toward the model by tip (value
   side +1.32 ± 0.35 pp of no-vig probability; spreads +0.56 to +0.65 pts,
   25–27% of the opening edge, resolved in both seasons). Any edge lives
   before the market finishes pricing (the open, the first snapshot), not at
   the close. (The +1.32 pp is 2025-26 DraftKings; the spread move is both
   seasons.)
3. **The box score is close to exhausted.** Four terms shipped after v1
   (season phase, injury-report availability, opponent 3-point luck with
   roster talent, free-throw %). Together they narrowed the gap to the close
   by about 0.01 to 0.015 per game (v4 vs the v1-style base −0.0090 /
   −0.0137; v5 and v6 together up to ~0.003 more). About fifteen other
   box-score and schedule ideas added nothing (§4). What the market knows
   and the model does not is mostly outside the box score: a persistent
   team-level view and the timing of player news.
4. **The model's own EV is not an edge.** At the close the blanket value
   bet claimed +11 to +16% and returned −6 to −11% against a −4% null
   (re-checked 2026-10-21 on v6; `research/dk_hypotheses.py` §7). On the
   model's side of the close it claims +6 pp of win probability and
   realises about −1 pp.

## 2. Strengths

| # | Strength | Evidence | Source |
|---|---|---|---|
| S1 | Calibrated against outcomes | Slope 1.04 ± 0.17 (DK 2025-26), 0.97 ± 0.15 (ESPN BET 2024-25); v5 pooled 1.02 (0.91–1.13); probit rejected | model_review §1; v5 audit C.2–C.3 |
| S2 | No home bias now | Model − market home P, games 10+: +0.26 ± 0.46 / +0.26 ± 0.49 pp; predicted − actual home win %, last 2 seasons +0.75 ± 0.97 pts (1 SE) | CLAUDE.md home-court; v5 audit B |
| S3 | Knows something the opener does not | w at the open +0.33 / +0.44; on the model's side at the open, claims of 12+ pp realise +9.7 / +9.5 pp (share ~0.6); line moves toward it by tip in both seasons | model_review §1, §3; ats_signal; open vs close |
| S4 | Not only injury timing | H3 at the DraftKings open without the injury-report terms: +15.2 ± 22.1 vs null (with them +11.2 ± 23.0); games 1–9, which have no report terms, also see the line move toward the model (+0.42 to +0.70 pts) | dk_hypotheses §5; ats_signal |
| S5 | Every shipped term earned its place out of sample | Phase −0.0036 ± 0.0036 / −0.0075 ± 0.0032; availability v3 − v2 −0.008 ± 0.010 in both seasons; v5 luck + talent −0.0036 / −0.0054 vs base v4; v6 FT% −0.0003 / −0.0010; carryover early season +0.02 behind the close vs +0.03 to +0.09 for this-season-only | MODEL.md §6 |
| S6 | Honest construction | Features strictly before the date, pregame lock, walk-forward ≈ leave-one-season-out, frozen hypotheses, a tag per formula, tests on lookahead | CLAUDE.md engineering rules |
| S8 | The logic and the shape hold on every season, not just the priced two | Walk-forward 2016–26: v6 beats the v4 core in 9 of 9 seasons (−0.0100 ± 0.0033); slope 0.99 ± 0.06. Frozen core on 2004–14: slope 1.03 ± 0.06, no worse than refitting it in that era (+0.0005 ± 0.0018); era weights correlate +0.96 with the committed ones | history_backtest A1–A2, B |
| S7 | Interpretable | The composite is a decomposition of four factors (r = 0.98 with margin); each later term has a basketball meaning | MODEL.md §6.3 |

## 3. Weaknesses

| # | Weakness | Evidence | Source |
|---|---|---|---|
| W1 | Trails the close wherever resolved | +0.0184 ± 0.0110 (DK), +0.0143 ± 0.0098 (ESPN BET), games 10+; Brier agrees; vs Kalshi +0.0200 ± 0.0114. The lean returns −1.7% vs a −4.1% null (DK), about 1 SE | model_review §1; kalshi_benchmark; ledger report |
| W2 | Loses most where it disagrees most | Gap by \|logit P − logit q\| at the close: +0.002 (< 0.1), +0.004 to +0.006 (0.1–0.25), +0.015 to +0.023 (0.25–0.5), +0.050 to +0.052 (> 0.5), same pattern in both seasons. Its big disagreements at the close are model error | model_review §2 |
| W3 | A persistent team-level view it cannot see | The market's correction is team-level (team-season R² 0.23–0.33), carries across seasons (r +0.40) and is present at the open; the factor gaps plus FT% explain 46–63% of it, the rest is outside the box score | CLAUDE.md team quality; v5 audit D |
| W4 | Coarse player information | Talent is last season's BPM (rookies at replacement) over the previous box score; the report gives Out/Doubtful only; newly-out players are double counted (≈ −1× the talent coefficient). Even perfect "who played" adds little over v5 (−0.0028 ± 0.0078 / +0.0014 ± 0.0079) | v5 audit E; roster minutes |
| W5 | Early season unresolved | Games 1–9: −0.0073 ± 0.0395 and +0.0092 ± 0.0420 vs the close (138 games each, ESPN BET); the carryover leans home +2.5 to +3.2 pp beyond the market. DraftKings priced no game 1–9, so H1 and H6 have no DraftKings hindsight | model_review §1; home-court; dk_hypotheses |
| W6 | Thin, in-sample evidence for the rules | Two priced seasons, one DraftKings season; the hypotheses came from a 271-cell scan where 3 passed (about 7 expected by chance); the DraftKings backtest re-reads the same games | CLAUDE.md hypotheses |
| W8 | The home edge is the training seasons' edge | Walk-forward 2017-18 to 2025-26 it ran 1.5 pts above the actual home rate (3–4 pts in 2020-21, 2021-22, 2023-24; within 1.3 in the last two); on 2004–14 the frozen core ran 2.8 pts below it. Recency weighting bought nothing (§4) | history_backtest A2, B |
| W9 | Opponent luck is small, but it stays | Pooled over 2016–26 and every route, dropped and refitted: +0.0002 ± 0.0009, helps in 6 of 9 seasons (talent +0.0059, back-to-back +0.0024). On the priced seasons, all on the availability route, dropping it fails the gate: v7 − v6 +0.0018 ± 0.0017 (2024-25 ESPN BET), +0.0013 ± 0.0028 (2025-26 DK). Owner kept v6, 2026-10-22 | history_backtest A7; walk-forward run 39698425564 |
| W7 | Segment results do not replicate | Heavy DK favourites (≥ 0.90) +0.050 ± 0.013 (n = 39) but −0.007 ± 0.034 in 2024-25; worst months differ by season. Treat single-season cells as noise until they replicate | model_review §2 |

## 4. Ruled out (do not retest without a new reason)

| Idea | Result (log loss vs the arm it extends, ± 95%) |
|---|---|
| 18 travel/altitude/rest features | 2.4% of the market's correction; worse jointly |
| Pace (possessions) | ±0.0003 |
| Strength of schedule | −0.0000 / −0.0002 |
| Late-season tank/top flags | −0.0010 ± 0.0023 / −0.0003 ± 0.0025 (held, not dropped) |
| Last season past game 10 | +0.0028 ± 0.0055 / +0.0041 ± 0.0048 (worse) |
| Recency-weighted intercept | within 0.0005 of v5 |
| A separate home back-to-back term | the residual flips sign between eras: −2 pts (2016–26), +2 pts (2004–14) |
| Dropping opponent luck (v7) | +0.0018 ± 0.0017 / +0.0013 ± 0.0028 vs v6 (worse; failed the gate) |
| Probit link | +0.0004 / +0.0003 (slightly worse) |
| Δ·\|Δ\| (curvature) | +0.0005 / −0.0007 |
| Eight free factor weights instead of Δ | +0.0004 / +0.0001 |
| Assists | +0.0006 / +0.0005 |
| In-season on/off | +0.0004 / −0.0003 |
| Win Shares (four variants) | none narrows the gap |
| Lineup minutes by position or role | within 0.0007 of proportional; one lineup term in place of v5's three: worse |
| Talent/luck fixed in the availability logit | worse (+0.0008 / +0.0016) |
| Kalshi instead of the sportsbook | the two closes carry the same information (−0.0018 ± 0.0015) |
| Blanket value bet and ATS twin | retired: claimed +11 to +16%, returned −6 to −11% |
| A winning in-sample price band as a filter | the DK −249 to −130 lean band did not survive the next model |

## 5. Guidelines

### Using the model's numbers

- **G1. Read P as a calibrated, less-informed forecast.** At the close the
  market's no-vig price is the better probability. Model − market at the
  close is a disagreement, not an edge (W1, W2).
- **G2. Never size a bet by the model's disagreement at the close.** Its
  realised share there is −0.19 ± 0.46 (DK) and −0.17 ± 0.48 (ESPN BET).
- **G3. Look for value early, and discount it.** At the open about a third
  of the model's disagreement is borne out (share +0.34 ± 0.47 / +0.37 ±
  0.45). Treat a claimed edge of x pp as roughly x/3, and expect part of it
  to be gone by the first snapshot (the line closes ~25% of the opening
  spread edge by tip).
- **G4. Bet only the frozen pre-registered rules, at their registered
  prices.** Bands, splits and cells (by price, spread, month, side) are
  descriptive monitoring, never filters (evidence rule 4).
- **G5. Judge results by book and season against the null and the
  same-game favourite, native apart from hindsight, and wait for power.**
  Bets a native test needs before its hindsight effect would sit 2 SE from
  the null (model_review §4):

  | Rule | Hindsight effect | Bets / season | Full effect | Half the effect |
  |---|---:|---:|---:|---:|
  | H5 (ATS, games 10+) | +7.1 pts (DK) | ~760 | ~730 (~1 season) | ~2,900 (~4) |
  | H3/H4 (edge ≥ 0.12) | +18.9 (DK) | ~130 | ~220 (~2) | ~870 (~7) |
  | H2/H2·F (value favourite) | +4.2 (DK) | ~490 | ~1,100 (~2) | ~4,500 (~9) |
  | H1 (games 1–9, edge ≥ 0.08) | +21.8 (ESPN BET) | ~60 | ~160 (~3) | ~630 (~11) |
  | H6 (ATS, games 1–9) | +8.1 (ESPN BET) | ~100 | ~560 (~6) | ~2,200 (~22) |

  Native bets are placed at the first snapshot, later than the open where
  the hindsight was measured, so expect effects nearer the "half" column.
  H5 is the only rule a single season can test; H1 and H6 cannot be judged
  for years.
- **G6. Use closing-line value as the early signal.** Whether the close
  moves toward a native bet is measurable on every bet long before its ROI
  resolves, and the hindsight says it should (S3). A rule whose native bets
  do not beat the close is unlikely to beat the null.

### Changing the model

- **G7. One gate for every change.** Walk-forward with the production
  routing, the same games, not worse on either large book-season (log loss,
  Brier agreeing), then refit, rescore and a new `MODEL_TAG`. A gain of more
  than 0.008 a season is a leakage tripwire, not a win. Read an ablation on
  the route and seasons production uses: the history backtest's pooled
  figure for luck (+0.0002) hid a cost of +0.0013 to +0.0018 on the
  availability route (W9).
- **G8. Stop mining the box score.** Four shipped terms bought ~0.01 to
  0.015 per game and about fifteen more bought nothing (§4). New work should target what the
  market has and the model lacks:
  1. *Player news and timing*: a real pregame roster (the
     `data/nba_injuries.csv` snapshots) and the newly-out arm (W4); the
     report's timing relative to the line.
  2. *The persistent team view*: the 37–54% of the market's team-level
     correction the factors do not explain (W3), with priors that are known
     before the season (roster changes, coaching), fitted on outcomes.
  3. *Early season*: priced games 1–9 accumulate slowly (~140 a season);
     the home lean (W5) is the first thing to test as they do.
- **G9. Fit on outcomes only.** A market residual may point at a feature,
  as it did for FT%; it is never the target.
- **G10. Re-run the checks after every rescore.** Rerun
  `research/model_review.py`, `research/dk_hypotheses.py` (§7, the retired
  bet) and the hypothesis scoreboard; keep native and reconstructed apart.
  After a change to prediction math also run the History backtest
  (workflow, 2 minutes on cached data): the change should not lose any
  season's term ablation or the out-of-era slope.

### Markets and data

- **G11. DraftKings is the market.** The pages, the report and the
  hypotheses grade at DraftKings (owner's decisions, 2026-10-21). Keep
  recording Kalshi's prices in the data; the benchmark shows the closes
  carry the same information, so this is a choice of bettable venue, not
  of better information.
- **G12. Protect the early prices.** The first snapshot (the morning build)
  and the open are where the model's information is worth something (G3).
  Their timing and completeness matter more than any refresh near tip.

## 6. Next tests, in order of value

1. **Native forward record** for H5, H4 and H2·F at the first snapshot,
   with native CLV per rule (G5, G6). H5 has the power to resolve in about
   a season.
2. **A pregame roster** from the injury snapshots, then the newly-out arm
   on it (G8.1).
3. **Disagreement shrinkage, as research only.** σ(logit q + w·(logit P −
   logit q)) with w fitted walk-forward at the open, against the raw P on
   the same games. It would quantify G3 per game; shipping it changes
   prediction math (a new tag).
4. **Early-season home lean** once a third priced season of games 1–9
   exists (W5).

Reproduce: `python research/model_review.py` (§1–4 above),
`python research/dk_hypotheses.py --open-price research/output/open_price.csv`
(DraftKings backtest and the retired-bet re-check), and the "History
backtest" workflow (`research/history_backtest.py`; S8, W8, W9).
