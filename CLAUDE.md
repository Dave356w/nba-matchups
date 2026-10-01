# Claude — working agreement for NBA Matchups

This is the sibling of `Dave356w/Dave356w` (XWOBA MLB Matchups), and the same
working agreement applies: be a rigorous, constructive collaborator; inspect
real code and data; recognize measured progress; describe uncertainty in
proportion to the evidence; and turn each observation into the next test. The
owner directs the product.

## Sources of truth

- `MODEL.md` + `nba_composite.py`: the model. `build_site.py`: the daily
  pipeline and pages. `market.py`: all price arithmetic, in one place.
  `ledger.py`: ledger invariants. `analysis.py`: calibration statistics.
- `data/nba_ledger.csv`: **native** rows, pregame-locked (snapshot < tip).
- `data/nba_reconstructed.csv`: **reconstructed** rows (leave-one-season-out,
  closing price). This is hindsight and is not forward evidence.

## Evidence rules

1. Native and reconstructed rows answer different questions. Report them
   with separate labels and counts; never pool them silently.
2. Compare the model with the market on the **same rows** using proper
   scores (Brier, log loss) and calibration. Accuracy or ROI alone does not
   show calibration. Each row's close comes from one book (`close_book`:
   DraftKings, or ESPN BET before late Nov 2025); report books separately.
   ESPN provider 59 (live odds) is in-game and never read.
3. No-vig q and posted break-even are different thresholds. An EV figure
   (win% − break-even) is centred on `market.ev_null` (q − break-even, about
   minus the hold), **not zero**. Print the null beside every EV figure.
4. Price bands are descriptive monitoring dimensions, not filters. An
   in-sample winning band is a hypothesis to test forward.
5. An interval crossing zero means the sample has not resolved the effect;
   it does not establish zero effect.

## Engineering rules

- **No lookahead.** Features use games strictly before the slate date. A
  pregame row is written or refreshed only before tip and is frozen after.
  A pending game never receives a closing line. `tests/test_ledger.py` and
  `tests/test_scoring.py` enforce these rules.
- Grading writes only result and open/close columns (`ledger.GRADE_COLUMNS`).
- A change to prediction math requires a new `MODEL_TAG`. A display change
  does not.
- Keep `site-build` serialized (`cancel-in-progress: false`). Fit model and
  Backfill history each have their own queue (a pending run in `site-build`
  is cancelled by the next hourly build) and retry their push. Tests gate
  PRs and are deliberately not wired into the daily build.
- Do not hand-commit bot-generated `data/` changes or `public/`.
- Pages: ± is 1 SE and labelled so; research scripts and the figures in
  this file are ± 95%. One colour rule on the ledger and calibration pages
  (`build_site.sig_cls`): colour marks a gap of ≥ 2 SE from its null (ROI
  vs the ROI null, a forecast vs the diagonal, the model vs the close),
  never sign alone. Sections split basis × closing book × season. The lean
  is the recorded `lean` column everywhere (`analysis.lean_is_home`).
- First snapshot (`ledger.FIRST_COLUMNS`): filled once from the first
  pregame snapshot with a model P and a price, never by a refresh or by
  grading, frozen at tip. `analysis.picks(price="first")` grades it with
  `first_p_home`. The reconstructed rows leave it blank.

Before a PR, run:

    python validate_data_files.py
    python -m pytest tests/ -q

## Open research questions (from the report §7–8)

- Market benchmark: Brier/log loss vs the DK close, and native CLV.
  Production v5, walk-forward (MODEL.md §6.1): trails the close by
  +0.0163 ± 0.0099 (2024-25 ESPN BET) and +0.0194 ± 0.0112 (2025-26 DK) log
  loss, games 10+ (v4: +0.0163 / +0.0216); games 1–9 and the 107 early 2025-26 ESPN BET games are
  unresolved. No demonstrated edge; native rows are the test.
- Player availability (injury reports, minutes shares) is the main missing
  information. `research/availability.py` (workflow "Availability ceiling")
  measures its HINDSIGHT ceiling: who actually played, relative to each
  team's rating window, added to walk-forward logits and compared with the
  close on the same games (v1: on/off-weighted; v2: last-season BBR BPM,
  name-matched, plus players new to the team). `research/pregame_availability.py`
  (workflow "Pregame availability") is the pregame version: the NBA's
  archived injury report, last edition at least --lead-minutes before tip
  (Out/Doubtful = out; or status play rates fitted on training seasons),
  same v2 values, and how often each status actually played. **Model v3/v4**
  (`player_availability.py`, `model/logit_avail.json`) ships its Out/Doubtful
  arm for games 10+: the build reads the latest NBA report and this season's
  box scores (`data/nba_box_<season>.csv`, extended daily). Rows the report
  does not cover (`player_availability.covers`: matchup on it, no team NOT
  YET SUBMITTED; an empty or unparsed report covers nothing) fall back to
  the base logit (v4 tag without "avail"), in the build, the fit and the
  backfill alike. Historically 4,922 of 4,923 games were covered.
  The daily build snapshots each game's ESPN injury list to
  `data/nba_injuries.csv` under the pregame lock (replaced only before tip,
  frozen after; "NONE" rows mark teams with nobody listed). The model does
  not read it yet.
  `research/roster_minutes.py` (workflow "Roster minutes", 2026-10-01
  audit) tests one lineup term in place of v5's talent_diff + av_bpm +
  av_min, so nothing is counted twice: pregame roster (previous box, or
  game k's box listing as the roster known before tip) x report
  participation (od, or q play rates) x expected minutes scaled to 240
  (cap 42) x last-season BPM value; plus a hindsight ceiling and v5 + the
  term. `x_list_pos` sends an Out player's minutes by position (last
  season's BBR Pos, PG=1..C=5, weight max(0, 1 − |Δpos|/2): a C's to C/PF,
  a PG's to PG/SG) instead of in proportion to minutes, on the same base.
  `x_list_role` is the owner's DK Showdown notebook rule: G/W/B roles,
  ranked by minutes; the next man up in the Out player's role takes his
  slot, and so on down the chart (no spill across roles). Walk-forward vs v5 and the close on the same covered games. It
  prints how often report Out players are on the box listing (the check
  that the listing is the roster, not who dressed). Not yet run.
- Early-season cold start: v2 (`cold_start.py`) ships the probe's
  carryover arm (ρ = 0.25) for games 1–9, labelled "early · carryover" on
  the card; game 0 still abstains. `research/cold_start_probe.py` remains
  the backtest (ρ, preseason κ, both) against the close on the same games.
  Watch the native early-season rows against it.
- ROI is the product goal: the ledger page grades 1u flat bets on the lean
  and on the value side (model P > no-vig q), with model WP, market WP,
  actual, and the ROI null beside every ROI. Graded rows now also record the
  closing spread (`close_spread`, home line, plus both spread prices, same
  book as the moneyline close), and the ledger page grades ATS the same way
  (lean and value side, break-even and ROI null beside each ROI).
  It also grades both sides by the picked price band (EV beside its null,
  z vs the ROI null); native rows are banded at the pregame price and the
  v3 rows are shown alone. The earlier band hypothesis (DraftKings leans at
  −249 to −130) did not survive v4: −1.8% at the close, −3.2% at the open,
  both at the ROI null (2025-26); it is retired.
- **Pre-registered forward hypotheses** (fixed 2026-09-30, before any native
  rows; thresholds may not be tuned on native data). Hindsight scan of 271
  context × band × season cells: 3 passed a naive 95% test, about the 7
  expected by chance. What replicated across seasons:
  - **H1** games 1–9, value side with model P − no-vig q ≥ 0.08: +17.8% ±
    25.0 ROI at the close (null −4.0%, n = 115; +24% 2024-25, +13% 2025-26,
    ROI rising with the threshold). Carryover model, no injury terms, so no
    timing contamination. Native test: the same rule at the pregame price.
  - **H2** games 10+, value side that is the favourite, at the open: +1.6%
    ± 4.6 (null −4.2%, n = 921); gone at the close (−3.7%).
  - **H3** games 10+, value side with P − q ≥ 0.12, at the open: +18.7% ±
    16.0 (n = 339); +1.0% at the close.
  H2/H3 exist only at the open, and the v4 availability terms use the
  report ≥ 30 min before tip, after the line opened, so they may be injury
  timing. `research/open_price.py` (workflow "Open vs close") re-scores
  without the report terms (`base`) to test that; if `base` loses them,
  they are retired. Report every hypothesis's native ROI beside its null,
  win or lose, with n; one season is not a verdict. The ledger page's
  scoreboard (`analysis.HYPOTHESES`, frozen) does this: native n and ROI at
  the pregame price beside the hindsight rule at its scan price, recomputed
  on the current reconstructed rows. On the v5 rows H1 is unchanged (+17.8%,
  n = 115); H2 is +0.4% ± 4.3 (n = 978) and H3 +19.7% ± 17.1 (n = 262),
  95%, vs the v4 figures above.
  - **H4** (fixed 2026-10-01, still before any native rows): H3's rule
    (games 10+, value side, P − q ≥ 0.12) graded at the **first
    snapshot**: the `first_*` ledger columns, the earliest pregame snapshot
    with a model P and a price (usually the 10:07 ET build on game day),
    written once and frozen; the model P is the one written then, from the
    injury report available then. Why: the build refreshes the pregame row
    hourly until tip, so the `pre` price ends near the close, where no
    threshold beats the null in hindsight (games 10+: −12.5% at edge ≥ 0
    to −2.5% at ≥ 0.12, 2025-26 DK); H2/H3 at `pre` may fail for that
    reason alone. Hindsight proxy: H3 at the open, +19.7% ± 17.1 (n = 262,
    v5 rows), rising with the edge in both seasons (≥ 0.12: +23.4% ± 13.8
    2024-25 ESPN BET, +15.1% ± 12.0 2025-26 DK, 1 SE); above 0.15 the
    samples collapse (n 15–17, signs flip), so the threshold stays 0.12.
    The first snapshot is later than the open and uses the morning report,
    so it is the bettable version of the open test, not a replica.
- Open vs close (hindsight, 2025-26 DK, games 10+): at the close the model
  adds nothing (w = −0.13 ± 0.41, the outcome's weight on the model's
  disagreement with q); at the open w = +0.29 ± 0.37, about the ~0.27 a
  value bet needs to clear a 4% hold, and the value side gains +1.32 ±
  0.35 pp of no-vig probability open → close (+2.4 / +1.1 pp in games 1–9,
  which have no injury terms). Unresolved; native CLV is the test.
- More game-log seasons to settle the half-life. Travel/altitude/rest
  context (hindsight scan, 2024-26): 18 schedule features explain 2.4% of
  the market's correction and worsen 2025-26 log loss jointly; not pursued.
- Team quality the four factors misread: the market's correction (logit q −
  logit P, sd 0.42) is largely team-level and persistent (team-season
  effects R² 0.23–0.33; same teams across seasons, r = +0.40), present at
  the open (not late news), and borrowing the market's past team view
  closes 18–39% of the log-loss gap. `research/team_quality.py` (workflow
  "Team quality") tests, walk-forward on the same games vs the close:
  (1a) 3-point luck — own and opponents' 3P% regressed to the league (3PA
  parsed from the cached BBR logs, `nc.EXTRA_STATS`; the model's COLS are
  unchanged); (1b) strength of schedule; (2) late-season tank/top flags and
  delta·[April] (April favourites: z +2.35 beyond the model, +2.15 beyond
  the market in the scan). **Results (2026-09-30, walk-forward, games 10+,
  log loss vs base v4; 2024-25 ESPN BET n = 1,071 / 2025-26 DK n = 964):**
  - (1a) luck: −0.0021 ± 0.0028 / −0.0023 ± 0.0033 (pooled ≈ −0.0022 ±
    0.0021, ~2 SE; Brier agrees). Fitted noise share of **opponents' 3P%
    0.71 / 0.78**, own 3P% −0.03 / +0.07 (skill): opponents' 3-point
    shooting in the rating window is mostly luck the model counts as
    defence. Closes ~8% of the gap to the close; v5 candidate (opponent
    3P% regressed), not shipped yet.
  - (1b) schedule: −0.0000 / −0.0002, coefficient +0.0035 per composite
    point. Not what is missing; dropped.
  - (2) late season: −0.0010 ± 0.0023 / −0.0003 ± 0.0025; signs stable
    (tank −0.10 / −0.13, April top team −0.32 / −0.39 logit). Unresolved;
    hold for more seasons.
  - None moves the market's team-level correction (team share 0.24 →
    0.20–0.24), so the persistent team disagreement is not luck, schedule
    or incentives.
  - (1d) last-season carryover past game 10: worse, +0.0028 ± 0.0055 /
    +0.0041 ± 0.0048; last season's weight fades from 1.2–1.5× this
    season's at opening night to ~0 late, so by game 10 it is spent.
    Dropped (games 1–9 keep the v2 carryover).
  - (1c) roster talent (minutes share × last-season BPM over the previous
    box score): the strongest lead. It takes about half the rating's weight
    (delta coefficient 0.031 → 0.018), and improves both seasons vs base fit
    on the same box seasons: −0.0013 ± 0.0109 / −0.0040 ± 0.0088; with
    prior and opponent luck −0.0029 / −0.0073 ± 0.0095 (gap to the DK close
    +0.029 → +0.022). Unresolved: only 2–3 box seasons to train on
    (base_box is itself worse than base). `research/box_history.py`
    (workflow "Box-score history") builds 2015-16 on, so the Team quality
    run can fit talent on the same seasons as base.
  - Rerun on the full box history (2015-16 on, talent fit on the same
    seasons as base): talent −0.0031 ± 0.0096 / −0.0032 ± 0.0087, stable
    coefficient (0.066 / 0.062; delta 0.024 → 0.015); talent + prior + luck
    −0.0036 ± 0.0098 / −0.0054 ± 0.0089 (pooled ≈ −0.0046 ± 0.0066).
    Unresolved per season, same sign and size in both.
  - **Model v5** (owner's decision, 2026-09-30): opponent luck + talent in
    both games-10+ logits (MODEL.md "v5"). Games whose v5 terms are missing
    use the frozen v4 base logit (`model/logit_v4.json`, v4 tag).
    **Active since 2026-09-30.** Gate (walk-forward `--avail --v5`, v5 − v4
    on the same games, games 10+): −0.0001 ± 0.0086 (2024-25 ESPN BET),
    −0.0022 ± 0.0068 (2025-26 DK), +0.0081 ± 0.0247 (2025-26 ESPN BET,
    n = 107); v5 − close +0.0163 / +0.0194. Refit `d765264` (base: talent
    0.058, luck 0.0125, delta 0.023 → 0.016; availability: talent 0.041,
    luck 0.016), rescore `a3b5415` (2,418 rows re-tagged v5). Unresolved;
    native rows are the forward test.
  - Tested and rejected: fixing talent/luck in the availability logit at
    the base fit's coefficients (`prod_fixed`, `player_availability.
    fit_fixed`, an offset) is worse than the shipped free fit on the same
    games: +0.0008 ± 0.0006 (2024-25 ESPN BET), +0.0016 ± 0.0018 (2025-26
    DK), −0.0013 ± 0.0065 (2025-26 ESPN BET, n = 107). The lower talent
    coefficient there (0.041 vs 0.058) is its overlap with av_bpm (both
    built from last-season BPM), which is also why v5's gain over v4 (which
    already has av_bpm) is below the team-quality arms (measured vs base v4
    without availability). The free fit stays.
- Walk-forward: `research/walk_forward.py` (workflow "Walk-forward
  backtest") re-scores reconstructed seasons with every fit on earlier
  seasons only, beside the leave-one-season-out rows and the close on the
  same games, plus a possession-based pace arm (games 10+). Its `prod` arm
  is the shipped v4 routing (`backfill_history.reconstruct_season(...,
  walk_forward=True)`: early carryover, base v4, availability v4 on covered
  games with `--avail`), split by route. Walk-forward costs +0.0010 ±
  0.0030 vs leave-one-season-out, so the reconstructed rows are not
  materially flattered. Pace adds nothing. Research only.
- One feature builder: `nba_composite.logit_inputs` (delta, b2b_net, phase
  from opening night, d_phase) serves build_games, the CLI scorer and
  `build_site.score_game`; research scripts reuse the `phase` column.
- Calibration shape: in-sample, the outcome's slope on the model logit is
  ~1 overall but ~1.4–1.9 in March–April (too flat) and ~0.8 before March.
  `research/calibration_shape.py` (workflow "Calibration shape") tests
  delta·|delta|, delta·phase and delta·[Mar+] arms walk-forward against the
  close on the same games. Walk-forward (logits 2016–2024/25): delta·phase
  beats base by −0.0036 ± 0.0036 (2024-25 ESPN BET) and −0.0075 ± 0.0033
  (2025-26 DK) log loss (reproduced exactly with the opening-night clock,
  2026-09-30); delta·|delta| adds nothing. The Pregame availability
  workflow also fits base+phase and od+phase to test it on top of v3.
  **Model v4** ships delta·phase in both logits (MODEL.md; tags
  `..._phase_v4` without a report, `..._phase_avail_v4` with one). The
  reconstructed rows are re-scored with v4 by `backfill_history.py
  --rescore` (workflow "Backfill history", prices and results kept).
