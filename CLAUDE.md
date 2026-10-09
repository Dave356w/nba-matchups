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
- `data/ledger_report.txt` (`report.py`): the bot-written plain-text
  readout, as in the sibling repos. Quote live numbers from it (or the
  pages, which call the same `analysis` functions), not from old PRs.
- `data/nba_ledger.csv`: **native** rows, pregame-locked (snapshot < tip).
- `data/nba_reconstructed.csv`: **reconstructed** rows (leave-one-season-out,
  closing price). This is hindsight and is not forward evidence.
- `data/nba_preseason.csv`: **preseason** rows (basis "preseason"):
  exhibitions, same schema and pregame lock as native, scored by the
  carryover logit on last season's games only (`build_site.PRESEASON_TAG`,
  route "preseason"), priced from **Kalshi** (`kalshi.py`, book "kalshi":
  YES asks with the taker fee as American odds; close = the last 1-minute
  candle ending at or before tip), never a sportsbook. Shown only on
  `preseason.html`; never pooled with
  native or reconstructed rows, the hypotheses or any fit. Starters rest,
  so these rows say how far the offseason view travels, not regular-season
  skill.

- **Kalshi is the pages' market** (owner's decision, 2026-10-06, after
  `research/kalshi_benchmark.py`; results in
  `diagnostics/kalshi_benchmark/`). Native and reconstructed rows carry
  Kalshi's prices beside the sportsbook's (`ledger.KALSHI_COLUMNS`:
  `kalshi_pre_*` under the pregame lock, `kalshi_first_*` written once with
  `first_*`, `kalshi_open/close_*` by grading; seasons from
  `ledger.KALSHI_FROM_SEASON` = 2025-26, the first Kalshi lists).
  `analysis.market_view` prices a row by Kalshi when it is graded with a
  Kalshi close (or pending with a Kalshi pregame price), else by its
  sportsbook; the Ledger, Calibration, Model and Today pages and the report
  read that view, so 2024-25 stays on ESPN BET. The sportsbook columns are
  kept. The pre-registered hypotheses read the raw rows: they stay on the
  sportsbook prices they were registered on. Reconstructed Kalshi prices
  are filled by "Backfill history" with `kalshi` checked
  (`backfill_history.py --kalshi`), never by hand.

## Evidence rules

1. Native and reconstructed rows answer different questions. Report them
   with separate labels and counts; never pool them silently.
2. Compare the model with the market on the **same rows** using proper
   scores (Brier, log loss) and calibration. Accuracy or ROI alone does not
   show calibration. Each row's close comes from one market (`close_book`
   in `analysis.market_view`: Kalshi from 2025-26, else DraftKings, or ESPN
   BET before late Nov 2025); report markets separately.
   ESPN provider 59 (live odds) is in-game and never read. Every flat-ROI
   figure also carries the same-row market-favourite baseline (`fav_roi`,
   1u on the favourite of the same games at the same price).
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
- Keep `site-build` serialized (`cancel-in-progress: false`). The
  validate-and-commit step runs on `!cancelled()` (validation inside it), so
  a render failure never costs a pregame snapshot; the Pages upload stays
  success-only. Fit model and
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
  slot, and so on down the chart (no spill across roles). Walk-forward
  vs v5 and the close on the same covered games. **Results (2026-10-01,
  games 10+, log loss vs v5, ± 95%; 2024-25 ESPN BET n = 1,071 / 2025-26
  DK n = 964):**
  - The box listing is not the roster: only 6–12% of report Out players
    are on it, and 11–28% of Questionable players are missing (late
    scratches). Every `x_list_*` arm and v5_x therefore carries hindsight
    and is not a pregame candidate; `x_prev_od` is the only clean arm.
  - `x_prev_od`: −0.0005 ± 0.0063 / +0.0059 ± 0.0066 (worse, ~1.8 SE).
    Replacing v5's three terms with the one lineup term does not help.
  - Hindsight ceiling `x_hind` (who played): −0.0028 ± 0.0078 / +0.0014 ±
    0.0079. Even perfect participation adds little over v5's terms.
  - Position (`x_list_pos`) and role depth chart (`x_list_role`) vs
    proportional (`x_list_od`): within 0.0007 in both seasons, coefficients
    unchanged; the term moves by sd 0.23–0.38 points/48 against an sd of
    ~6.5. Where an Out player's minutes go is not what is missing.
  - The term duplicates talent_diff (r = +0.90 / +0.92); in v5_x the
    talent coefficient falls to 0.016 / 0.004. v5_x: −0.0023 ± 0.0049 /
    −0.0028 ± 0.0045, but it uses the listing. Gap to the close unchanged
    (+0.014 to +0.025 in every arm). 2025-26 ESPN BET (n = 107)
    unresolved. Not pursued; a truly pregame roster (e.g. the
    `data/nba_injuries.csv` snapshots) is the only open variant, with
    little upside per the hindsight ceiling.
- Early-season cold start: v2 (`cold_start.py`) ships the probe's
  carryover arm (ρ = 0.25) for games 1–9, labelled "early · carryover" on
  the card; game 0 still abstains. `research/cold_start_probe.py` remains
  the backtest (ρ, preseason κ, both) against the close on the same games.
  Watch the native early-season rows against it.
- ROI is the product goal: the ledger page grades 1u flat bets on the lean
  (the blanket value side, model P > no-vig q, and its ATS twin were retired
  from the pages and report 2026-10-06, owner's decision: at the close its
  claimed EV was +11 to +16% per book-season while it returned −6 to −11%,
  null ≈ −4%; ATS −2.7 to −7.4% vs −4.5%; `analysis.SHOWN_RULES`; value-side
  rules are tested only as the frozen hypotheses below), with model WP, market WP,
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
  - **A1** (amendment, owner's decision 2026-10-01, still before any native
    rows; v5 audit Task A): H2–H4 are also scored at the decision-time price,
    the first snapshot (`first_*`, which also records the injury-report
    edition and route behind its P: `first_report_utc`, `first_route`).
    H3 there is H4; H2 there is **H2·F** (`analysis.HYPOTHESES`). H2 and H3
    at `pre` are unchanged; no threshold moved.
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
- **Model v6** (owner's decision, 2026-10-01): v5 + ft_diff, the own FT%
  gap (decayed FTM/FTA, home − away), in both games-10+ logits. The four
  factors read free throws only as FTA/FGA. Source: v5 audit Task D
  (`diagnostics/v5_audit/RESULTS.md`): the market's correction loads on own
  FT% in both seasons (t 3.0 / 9.7). Gate (walk-forward `--avail --v6`, v6
  − v5, games 10+, ± 95%): −0.0003 ± 0.0008 (2024-25 ESPN BET), −0.0010 ±
  0.0011 (2025-26 DK), −0.0018 ± 0.0042 (2025-26 ESPN BET, n = 107); v6 −
  close +0.0160 / +0.0184. Unresolved per season; native rows are the
  forward test. **Active:** refit `f299ac6` (ft_diff 0.0151 base, 0.0175
  availability), rescore `f85e3ff` (2,418 rows re-tagged v6; leave-one-
  season-out v6 − v5 −0.0002 ± 0.0016 / −0.0010 ± 0.0011).
- v5 audit (`HANDOFF_v5_audit.md`, results in `diagnostics/v5_audit/`,
  workflow "v5 audit"), 2026-10-01: home-court drift +1.56 ± 0.79 pts
  pooled over the last 3 seasons (rule not met; no change); v5 is not flat
  against outcomes (slope 1.02, 0.91–1.13), probit rejected; stale talent:
  newly-out players ≈ −1× the talent coefficient, gain unresolved.
- Assists: the four factors read none; they reach the model only through
  BPM (talent_diff, av_bpm). `research/team_quality.py` arms `ast` (base +
  own and opponents' assist rate, 100 × AST / FG, decayed, home − away; vs
  base) and `v5_ast` (v5 base logit + the same; vs `v5_base` on the box
  seasons). Assists are parsed as a research-only extra stat
  (`nc.EXTRA_STATS`); COLS unchanged. **Results (2026-10-01, run
  36815424997, assists for all test games; log loss, ± 95%; 2024-25 ESPN
  BET n = 1,071 / 2025-26 DK n = 964):** `ast` vs base +0.0006 ± 0.0008 /
  +0.0005 ± 0.0005; `v5_ast` vs v5_base +0.0003 ± 0.0008 / +0.0001 ±
  0.0006 (2025-26 ESPN BET, n = 107: −0.0003 ± 0.0015 / +0.0006 ± 0.0021).
  Coefficients tiny (own +0.003 to +0.005 logit per assist-rate point) and
  the defensive term flips sign between arms; gap to the close and the
  team share of the market's correction unchanged. Assists add nothing
  beyond the four factors and BPM; dropped.
- In-season player values (owner's choice, 2026-10-01): talent_diff is
  last season's BPM, frozen, with unmatched players (rookies) at
  replacement. `research/team_quality.py` adds `onoff_diff`: minutes share
  × each player's season-to-date on/off from box plus-minus and the final
  margin (games before the date), shrunk by minutes / (minutes + 1000).
  Arms `v5_oo` (v5 base + onoff_diff; the fitted weights are the shrinkage
  of last season's value toward this season's) and `oo_luck` (luck +
  onoff_diff, no last-season value), both vs `v5_base`. **Results
  (2026-10-01, run 36816566421, on/off for all test games; log loss vs
  v5_base, ± 95%; 2024-25 ESPN BET n = 1,071 / 2025-26 DK n = 964):**
  `v5_oo` +0.0004 ± 0.0022 / −0.0003 ± 0.0015 (onoff_diff +0.011 / +0.010
  logit per point; talent 0.064 → 0.060 / 0.060 → 0.056, so the fitted
  shrinkage stays almost entirely on last season's value); `oo_luck`
  +0.0025 ± 0.0089 / +0.0015 ± 0.0078 (2025-26 ESPN BET, n = 107: +0.0031
  ± 0.0037 / +0.0101 ± 0.0215). corr(onoff_diff, talent_diff) +0.50
  (2025-26). Gap to the close and team share unchanged. In-season on/off
  adds nothing to last-season BPM and cannot replace it; dropped.
- Win Shares (owner's request, 2026-10-09, from a Gemini WS/48 outline):
  `research/win_shares.py` computes BBR's Win Shares from full ESPN box
  lines, games strictly before the date (2024-25 full season vs BBR's
  table: WS corr 1.000, MAE 0.03; WS/48 1000+ MP MAE 0.0009), and
  `research/team_quality.py --ws` swaps it into v6's base logit (no
  injury-report terms) in place of talent_diff's BPM, same roster and
  minutes share (not usage-weighted); replacement = BPM −2.0 mapped to
  WS/48 (0.064–0.070). Walk-forward, box seasons 2016-2019 + 2021-2025,
  games 10+, log loss vs v6_base, ± 95% (2024-25 ESPN BET n = 1,071 /
  2025-26 DK n = 964): last-season WS/48 `v6_ws` +0.0013 ± 0.0035 /
  +0.0048 ± 0.0038 (worse); Bayes prior → season to date, M0 = 750 MP
  (the outline's rule) `v6_wsb` +0.0022 ± 0.0051 / +0.0001 ± 0.0042;
  season to date only `v6_wso` +0.0007 ± 0.0077 / −0.0035 ± 0.0062;
  free mix `v6_ws_wso` +0.0015 ± 0.0042 / +0.0016 ± 0.0035; BPM kept +
  wsb `v6_bpm_wsb` +0.0012 ± 0.0025 / +0.0002 ± 0.0020 (BPM coefficient
  0.062 → 0.038). v6_base − close +0.0184 / +0.0219; no arm narrows it
  outside noise. 2025-26 ESPN BET (n = 107) unresolved. WS/48 does not
  beat last-season BPM; in-season box-score value adds nothing resolved.
  Dropped. (The outline's season-level MAE of 0.31 wins was fitted on the
  same season's outcomes, which Win Shares are built from.)
- Home-court drift (2026-10-01, reconstructed rows, mean home P, ± 95%):
  none. Games 10+: model − market +0.26 ± 0.46 pp (2024-25) / +0.26 ±
  0.49 pp (2025-26), logit +0.016 ± 0.023 / +0.014 ± 0.024; model and
  market within ~1 pp of the actual home rate (± 2.7). The intercept fit
  on 2016–2024 still matches the current home edge; no recency weighting.
  Games 1–9 (carryover): the model leans home +2.5 ± 1.6 / +3.2 ± 1.9 pp
  beyond the market (logit +0.12 ± 0.07 / +0.14 ± 0.08), but home teams
  won 58.7% of those 276 games (model 58.1 / 58.4%, market 55.6 / 55.2%,
  ± 7.7); shifting early logits toward the market worsens log loss
  (0.5940 → 0.5945–0.5964). Unresolved; a lean to watch, not a fix. H1's
  hindsight picks are 80 home / 35 away (n = 115), so H1 is partly an
  early-season home lean: report native H1 split home / away (the rule
  itself stays frozen).
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
