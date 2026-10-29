# Historical research and original registrations

Archived on 2026-10-23. These are dated historical entries, not current
production instructions. Current rules are in ../CLAUDE.md. Original
registration commit: `2e301d3`, 2026-10-23T19:46:58Z.

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
  not read it yet; an opened Picks card shows it (`build_site.injury_block`,
  display only).
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
  Sleeper projections (owner's request, 2026-10-22; `sleeper.py`; its CLI
  writes `research/output/sleeper/`): RotoWire's projected
  minutes per player per game, Sleeper's depth chart and injury
  designations. Archive from 2019-20, complete (team-games 240 ± 1–8 min)
  from 2022-23. Each week's rows carry one `last_modified` after the week,
  so the archive is hindsight; against the ESPN box it does not look
  edited after the game (2025-26, 1,231 games: minutes played corr 0.82,
  MAE 4.7; 911 DNPs kept a projection, 2,781 players listed at 0 played).
  `snapshot` + `pregame` (fetch-stamped, read only before the ledger's
  tip) are the clean version, and the daily build now records it
  (`data/nba_sleeper.csv`, below). Model update?
  (owner's question; `research/sleeper_signal.py`, `diagnostics/
  sleeper_signal/RESULTS.txt`; stack on logit P or logit q, weights from
  the other priced season, games 10+, ± 95%, 2025-26 DK n = 964 / 2024-25
  ESPN BET n = 1,060): projected minutes × v6's player values (`s_tal`)
  adds −0.0006 ± 0.0018 / −0.0004 ± 0.0017 to v6 and nothing to the
  close. RotoWire's projected points margin (`s_margin`) is the market:
  corr +0.98 with the closing spread, weight on the close twice the open
  (+3.8 vs +1.9, resolved), nothing added to the close (−0.0000 ±
  0.0002); its −0.0134 ± 0.0093 / −0.0094 ± 0.0090 "gain" over v6 puts
  0.05–0.16 of the weight on v6, i.e. it imports the price (above G7's
  0.008 leakage tripwire). Neither is a model update; `s_tal` also
  follows the close, not the open (+5.1 to +7.0 vs −0.8 to +0.4), so the
  lineup news it carries is priced by tip.
  Fantasy-projection team signals (owner's DK Showdown work, March 2026,
  outside the repo; `research/fantasy_signal.py`, `diagnostics/
  fantasy_signal/RESULTS.txt`): TOP3_HARMONIC_ALL, the top three DK
  projections weighted 1, ½, ⅓ over the same series (per-depth
  denominator), delta home − away; on ~58 Showdown games r = 0.42 with the
  margin, ~74% directional. On every priced game (Sleeper archive,
  hindsight; 2025-26 DK n = 964 / 2024-25 ESPN BET n = 1,198): picks the
  winner 66.4 / 63.4% vs the close favourite 69.4 / 69.9% on the same games
  (r with the margin +0.41 / +0.35 vs the closing spread's +0.53 / +0.52;
  partial r given the spread −0.02 ± 0.06 / +0.01 ± 0.06); March's delta
  floors 4 / 6 win 74.0 / 76.9% and 68.1 / 70.4%, the favourite as much or
  more on the same games; its close underdogs +1.0% ± 17.4 / +1.6% ± 16.1
  ROI vs null −4.1%; added to the close 0.0000 in every season; added to
  v6, games 10+, −0.0044 ± 0.0031 (DK) / −0.0011 ± 0.0058, information the
  close already has. March's best configs (top-1, top-2 equal-weight) do
  worse here (top-1 65.1 / 60.7%). No signal beyond the price; not a model
  update.
- Early-season cold start: v2 (`cold_start.py`) ships the probe's
  carryover arm (ρ = 0.25) for games 1–9, labelled "early · carryover" on
  the card; game 0 still abstains. `research/cold_start_probe.py` remains
  the backtest (ρ, preseason κ, both) against the close on the same games.
  Watch the native early-season rows against it. Roster information for
  games 1–9 (owner's request, 2026-10-22; `research/early_roster.py`,
  workflow "Early roster", run 39830317895, `diagnostics/early_roster/
  RESULTS.txt`): the early route sees no players. Adding the games-10+
  `talent_diff` (unchanged) to the early logit, walk-forward on games 1–9 of
  2016-17 to 2025-26 (n = 1,219): −0.0105 ± 0.0087 log loss vs outcomes
  (Brier agrees), better in 8 of 9 seasons, coefficient 0.072–0.078 from
  2018-19 on. Above G7's 0.008 tripwire, checked: walk-forward throughout,
  box score strictly before the date, and it narrows but does not beat the
  close (2024-25 / 2025-26 ESPN BET, n = 138 each: early − close −0.0080 /
  +0.0125, early_tal − close −0.0124 / +0.0068; early_tal − early −0.0045 ±
  0.0220 / −0.0057 ± 0.0248). The market's correction of the early model
  follows talent_diff (corr +0.42 ± 0.12, n = 276). H1's hindsight profit
  sat in bets roster talent was against (n 77, +32.4% ± 33.2) more than
  those it backed (n 38, −11.8% ± 33.3), unresolved. **Model v7** (owner's
  decision, 2026-10-22; not the rejected v7 candidate of PR #80): shipped for games 1–9 (tag `..._carry25_talent_
  early_v7`; games 10+ keep v6), with **H1 and H6 kept on the carryover
  model**: every early row records the carryover logit's P
  (`ledger.CARRY_COLUMNS` `carry_p_home`, refreshed until tip;
  `first_carry_p_home`, written once with `first_*`; reconstructed rows via
  the rescore), and `analysis.carry_view` hands it to H1 / H6 (and the H6
  card badge). The carryover logit is frozen in `model/logit_early_v6.json`
  (also preseason, and early games without talent). MODEL.md "v7".
  **Unified model, v8 candidate** (owner's request, 2026-10-22: one model
  from day 1 to the end of the season; `research/unified.py`, workflow
  "Unified model (v8)", run 39867001658, `diagnostics/unified/RESULTS.txt`,
  PR Dave356w/nba-matchups#93): one rating for every game, last season's
  totals × rho(i) = 0.25 × max(0, 1 − i/K) with one decay across the
  offseason (i = the team's games played; game 0 = last season's rating),
  luck and FT% on the same totals; v6's base and availability logits fitted
  on every game (availability from g = 1); game 0's roster from the
  preseason box scores (ESPN, last-season role × BPM; corr +0.94 with the
  opener's roster talent, 285 team-seasons). Walk-forward 2016–26 against
  production's routing rebuilt on the same games (the rebuilt inputs match
  the production builders to 1e-14). The rule fixed before the run (lowest
  pooled log loss on games 1+) picks K = 10: games 1+ −0.0003 ± 0.0008
  (Brier −0.0001 ± 0.0003, n = 10,767, better in 6 of 9 seasons); games
  1–9 −0.0015 ± 0.0056, 10–19 −0.0011 ± 0.0017, 20+ +0.0001 ± 0.0004;
  no tripwire. Close, same games: 2024-25 ESPN BET −0.0011 ± 0.0025
  (Brier −0.0004), 2025-26 DK −0.0009 ± 0.0010 (Brier −0.0003): passes
  G7; 2025-26 ESPN BET games 10+ (n = 107) +0.0057 ± 0.0051 (Brier
  +0.0028 ± 0.0020), its games 1–9 −0.0077 ± 0.0199. Game 0 (n = 143,
  production abstains): −0.0145 ± 0.0214 vs the carryover logit without a
  roster, −0.067 ± 0.047 vs the home edge; no game-0 row is priced. The
  game-10 jump in a team's rating shrinks (sd of the game 9 → 10 change 7.4
  → 4.1 win-% points; K = 20 / 40 smoother still, 3.1 / 2.7, within 0.0002 in
  accuracy). talent × last season's share adds nothing. Equal accuracy, one
  formula, every game projected. **Model v8** (owner's decision,
  2026-10-22; MODEL.md "v8"): shipped as tested (K = 10). One route for
  every game (`build_site.score_game_v8`, `nba_composite.fade_v8_games`):
  the availability logit on covered games from game 1, the base logit
  elsewhere (game 0 included), both fitted on every game (`fit-logit --v8`,
  `player_availability.py fit --v8`); a missing term falls back to the
  carryover logit before game 10 (v2 tag) or the frozen v4 logit after.
  Game 0's roster is this season's preseason box scores
  (`player_availability.update_preseason_box`, `data/nba_box_pre_<season>
  .csv`, fetched once by the build from opening night, frozen). H1 / H6
  stay on the carryover P (`carry_p_home`, games 1–9); v7's
  `model/logit_early.json` is retired. Tags `..._fade10_phase_luck_talent_
  ft_v8` / `..._ft_avail_v8`. **Active since 2026-10-22:** preseason boxes
  `9f9ac3f` (10 seasons, 0 failed requests, 30 teams each), refit `567753d`
  (base n = 11,687, availability n = 4,847), rescore `3f10659` (2,450
  rows: every row v8, 1,209 a season on the availability logit; the 32
  game-0 rows v7 never wrote added, all ESPN BET; `carry_p_home` identical
  on the 276 early rows, so H1 n 115 +17.8% and H6 n 201 +3.6% stand).
  First native v8 row: DEN @ OKC, 2026-10-22, game 0 for DEN. Game 0 is
  in no hypothesis (`analysis.stage`: H1 / H6 read games 1–9, H2–H5 games
  10+; the Ledger's lean folds show it as "Game 0"). Re-check on the v8
  rows (G10; `diagnostics/model_review`, `diagnostics/dk_hypotheses`, ± 95%):
  model − close, games 10+, +0.0174 ± 0.0112 (2025-26 DK; v6 +0.0184) /
  +0.0152 ± 0.0101 (2024-25 ESPN BET; v6 +0.0143); games 1–9 −0.0225 ±
  0.0253 / −0.0009 ± 0.0260 (ESPN BET 2024-25 / 2025-26), unresolved; the
  DraftKings backtest vs null at the open H2 +4.5 ± 6.7 (n = 417), H3/H4
  +22.9 ± 24.4 (n = 124), H5 +6.6 ± 7.1 (n = 697), at the close −3.6 /
  +7.0 / +0.5; the retired value bet at the DK close −12.7% (null −4.1%).
  Data note: in 2015-16 to 2017-18, 161–163 ESPN box scores a season
  record no minutes; talent then reads the team's last box with minutes
  (both arms, every fit since v5).
- ROI is the product goal: the ledger page grades 1u flat bets on the lean
  (the blanket value side, model P > no-vig q, and its ATS twin were retired
  from the pages and report 2026-10-06, owner's decision: at the close its
  claimed EV was +11 to +16% per book-season while it returned −6 to −11%,
  null ≈ −4%; ATS −2.7 to −7.4% vs −4.5%; `analysis.SHOWN_RULES`; value-side
  rules are tested only as the frozen hypotheses below; re-checked 2026-10-21
  on the v6 rows, owner's request, `research/dk_hypotheses.py` §7: claimed
  +11.5 / +16.2 / +16.0%, returned −6.1 / −11.0 / −6.9% (2024-25 ESPN BET /
  2025-26 DK / 2025-26 ESPN BET), null −4.1 to −4.2%; ATS −7.4 / −6.1 /
  −3.2% vs −4.5%; no exact P = q ties), with model WP, market WP,
  actual, and the ROI null beside every ROI. Graded rows now also record the
  closing spread (`close_spread`, home line, plus both spread prices, same
  book as the moneyline close), and the ledger page grades ATS the same way
  (lean and value side, break-even and ROI null beside each ROI).
  It also grades both sides by the picked price band (EV beside its null,
  z vs the ROI null); native rows are banded at the pregame price and the
  v3 rows are shown alone. ATS is banded the same way (owner's request,
  2026-10-21): the Ledger's ATS folds (lean, closing spread) and the H5/H6
  bets (native beside the DraftKings hindsight; the report likewise) by the
  picked side's line and spread price (`analysis.ATS_LINE_BANDS`,
  `ATS_PRICE_BANDS`, `ats_by_band`; descriptive, never a filter; no SE or z
  below `BAND_MIN_N` = 10 bets). The earlier band hypothesis (DraftKings leans at
  −249 to −130) did not survive v4: −1.8% at the close, −3.2% at the open,
  both at the ROI null (2025-26); it is retired.
- **First-snapshot amendment (owner's plan, registered 2026-10-23T19:46:58Z).**
  Registered before CHI @ MIL (2026-10-23) is final: the fetched native
  ledger at `b4fcb67` has no result for game 401909848 (tip 20:00Z).
  This registration is committed on its own before implementation or
  computing the new segments' hindsight figures.
  - Keep exactly three forward tests, each at the frozen **first snapshot**:
    **H5** (spread), **H2·F** (moneyline value favourite), **H4** (moneyline
    value side with P − no-vig q ≥ 0.12). H5 keeps sigma 13.5 × Φ⁻¹(P),
    edge ≥ 1 point and both main-line prices −135..+115. No thresholds,
    price filters or games-10+ membership change.
  - Extend each to games **1+** by adding its own **games 1–9 segment**,
    using the primary model P (v9), never the carryover model P. Games
    10+ keep their existing registered rows. Show games 1–9 and games
    10+ separately, never pooled, until at least one season of each exists
    (any later pooling requires an explicit amendment). Game 0 remains
    outside all hypotheses. Existing first snapshots stay immutable,
    including their recorded model tags and probabilities.
  - Retire **H1**, **H6**, and **H2/H3 at the late pregame price** as of
    this timestamp. Keep all four on the Ledger under **Retired**, with
    their registered hindsight figures and their native bets to date;
    none has a graded native bet at registration. Do not admit later
    snapshots or grade these bets into any active forward record.
    H1/H6 retain their historical carryover P for archival reproduction.
  - The early segments replace H1/H6. Compute their hindsight on the
    **276 ESPN BET early games** using v9 rows, for information only;
    label this hindsight explicitly, apart from DraftKings and the native
    forward test. No threshold selection or new forward evidence.
  - Early v9 uses injury-report inputs: these new segments inherit the
    injury-timing caveat of H2–H5. The first snapshot is later than the
    open and is not an exact replica of the opening-price backtest.
  - Ledger scoreboard, report, Results/Performance spread tiles and Picks
    badges use these same active rules and separate stage rows. Test
    membership by stage, retirement retention/exclusion, and frozen
    thresholds. Run data validation and the full suite; capture page
    screenshots. This amendment supersedes conflicting active-rule
    descriptions below; the original registrations remain as history.
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
  - **H5** (owner's request, fixed 2026-10-20, before any native row had a
    model P; first expected 2026-10-22): games 10+, against the spread.
    The model's home margin 13.5 × Φ⁻¹(P) (σ frozen in the rule, not read
    from `analysis.ATS_SIGMA`); bet its side when |margin + home line| ≥ 1
    pt; main-line prices only (both sides −135..+115,
    `analysis.ATS_MAIN_LINE`). Native at the first snapshot (`first_spread`,
    its prices, `first_p_home`); hindsight at the same book's open
    (`diagnostics/ats_signal/open_spreads.csv`), pooled as registered:
    +2.2% (n = 1,528, null −4.5%, z +2.7 at 1 SE); +2.1% 2024-25 ESPN BET (n =
    743), +2.5% 2025-26 DK (n = 703), −0.4% 2025-26 ESPN BET (n = 82); the
    same rule at the close −5.5% (n = 1,550). Why 1 pt: the smallest whole
    point, where both seasons agree (vs null +6.6 ± 6.8 / +7.1 ± 7.0, 95%),
    not the best cell (DK 2 pts, z +2.1). Caveats as H4: the first snapshot
    is later than the open, so part of the line move (25–27% of the opening
    edge by tip) may be gone, and games 10+ may be injury timing.
  - **H6** (owner's request, fixed 2026-10-20 with H5, still before any
    native row had a model P): H5's rule unchanged (σ 13.5, ≥ 1 pt,
    main-line prices, first snapshot; hindsight at the open) on games 1–9,
    the carryover model with no injury-report terms, so no timing
    contamination. Hindsight at the open, ESPN BET, pooled: +3.6% (n = 201,
    null −4.5%, z +1.2 at 1 SE); +9.1% 2024-25 (n = 86, vs null +13.6 ±
    20.2, 95%), −0.5% 2025-26 (n = 115, +4.0 ± 17.5): unresolved and not
    consistent across seasons; at the close −4.5% (n = 222, the null). The
    threshold was not re-chosen for these games. About 100 bets a season:
    one season is far from a verdict.
  - The Ledger page and the report also show H5 and H6 by basis × the
    graded price's book × season (`analysis.hypothesis_by_season`, the same
    bets, never pooled), under the scoreboard's pooled-as-registered row.
  - DraftKings backtest (owner's question, 2026-10-21;
    `research/dk_hypotheses.py`, results `diagnostics/dk_hypotheses/
    RESULTS.txt`): the frozen rules on the 964 reconstructed 2025-26 games
    with a DK close. ESPN lists DK from 2025-11-25 only, so there is no
    2024-25 DK and no DK game 1–9: H1 and H6 cannot be tested on DK.
    Hindsight, ROI − null ± 95%: at the open H2 +4.2 ± 6.7 (n = 412), H3/H4
    +18.9 ± 25.0 (n = 119), H5 +7.1 ± 7.0 (n = 703, z +2.0); at the close
    −2.3, −0.3, +0.1 (the null). The same open bets at the close price keep
    +1.4 / +11.3 / +4.1 (the line moves toward them +1.6 pp / +4.1 pp /
    +0.64 pts). H3 and H5 on DK sit in late Nov–Jan (+58 ± 45, n = 41;
    +10.8 ± 10.3, n = 323) more than Feb–Apr (−1.7 ± 29, n = 78; +3.9 ±
    9.6, n = 380). Without the injury-report terms (Open vs close run
    36754366300, v4 rows) H3 at the DK open is +15.2 ± 22.1 (v4 +11.2 ±
    23.0) and H2 +2.1 ± 7.8 (v4 +4.2 ± 7.7): not lost to injury timing on
    DK, unresolved. Not an independent test: these games were in the scans
    that chose the rules; native rows are the test. The Ledger page (and
    the report) show it as the "DraftKings backtest"
    (`analysis.hypothesis_backtest`: open, close, the open bets at the
    close with the line move, by half-season), and since 2026-10-21 the
    scoreboard's hindsight and the H5/H6 by-season rows are DraftKings
    only (owner's decision; ESPN BET is no longer offered), so H1 and H6
    show no hindsight there. The registered pooled figures above stand.
- Open vs close (hindsight, 2025-26 DK, games 10+): at the close the model
  adds nothing (w = −0.13 ± 0.41, the outcome's weight on the model's
  disagreement with q); at the open w = +0.29 ± 0.37, about the ~0.27 a
  value bet needs to clear a 4% hold, and the value side gains +1.32 ±
  0.35 pp of no-vig probability open → close (+2.4 / +1.1 pp in games 1–9,
  which have no injury terms). Unresolved; native CLV is the test.
- Against the spread (owner's question, 2026-10-20): `research/ats_signal.py`
  (results `diagnostics/ats_signal/RESULTS.txt`) maps the reconstructed P to
  a margin, 13.5 × Φ⁻¹(P) (`analysis.ATS_SIGMA`, not tuned; the market's own
  moneyline-to-line scale is 12.9), and bets the model's side when |margin
  + home line| ≥ 0–5 pts, at the row's close and at the same book's open
  (ESPN core odds, cached in `open_spreads.csv`; 71 ESPN BET "opens" priced
  outside −135..+115 were alternate lines and are dropped; every line is a
  half point, so no pushes). Hindsight, ± 95%, games 10+ (2024-25 ESPN BET
  n = 1,028 at the open / 1,071 at the close; 2025-26 DK n = 964):
  - Close: nothing. Signal slope (cover margin on edge) −0.06 ± 0.28 /
    −0.12 ± 0.28; edge ≥ 0 ROI −8.2% / −6.1% vs null −4.5% (the retired
    ATS value side).
  - Open: slope +0.31 ± 0.28 / +0.29 ± 0.29; edge ≥ 0 covers 53.0% /
    52.3%, ROI +1.1% / −0.3% (vs null +5.5 ± 5.8 / +4.2 ± 6.0); edge ≥ 1
    +2.1% / +2.5% (vs null +6.6 ± 6.8 / +7.1 ± 7.0). Near break-even after
    the hold; unresolved.
  - The line moves to the model's opening side by tip: +0.65 ± 0.13 /
    +0.56 ± 0.14 pts, closing 27 ± 4% / 25 ± 4% of the opening edge (home
    side, same games, −0.05 / +0.04). Resolved in both seasons; what is
    left at the close carries nothing. Games 1–9 (no injury terms, ESPN
    BET, n = 110/138): +0.70 ± 0.41 / +0.42 ± 0.36 pts, ROI vs null
    unresolved (+10.8 ± 17.9 / +4.1 ± 16.0).
  - 3 of 60 ROI cells clear |z| ≥ 1.96 (about 3 by chance); none
    replicates across seasons. Same reading as the moneyline open vs close:
    the model knows something the opener does not and the market prices
    it by tip; in games 10+ part may be injury-report timing. Native rows
    record the pregame and first-snapshot spread from 2026-10-20 (see
    "Pregame spread" above), so a forward ATS test can be graded at a
    bettable price. H5 and H6 (above, under the hypotheses) are that test.
- Margin model (owner's question, 2026-10-23: can a model fitted on
  margins give the spread, and is it also accurate for win probability?
  `research/margin_model.py`, workflow "Margin model", run 39997006384,
  `diagnostics/margin_model/RESULTS.txt`): v8's inputs, routes and
  walk-forward seasons (unified.py, fade10), OLS on the final home margin
  per route, P = Φ(m / σ) with σ the route's training residual SD (11.7 →
  13.5 over the folds), plus a probit reading. Walk-forward 2016–26, games
  1+ (n = 10,767), M − v8, ± 95%: log loss −0.0008 ± 0.0010, Brier −0.0003
  ± 0.0004, better in 4 of 9 seasons; calibration slope 1.08 (v8 0.95);
  probit reading −0.0004 ± 0.0006 (slope 0.96). Close, same games, games
  10+: 2024-25 ESPN BET −0.0014 ± 0.0030 (Brier −0.0005), 2025-26 DK
  −0.0004 ± 0.0028 (Brier −0.0001): passes G7, no tripwire, unresolved;
  M − close +0.0144 / +0.0170 (v8 +0.0158 / +0.0174). 2025-26 ESPN BET
  games 1–9 (n = 138) +0.0080 ± 0.0073, worse, small sample. Spread: M's
  margin beats v8's 13.5 × Φ⁻¹(P), MAE −0.065 ± 0.027 pts (RMSE 13.46 vs
  13.53; home bias +0.37 vs +0.78), resolved pooled, not per book-season;
  it trails the close by +0.24 ± 0.15 / +0.29 ± 0.19 MAE and ties the open
  (−0.13 ± 0.19 / +0.04 ± 0.19). ATS slope on the edge at the open +0.54 /
  +0.34 (v8 +0.48 / +0.31), at the close −0.11 / −0.09, the same reading as
  v8. The probit's margin scale 1/b is 11.8–12.3, not 13.5. Equal accuracy
  for win probability, a slightly better spread. **Model v9** (owner's
  decision, 2026-10-23, "ship it"; MODEL.md "v9"): shipped as tested. Both
  v8 routes fitted on the margin (`nba_composite.fit_margin`, `fit-logit
  --v9`, `player_availability.py fit --v9`; model files `"link":
  "margin"`, `predict` → Φ(m / σ)); rows record the model's margin
  (`ledger.MARGIN_COLUMNS` `margin_home`, a pregame column, refreshed until
  tip, frozen; blank on logit rows); tags `..._ft_margin_v9` /
  `..._ft_margin_avail_v9`; the fallbacks unchanged. The card's spread lean
  uses `margin_home` (`analysis.spread_lean`, `ats_sides(own_margin=True)`,
  display only); H5 / H6 keep 13.5 × Φ⁻¹(P), as registered. **Active since
  2026-10-23:** refit `42aa0ab` (base n = 11,687, σ 13.30; availability
  n = 4,847, σ 13.64), rescore `537cb10` (2,450 rows v9, margin on every
  row; `carry_p_home` unchanged, H1 n 115 +17.8%, H6 n 201 +3.6%), rebuilt
  season `98e5f4c`. Re-check on the v9 rows (G10;
  `diagnostics/model_review`, `diagnostics/dk_hypotheses`, ± 95%): model −
  close, games 10+, +0.0170 ± 0.0107 (2025-26 DK; v8 +0.0174) / +0.0130 ±
  0.0093 (2024-25 ESPN BET; v8 +0.0152); the DraftKings backtest vs null at
  the open H2 +6.2 ± 7.7 (n = 324; v8 +4.5, n = 417), H3/H4 +18.2 ± 28.9
  (n = 112; v8 +22.9), H5 +7.0 ± 7.1 (n = 696; v8 +6.6); the retired value
  bet at the DK close −11.3% (null −4.1%; v8 −12.7%). Unresolved, as
  before; native rows are the test.
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
- **v7 candidate rejected** (owner's request 2026-10-21 to drop opponent
  luck after the history backtest; owner's decision 2026-10-22 to keep v6):
  v6 without `luck_def`, closed PR Dave356w/nba-matchups#80 (code at
  `79f6270`). Gate
  (walk-forward `--avail --v7`, run 39698425564, v7 − v6 on the same games,
  games 10+, ± 95%): +0.0018 ± 0.0017 (2024-25 ESPN BET), +0.0013 ± 0.0028
  (2025-26 DK), −0.0094 ± 0.0151 (2025-26 ESPN BET, n = 107); Brier agrees
  on both large book-seasons; v7 − close +0.0178 / +0.0197 (v6 +0.0160 /
  +0.0184). Worse on both large book-seasons, resolved on 2024-25: fails
  the gate. Every priced game 10+ runs through the availability logit,
  where luck's coefficient is larger (0.016–0.018 vs 0.0125 base); the
  history backtest's pooled ablation (+0.0002, below) is mostly the base
  route of 2016-17 to 2022-23 and understated it. Judge a term on the
  route and seasons production uses. Luck stays.
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
- In-season BPM (owner's question, 2026-10-22; `research/bpm_live.py`,
  `research/team_quality.py --bpm`, workflow "Team quality" with args
  `--bpm`, run 39823000315; `diagnostics/bpm_live/RESULTS.txt`):
  production's player values are last season's BPM, frozen (minutes
  update, values do not). BPM to date is rebuilt from ESPN box lines
  (league-relative per-100 rates + team net rating, mapped on BBR's BPM of
  2016–2024: held-out corr 0.954–0.969, MAE 0.57–0.67, no bias) and blended
  walk-forward. Games 10+, log loss vs v6_base, ± 95% (2024-25 ESPN BET n =
  1,071 / 2025-26 DK n = 964): Bayes (prior M0 = 750 min) `v6_bpmb`
  +0.0049 ± 0.0038 / −0.0002 ± 0.0038 (worse on 2024-25, Brier agrees);
  season only `v6_bpmo` +0.0039 ± 0.0066 / −0.0010 ± 0.0062; frozen +
  in-season, free weights `v6_bpm_bpmo` +0.0011 ± 0.0012 / −0.0004 ±
  0.0008 (in-season weight 0.013–0.021 vs the frozen term's 0.054). BPM to
  date overlaps the team rating (delta's weight 0.017 → 0.011–0.012); what
  talent_diff adds is the pre-season roster view. Base route only. Dropped,
  like on/off and Win Shares. Small-sample players (owner's follow-up, run
  39826197099): production's shrinkage extended by this season's minutes,
  (500 R + MP0 BPM0 + MP obs) / (500 + MP0 + MP), predicts a player's own
  second-half rebuilt BPM better than the frozen value in every group
  (rookies MAE 1.95 → 1.66, under 500 last-season MP 1.97 → 1.62; same box
  metric, so flattering), but not games: updated for players under 500
  last-season minutes only `v6_bpmn` +0.0002 ± 0.0016 / −0.0000 ± 0.0018
  (corr 0.986 with talent_diff: small minutes shares); for everyone
  `v6_bpmp` +0.0033 ± 0.0025 / −0.0004 ± 0.0026 (worse on 2024-25).
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
- History backtest (owner's question, 2026-10-21; `research/
  history_backtest.py`, workflow "History backtest", results
  `diagnostics/history_backtest/RESULTS.txt`, MODEL.md §6.4): the model
  against outcomes only, games 10+, on every season the logs reach (no
  prices, so nothing about the market; stats.nba.com does not answer from
  the runners, so BBR and ESPN as in production). Walk-forward v6 routing
  2016-17 to 2025-26 (n = 9,509): v6 − v4 core −0.0100 ± 0.0033 log loss,
  better in 9 of 9 seasons. Terms dropped in turn (positive = helps):
  talent +0.0059 ± 0.0026, b2b +0.0024 ± 0.0012, Δ·phase +0.0009 ± 0.0006
  (8 of 9 each), availability +0.0029 ± 0.0033 (3 of 3), FT% +0.0004 ±
  0.0005 (8 of 9), opponent luck +0.0002 ± 0.0009 (6 of 9, the weakest
  pooled; but dropping it fails the gate on the priced seasons, v7 above).
  Calibration slope 0.99 ± 0.06 (folds with 2+ training seasons); Δ·phase
  takes the last third from 1.24 (v1) to 1.04. Predicted home rate 1.5 pts
  above the actual (2017-18 to 2025-26). Out of era 2004-05 to 2013-14
  (n = 10,480), the frozen committed core: slope 1.03 ± 0.06, home teams
  won 2.8 pts more than predicted; frozen − full in-era refit +0.0005 ± 0.0018; era weights
  correlate +0.96 with the committed ones. The logic and the shape transfer;
  the home intercept moves with the era (recency weighting already ruled
  out). The home-team back-to-back residual flips sign between eras
  (−2 / +2 pts): not a lead. Takes 2 minutes on cached data; rerun after
  any model change. Follow-ups (owner's request 2026-10-22, the adjustments
  ranked in the guidelines; run 39700532844): (A8) an in-season home-edge
  update (this season's residual on earlier dates, empirical-Bayes prior,
  tau ~0.04 logit) −0.0001 ± 0.0002, vs the close +0.0000 / +0.0001: not
  worth a change; (A9) drafted rookies valued by pick (rookie BPM ≈ −1.26
  − 0.45 log pick, earlier seasons only) −0.0002 ± 0.0002, better in 9 of
  9, vs the close −0.0001 ± 0.0006 / −0.0002 ± 0.0006: passes the gate but
  small, for the next model version; (C) games 1–9 against outcomes,
  walk-forward 2016–26 (n = 1,258): home lean +0.2 ± 2.6 pts, slope 0.88 ±
  0.18; frozen early logit on 2005–14: −0.1 ± 2.6, 0.90 ± 0.17. The early
  home lean beyond the market is not a bias against outcomes.
- Disagreement shrinkage at the open (owner's request 2026-10-22;
  `research/open_shrinkage.py`, `diagnostics/open_shrinkage/RESULTS.txt`):
  P* = σ(logit q_open + w (logit P − logit q_open)), w from the other priced
  season (+0.43 / +0.36). Games 10+, ± 95%: P* − open −0.0009 ± 0.0046
  (2025-26 DK, walk-forward) / −0.0026 ± 0.0037 (2024-25 ESPN BET,
  hindsight); P* − P −0.0081 ± 0.0067 / −0.0047 ± 0.0064; realised /
  claimed edge 0.81 / 1.02 (raw P 0.34 / 0.37). Bets on P*'s side at the
  posted open with claimed EV > 0: −7.5% ± 12.0 / +5.7% ± 13.2 (null
  −4.1%), unresolved and opposite; the line moves toward the bet in step
  with the claim in both seasons. P* is the honest reading of an edge; no
  bet rule from it without pre-registration. Games 1–9: w unstable (0.38 /
  0.97, n = 138). The Picks card's edge and EV use it (w 0.40, display
  only, 2026-10-22). Not registered as a hypothesis (owner asked for it
  only if it could replace one): "H7", the value side at the first
  snapshot when P*'s EV > 0, would replace H2·F on logic (same question,
  same price, with the hold), but in hindsight at the open it is +3.9 vs
  the null pooled (−2.5 DK, +9.0 ESPN BET) against H2's +4.7 (+4.2 / +5.1),
  and its underdog bets swing ~1.5 units each against ~0.7: ~5,800 bets
  (~10 seasons) to resolve its effect against ~1,100 (~2) for H2·F. H2·F
  stays.
- In-game Kalshi, strength vs reaction (owner's notebook
  `nba_strength_vs_reaction.ipynb`, 2026-10-22; `research/
  strength_vs_reaction.py`, `diagnostics/strength_vs_reaction/RESULTS.txt`):
  p* = Φ(Δ/(σ√τ) + z0√τ), a Brownian live price from ESPN play-by-play
  (wallclock) with z0 from the Kalshi pregame mid (MKT), the model's P
  (MODEL, reconstructed v6) or a z-blend fitted on train (w 0.32 ± 0.43),
  σ = 17.3 from the score process, against the home market's live 1-minute
  Kalshi mid. 2025-26, 1,161 games: train to 2026-01-31 (681), test Feb–Apr
  (480); ± 95% game-clustered; hindsight model rows. Test: pregame MODEL −
  MKT +0.0341 ± 0.0186 log loss, BLEND − MKT +0.0055 ± 0.0060; in-game
  p*(MKT) − live mid +0.0011 ± 0.0062, p*(MODEL) − p*(MKT) +0.0145 ±
  0.0101; the outcome's weight on p*'s disagreement with the mid +0.40 ±
  0.43 (MKT), −0.08 ± 0.32 (MODEL). Trades (first quiet minute with p* −
  ask − fee ≥ 4c, filled at the worst quote 2 min later, per $1 contract,
  claimed +5.5 to +8.1c): MKT −4.4c ± 4.7 (n = 337), MODEL −5.5c ± 4.0
  (n = 418), BLEND −4.5c ± 4.5 (n = 354), null −2.9 to −3.2c. Every
  notebook rule fails: no strength edge (the model trails the pregame mid,
  as at the close) and no reaction edge after the spread and fee. On the
  same minutes the mid's fitted σ (21.7–22.6) exceeds the outcome's
  (17.8–19.8; +2.8 ± 2.3 train, +3.9 ± 2.4 test), most in the first half
  and unresolved in the 4th quarter, while the pregame mid's own slope moves
  0.82 → 1.28 between the splits: not clean evidence that the mid
  under-reacts to the score, and p* does not beat it either way. The
  score-process σ overstates the game's (regulation margin around the
  spread 13.85 ± 0.56) because consecutive quarters correlate −0.12 (leads
  shrink). ESPN's wallclock and the mid agree within a minute (corr of
  minute changes 0.69 at lag 0, 0.10 a minute later). Not pursued.
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
- Questionable weighting and lock lead (owner's request, 2026-10-28;
  `research/questionable_weight.py`, [results](../diagnostics/questionable_weight/RESULTS.txt)).
  NBA reports 2022-23 to 2025-26, rotation players: absences already
  flagged Out/Doubtful at 18 h before tip 56–57% (back-to-backs 23–26%,
  others 71–72%), flat to 8 h, then 76–79% at 6 h and 78–84% at 5 h (the
  teams' ~1 pm local game-day reports), 85–89% at 2 h. Questionable
  rotation players played 53–63% at every lead. Walk-forward on v10's
  report route, 2023-24 to 2025-26, games 1+ (± 95%): od at 18 h minus od
  at 30 min +0.0036 ± 0.0023 (back-to-back +0.0064, other +0.0024); at
  5 h +0.0016 ± 0.0016. Questionable at its play rate (q) minus od:
  −0.0012 ± 0.0010 at 18 h, −0.0006 ± 0.0012 at 5 h; at 5 h DraftKings
  2025-26 +0.0007, so q **fails** the G7 rule set before the run (both
  leads). Refits at the lock lead (odL, qL) fail too. No historical price
  at the lock lead exists, so the price side rests on the thin native
  path; any lock change needs a registered amendment.
