"""Historical reconstruction routes, retained for closed research only."""
import numpy as np
import pandas as pd
import build_site
import nba_composite as nc
import player_availability as pav
from research import legacy_cold_start as cold_start
WEIGHT_YEARS = [2015, 2016, 2017, 2018, 2019] + list(range(2021, 2027))
GAME_YEARS = [2023, 2024, 2025, 2026]
PHASE_YEARS = [2016, 2017, 2018, 2019] + list(range(2021, 2027))
REPORT_YEARS = [2023, 2024, 2025, 2026]
MODEL_COLUMNS = ["gp_home", "gp_away", "delta", "p_home", "lean", "p_lean",
                 "model_tag", "carry_p_home", "margin_home"]
# v7: the early (games 1-9) talent logit's seasons, the box-score seasons
# (research/early_roster.py; the first needs the BPM table of the season
# before it). The carryover logit keeps GAME_YEARS.
EARLY_TALENT_YEARS = [2016, 2017, 2018, 2019] + list(range(2021, 2027))


def training_years(years, y, walk_forward=False):
    """Seasons a fit for test season y may use: every other season (leave one
    season out), or only earlier ones (walk-forward)."""
    return [t for t in years if (t < y if walk_forward else t != y)]


def reconstruct_season(y, game_years=GAME_YEARS, weight_years=WEIGHT_YEARS,
                       phase_years=PHASE_YEARS, report_years=REPORT_YEARS,
                       terms=None, walk_forward=False, v5=False, talent=None,
                       v6=False, v7=False):
    """Predictions, routed as in production, for every qualifying game of
    season y: `route` is "early" (games 1-9, carryover), "avail" (games 10+
    the injury report covers) or "base" (other games 10+).

    Every fit leaves season y out; with walk_forward=True it uses earlier
    seasons only (research/walk_forward.py). `terms` ({season:
    player_availability.season_terms frame, covered games only}) enables the
    availability arm for games 10+ of report seasons; None skips it.

    v5=True adds luck_def and talent_diff to both games-10+ logits (talent:
    {season: talent function}; default player_availability.season_talent,
    from the cached box scores). As in the daily build, a game whose v5 terms
    are missing keeps the v4 base prediction and tag. p_v4 always holds the
    v4 routing's prediction for the same game (the v5 gate). With v5, p_fixed
    is the v5 routing with the availability logit's luck_def / talent_diff
    coefficients fixed from the v5 base fit (player_availability.fit_fixed);
    research only.

    v6=True (implies v5): model v6, v5 plus ft_diff (own FT% gap) in both
    games-10+ logits, as p_home with the v6 tags; p_v5 holds the v5
    routing's prediction for the same game (the v6 gate). Games whose v6
    terms are missing keep the v4 fallback, as in the daily build.

    v7=True (implies v6): games 1-9 add talent_diff to the early logit
    (cold_start.FEATURES_TALENT, fit on EARLY_TALENT_YEARS without y) and
    take the v7 early tag; games without talent keep the carryover P and the
    v6 tag. carry_p_home holds the carryover logit's P for every early game
    (H1 and H6 read it); blank for games 10+."""
    v6 = v6 or v7
    v5 = v5 or v6

    def use(years):
        out = training_years(years, y, walk_forward)
        if not out:
            raise SystemExit(f"season {y}: no training seasons in {list(years)}")
        return out
    memo = dict(talent or {})

    def tal(t):
        if t not in memo:
            try:
                memo[t] = pav.season_talent(t)
            except Exception as e:  # noqa: BLE001 - rows fall back to v4
                print(f"season {t}: no talent ({e!r})", flush=True)
                memo[t] = None
        return memo[t]

    def games_for(t):
        return nc.build_games(t, weights, talent=tal(t)) if v5 else nc.build_games(t, weights)

    weights = nc.fit_weights(use(weight_years))
    phase_train = use(phase_years)
    games = {t: games_for(t) for t in set(phase_train) | {y}}
    train = pd.concat([games[t] for t in phase_train])
    feats = nc.PHASE_FEATURES
    model = nc.fit_logit(train[feats].values, train["win"], feats)
    model["features"] = feats
    test = games[y].copy()
    test["p_home"] = nc.predict(model, test[feats].values)
    test["p_v4"] = test["p_home"]
    test["model_tag"] = build_site.MODEL_TAG_V4
    test["route"] = "base"
    if v5:
        f5 = nc.V5_FEATURES
        ok = np.isfinite(train[f5].to_numpy(float)).all(axis=1)
        model = nc.fit_logit(train.loc[ok, f5].values, train.loc[ok, "win"], f5)
        fin = np.isfinite(test[f5].to_numpy(float)).all(axis=1)
        test.loc[fin, "p_home"] = nc.predict(model, test.loc[fin, f5].values)
        test.loc[fin, "model_tag"] = build_site.MODEL_TAG_V5
        print(f"season {y}: v5 base on {int(fin.sum())}/{len(test)} games 10+ "
              f"(fit n={int(ok.sum())})", flush=True)
    test["p_fixed"] = test["p_home"]
    base5 = model
    test["p_v5"] = test["p_home"]
    if v6:
        f6 = nc.V6_FEATURES
        ok = np.isfinite(train[f6].to_numpy(float)).all(axis=1)
        m6 = nc.fit_logit(train.loc[ok, f6].values, train.loc[ok, "win"], f6)
        fin = np.isfinite(test[f6].to_numpy(float)).all(axis=1)
        test.loc[fin, "p_home"] = nc.predict(m6, test.loc[fin, f6].values)
        test.loc[fin, "model_tag"] = build_site.MODEL_TAG_V6
        model = m6
        print(f"season {y}: v6 candidate base on {int(fin.sum())}/{len(test)} games "
              f"10+ (fit n={int(ok.sum())}; ft_diff {m6['coef'][-1]:+.4f})", flush=True)
    if terms and y in terms:
        others = [t for t in training_years(report_years, y, walk_forward)
                  if t in terms]
        if others:
            tr = pd.concat([pav.with_terms(games[t] if t in games else games_for(t),
                                           terms[t]) for t in others])
            te = pav.with_terms(test, terms[y])
            tkey = (pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d")
                    + test["home"] + test["away"])
            key = te["slate_date"] + te["home"] + te["away"]
            arms = [(pav.FEATURES_V4, build_site.MODEL_TAG_V4_AVAIL, "p_v4"),
                    (pav.FEATURES_V5, build_site.MODEL_TAG_V5_AVAIL,
                     "p_v5" if v6 else "p_home")]
            if v6:
                arms.append((pav.FEATURES_V6, build_site.MODEL_TAG_V6_AVAIL, "p_home"))
            for af, tag, col in arms:
                if af is pav.FEATURES_V5 and not v5:
                    continue
                ok = np.isfinite(tr[af].to_numpy(float)).all(axis=1)
                am = nc.fit_logit(tr.loc[ok, af].values, tr.loc[ok, "win"], af)
                fin = np.isfinite(te[af].to_numpy(float)).all(axis=1)
                pmap = dict(zip(key[fin], nc.predict(am, te.loc[fin, af].values)))
                hit = tkey.isin(pmap)
                test.loc[hit, col] = tkey[hit].map(pmap).to_numpy(float)
                if col == "p_v5":                 # the v6 gate's v5 arm
                    test.loc[hit, "route"] = "avail"
                if col == "p_home" or not v5:
                    test.loc[hit, "p_home"] = tkey[hit].map(pmap).to_numpy(float)
                    test.loc[hit, "model_tag"] = tag
                    test.loc[hit, "route"] = "avail"
                print(f"season {y}: availability arm ({'v6' if af is pav.FEATURES_V6 else 'v5' if af is pav.FEATURES_V5 else 'v4'}) "
                      f"on {int(hit.sum())}/{len(test)} games 10+ (fit on {others}, "
                      f"n={int(ok.sum())})", flush=True)
                if af is pav.FEATURES_V5:
                    test.loc[hit, "p_fixed"] = test.loc[hit, "p_home"]
                    fm = pav.fit_fixed(tr.loc[ok], base5)
                    fx = dict(zip(key[fin], nc.predict(fm, te.loc[fin, af].values)))
                    test.loc[hit, "p_fixed"] = tkey[hit].map(fx).to_numpy(float)
                    print(f"season {y}: fixed-talent arm  " + "  ".join(
                        f"{f}={c:+.4f}" for f, c in zip(fm["features"], fm["coef"]))
                        + f"  (free fit: " + "  ".join(
                        f"{f}={c:+.4f}" for f, c in zip(am["features"], am["coef"])
                        if f in pav.FIXED_V5) + ")", flush=True)
    # games 1-9: the v2 carryover model, its logit also fit without y; v7
    # adds talent_diff in a second early logit (also without y).
    early = cold_start.fit_early(use(game_years), weights)
    e = cold_start.early_games(y, weights, min_gp=1, window=nc.MIN_GAMES,
                               talent=tal(y) if v7 else None)
    if len(e):
        e["p_home"] = nc.predict(early, e[cold_start.FEATURES].values)
        e["carry_p_home"] = e["p_home"]
        e["p_v4"] = e["p_home"]
        e["p_fixed"] = e["p_home"]
        e["p_v5"] = e["p_home"]
        e["model_tag"] = (build_site.MODEL_TAG_V6 if v6 else
                          build_site.MODEL_TAG_V5 if v5 else build_site.MODEL_TAG_V4)
        e["route"] = "early"
        ty = [t for t in use(EARLY_TALENT_YEARS) if tal(t) is not None] if v7 else []
        if ty and "talent_diff" in e:
            et = cold_start.fit_early(ty, weights, talent={t: tal(t) for t in ty})
            fin = np.isfinite(e["talent_diff"].to_numpy(float))
            e.loc[fin, "p_home"] = nc.predict(et, e.loc[fin, cold_start.FEATURES_TALENT].values)
            e.loc[fin, "model_tag"] = build_site.MODEL_TAG_V7_EARLY
            print(f"season {y}: v7 early on {int(fin.sum())}/{len(e)} games 1-9 (fit on "
                  f"{ty}, n={et['n_games']}; talent {et['coef'][-1]:+.4f})", flush=True)
        elif v7:
            print(f"season {y}: no talent for games 1-9; carryover P and v6 tag",
                  flush=True)
        test = pd.concat([test, e], ignore_index=True)
    if not v6:
        test["p_v5"] = test["p_home"]
    return test, weights, model


