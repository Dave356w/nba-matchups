"""Legacy v2-v7 scoring, retained only for historical research."""
import numpy as np
import pandas as pd
import cold_start
import nba_composite as nc

def score_legacy_game(logs, home, away, date, weights, model, half_life=None,
               early=None, prior_logs=None, avail_model=None, avail=None,
               opening=None, talent=None, fallback=None, carry=None, talent0=None):
    """Pregame composite and P(home win) from games strictly before `date`.

    From MIN_GAMES games on: the base logit. Below that, when both teams have
    played at least once and `early` (model/logit_early.json) and last
    season's logs are given: the carryover model (cold_start.py). Otherwise
    (game 0, or no early model) it abstains: delta/p NaN.

    v3: from MIN_GAMES on, when `avail_model` (model/logit_avail.json) and this
    game's availability terms `avail` ({'av_min', 'av_bpm'}) are given, the
    availability logit is used and the row is tagged MODEL_TAG_V3.

    v4: models whose features include d_phase get delta * season phase, with
    phase measured from `opening` (default: the earliest date in `logs`, i.e.
    opening night). Every scored row carries model_tag (see model_tag()).

    v5: models whose features include luck_def / talent_diff get opponent
    3-point luck (nba_composite.opp_luck, league 3P% before `date`) and
    talent(team, date) (player_availability.talent_fn), home - away. When a
    term is missing the game is scored by `fallback` (the frozen v4 base
    logit) and tagged v4; without a fallback it abstains.

    v6: models whose features include ft_diff also get the own FT% gap
    (nba_composite.own_ft_pct over each team's games before `date`), home -
    away; same fallback.

    v7: an `early` logit with talent_diff (games 1-9) gets talent(team, date)
    like the games-10+ logits and the MODEL_TAG_V7_EARLY tag; without talent
    the game is scored by `carry` (the carryover logit, default `early`
    itself when it has no talent) and keeps the v6 tag. Every early row also
    carries carry_p_home, the carryover logit's P (H1 and H6 read it).

    v8 models (is_v8) are scored by score_game_v8 (talent0: preseason rosters).
    """
    from build_site import (is_v8, score_game_v8, is_early_talent, model_tag,
                            MODEL_TAG_V7_EARLY)
    if is_v8(model):
        return score_game_v8(logs, home, away, date, weights, model, prior_logs,
                             avail_model, avail, opening, talent, talent0,
                             fallback, carry, half_life)
    hl = model.get("half_life", nc.HALF_LIFE) if half_life is None else half_life
    date = pd.Timestamp(date)
    out = dict(gp_home=np.nan, gp_away=np.nan, home_b2b=np.nan,
               away_b2b=np.nan, delta=np.nan, p_home=np.nan, lean=np.nan,
               p_lean=np.nan)
    if home not in logs or away not in logs:
        return out
    lh, la = logs[home], logs[away]
    ih, ia = int((lh["date"] < date).sum()), int((la["date"] < date).sum())
    out.update(gp_home=ih, gp_away=ia)
    rh, ra = nc.rest_days(lh, ih, date), nc.rest_days(la, ia, date)
    if min(ih, ia) >= nc.MIN_GAMES:
        fh = nc.decayed_features(lh, ih, hl)
        fa = nc.decayed_features(la, ia, hl)
        d = nc.composite(fh - fa, weights["sd"], weights["w"])
        use, out["model_tag"] = model, model_tag(model)
        if avail_model is not None and avail is not None:
            use, out["model_tag"] = avail_model, model_tag(avail_model, avail=True)
    elif min(ih, ia) >= 1 and early is not None and prior_logs:
        d = cold_start.carry_delta(lh, ih, prior_logs.get(home), la, ia,
                                   prior_logs.get(away), weights,
                                   rho=early.get("rho", cold_start.RHO),
                                   half_life=early.get("half_life", nc.HALF_LIFE))
        if d is None:
            return out
        if carry is None and not is_early_talent(early):
            carry = early
        use, out["model_tag"] = early, (MODEL_TAG_V7_EARLY if is_early_talent(early)
                                        else model_tag(model))
        fallback = carry          # the carryover logit, not the games-10+ v4 one
    else:
        return out
    vals = {**nc.logit_inputs(d, rh, ra, date, nc.season_opening(logs)
                              if opening is None else opening), **(avail or {})}
    feats = use["features"]
    if "luck_def" in feats:
        pct = nc.league_3p_before(logs, date)
        vals["luck_def"] = nc.opp_luck(lh, ih, pct, weights, hl) \
            - nc.opp_luck(la, ia, pct, weights, hl)
    if "talent_diff" in feats:
        vals["talent_diff"] = (talent(home, date) - talent(away, date)) if talent \
            else float("nan")
    if "ft_diff" in feats:
        vals["ft_diff"] = nc.own_ft_pct(lh, ih, hl) - nc.own_ft_pct(la, ia, hl)
    if not all(np.isfinite(vals.get(f, np.nan)) for f in feats):
        if fallback is None:
            return out
        use = fallback
        out["model_tag"] = model_tag(model) if use is carry else model_tag(fallback)
        feats = use["features"]
    p = float(nc.predict(use, [[vals[f] for f in feats]])[0])
    lean_home = p >= 0.5
    out.update(home_b2b=int(rh == 0), away_b2b=int(ra == 0),
               delta=round(d, 3), p_home=round(p, 5),
               lean=home if lean_home else away,
               p_lean=round(p if lean_home else 1 - p, 5))
    return out


