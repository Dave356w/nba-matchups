"""Pregame NBA prop estimates and a frozen archive for DFS comparisons."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import math
import os
import re
from statistics import NormalDist, linear_regression, median
import tempfile

import ledger
import market

PATH = "data/nba_dfs_market.csv"
FIELDS = ("game_id", "team", "name", "stat", "source", "tip_utc",
          "snapshot_utc", "mean")
FEEDS = {
    "Bovada": "https://www.bovada.lv/services/sports/event/v2/events/A/description/basketball/nba",
    "Underdog": "https://api.underdogfantasy.com/v2/over_under_lines?product=fantasy&sport_id=NBA",
}
UNDERDOG_STATS = {
    "Points": "pts", "Rebounds": "reb", "Assists": "ast",
    "Steals": "stl", "Blocks": "blk", "Turnovers": "to",
}
BOVADA_STATS = {
    "Total Points": "pts", "Total Rebounds": "reb", "Total Assists": "ast",
    "Total Steals": "stl", "Total Blocks": "blk", "Total Turnovers": "to",
}


def estimate(quotes):
    """Fit a normal location/scale to ladders, or a Poisson mean to sparse prices.

    Each quote is (line, no-vig over probability). At a whole number the
    probabilities describe outcomes conditional on the bet settling.
    """
    by_line = defaultdict(list)
    for line, probability in quotes:
        by_line[line].append(probability)
    points = sorted((line, median(probs)) for line, probs in by_line.items())
    half = [(line, p) for line, p in points if line % 1 == .5]
    if len(half) >= 3:
        quantiles = [NormalDist().inv_cdf(1 - p) for _, p in half]
        if len(set(quantiles)) >= 2:
            scale, mean = linear_regression(quantiles, [line for line, _ in half])
            if scale > 0 and mean >= 0:
                return mean

    def mean_at_price(line, target):
        count = int(line)
        low, high = 0., 3 * line + 20
        for _ in range(28):
            mid = (low + high) / 2
            mass = cdf = math.exp(-mid)
            for k in range(1, count + 1):
                mass *= mid / k
                cdf += mass
            resolved = 1 - mass if line == count else 1
            over = (1 - cdf) / resolved if resolved > 1e-12 else 0
            if over < target:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    return median(mean_at_price(line, p) for line, p in points)


def paired_quotes(outcomes):
    """Pair Over and Under prices at the same threshold within one market."""
    prices = defaultdict(dict)
    for line, side, price in outcomes:
        try:
            line = float(line)
        except (TypeError, ValueError):
            continue
        if 0 < line <= 100 and line % 1 in (0, .5):
            prices[line][side] = 100 if price == "EVEN" else price
    for line, pair in prices.items():
        if {"over", "under"} <= pair.keys():
            probability = market.devig(pair["over"], pair["under"])
            if 0 < probability < 1:
                yield line, probability


def underdog(payload):
    appearances = {a["id"]: a for a in payload["appearances"]}
    players = {p["id"]: p for p in payload["players"]}
    games = {g["id"]: g for g in payload["games"]}
    for row in payload["over_under_lines"]:
        if row.get("line_type") != "balanced" or row.get("live_event") or row.get("status") != "active":
            continue
        prop = row.get("over_under") or {}
        appearance_stat = prop.get("appearance_stat") or {}
        stat = UNDERDOG_STATS.get(appearance_stat.get("display_stat"))
        if not stat or prop.get("category") != "player_prop":
            continue
        appearance = appearances.get(appearance_stat.get("appearance_id"), {})
        game = games.get(appearance.get("match_id"), {})
        player = players.get(appearance.get("player_id"), {})
        if game.get("sport_id") != "NBA" or game.get("status") != "scheduled":
            continue
        matchup = game.get("abbreviated_title", "").split(" @ ")
        if len(matchup) != 2:
            continue
        away, home = map(market.bbr_code, matchup)
        team = {game["away_team_id"]: away, game["home_team_id"]: home}.get(appearance.get("team_id"))
        name = " ".join(filter(None, (player.get("first_name"), player.get("last_name"))))
        if team and name:
            options = [(row.get("stat_value"), {"higher": "over", "lower": "under"}.get(o.get("choice")),
                        o.get("american_price")) for o in row.get("options", []) if o.get("status") == "active"]
            yield (away, home), team, name, stat, game.get("scheduled_at"), options


def bovada(payload, team_names):
    by_name = {name: code for code, name in team_names.items()}
    for group in payload:
        for event in group.get("events", []):
            if event.get("live"):
                continue
            teams = {t.get("home"): by_name.get(t.get("shortName")) for t in event.get("competitors", [])}
            if not teams.get(False) or not teams.get(True) or not event.get("startTime"):
                continue
            matchup = teams[False], teams[True]
            tip = datetime.fromtimestamp(event["startTime"] / 1000, timezone.utc)
            for display in event.get("displayGroups", []):
                for prop in display.get("markets", []):
                    if prop.get("status") != "O" or (prop.get("period") or {}).get("abbreviation") != "G":
                        continue
                    label, _, player = prop.get("description", "").partition(" - ")
                    match = re.fullmatch(r"(.+) \(([A-Z]+)\)", player)
                    if label not in BOVADA_STATS or not match:
                        continue
                    name, code = match.groups()
                    team = market.bbr_code(code)
                    if team not in matchup:
                        continue
                    options = []
                    for o in prop.get("outcomes", []):
                        if o.get("status") == "O" and o.get("type") in ("O", "U"):
                            price = o.get("price") or {}
                            options.append((price.get("handicap"), {"O": "over", "U": "under"}[o["type"]],
                                            price.get("american")))
                    yield matchup, team, name, BOVADA_STATS[label], ledger.fmt_utc(tip), options


def rows_for(source, props, schedule, captured, normalize):
    quotes = defaultdict(list)
    names = {}
    for matchup, team, name, stat, tip, options in props:
        prop_tip = ledger.parse_utc(tip)
        game = schedule.get((*matchup, prop_tip))
        # Exact team pairing and tip avoid joining stale or rescheduled events.
        if not game or captured >= prop_tip:
            continue
        key = (str(game.game_id), team, normalize(name), stat, game.tip_utc)
        names[key] = name
        quotes[key].extend(paired_quotes(options))
    rows = []
    for key, values in quotes.items():
        if values:
            game, team, _, stat, tip = key
            rows.append(dict(game_id=game, team=team, name=names[key], stat=stat,
                             source=source, tip_utc=tip, snapshot_utc=ledger.fmt_utc(captured),
                             mean=round(estimate(values), 4)))
    return rows


def load(path=PATH):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(rows, path):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False,
                                     dir=os.path.dirname(path) or ".") as tmp:
        try:
            writer = csv.DictWriter(tmp, FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            tmp.close()
            os.replace(tmp.name, path)
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)


def snapshot(native, today, team_names, normalize, log=print, path=PATH, fetch=market.get_json,
             clock=ledger.utc_now):
    """Keep the latest usable pregame snapshot per game and provider."""
    try:
        now = clock()
        schedule = {}
        for r in native.itertuples():
            if r.slate_date == today:
                tip = ledger.parse_utc(r.tip_utc)
                if tip and tip > now:
                    schedule[r.away, r.home, tip] = r
        if not schedule:
            return

        def capture(source, url):
            payload = fetch(url, tries=1)
            captured = clock()
            props = bovada(payload, team_names) if source == "Bovada" else underdog(payload)
            return rows_for(source, props, schedule, captured, normalize)

        new = []
        with ThreadPoolExecutor(max_workers=len(FEEDS)) as executor:
            futures = {source: executor.submit(capture, source, url) for source, url in FEEDS.items()}
            for source, future in futures.items():
                try:
                    rows = future.result()
                    new.extend(rows)
                    log(f"DFS market {source}: {len(rows)} pregame estimates")
                except Exception as exc:  # An unavailable source leaves its previous snapshot intact.
                    log(f"DFS market {source} unavailable ({exc!r})")
        if new:
            refreshed = {(r["game_id"], r["source"]) for r in new}
            old = [r for r in load(path) if (r["game_id"], r["source"]) not in refreshed]
            save(sorted(old + new, key=lambda r: (r["tip_utc"], r["game_id"], r["source"],
                                                   r["team"], r["name"], r["stat"])), path)
    except Exception as exc:
        log(f"DFS market snapshot unavailable ({exc!r})")


def index(rows, game_tips, normalize, as_of):
    """Prefer Bovada for each stat when both providers have a price."""
    players = defaultdict(dict)
    for row in sorted(rows, key=lambda r: (r["source"] == "Bovada", r["snapshot_utc"])):
        tip = game_tips.get(row["game_id"])
        if (tip and ledger.parse_utc(row["tip_utc"]) == tip
                and (captured := ledger.parse_utc(row["snapshot_utc"]))
                and captured < tip and captured <= as_of):
            key = row["game_id"], row["team"], normalize(row["name"])
            players[key][row["stat"]] = dict(mean=float(row["mean"]), source=row["source"],
                                            snapshot_utc=row["snapshot_utc"])
    return players
