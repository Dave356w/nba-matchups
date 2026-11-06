#!/usr/bin/env python3
"""One-time ou3 rescore of the O/U locks taken before ou3 (CLAUDE.md O/U
model ou3 amendment, registered 2026-11-06T21:10:26Z).

Each native 5 h O/U lock whose tag ends `_ou1` or `_ou2` (from the
2026-10-29 slate, voided locks excluded) is rescored once with the ou3 model
at its recorded lock, from what was known then:
  * the lock's line, prices and book (lock_ou_*);
  * T* from game logs strictly before the slate (totals.Season: pace and
    points per 100 scored and allowed, form, back-to-backs, phase and both
    teams' days of rest), last season's logs as the prior;
  * the latest NBA injury report at or before the lock snapshot, box scores
    strictly before the slate, last season's OBPM / DBPM and v11's play-rate
    table (as ou_rescore.py).
Nothing after the lock is read. ou3 changes T* itself (its fit adds rest),
so unlike ou2's rescore the recorded lock_ou_total is not reused. Rows go to
data/nba_ou3_rescore.csv, written once per game (an existing row is never
changed); the ledger's first_ou_* / lock_ou_* and data/nba_ou_rescore.csv
stay as recorded. analysis.with_ou3 reads them for the ou3 rows.

  python ou3_rescore.py           # rescore any pending lock (needs an ou3 model)

The daily build calls update() after saving the ledger; it never raises, and
it does nothing until model/totals.json is an ou3 file (Fit model).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import analysis
import ledger
import nba_composite as nc
import player_availability as pav
import totals

ROOT = Path(__file__).resolve().parent
PATH = ROOT / "data" / "nba_ou3_rescore.csv"
COLUMNS = ["game_id", "slate_date", "lock_ou_snapshot_utc", "rescore_ou_tag", "rescore_ou_total",
           "rescore_ou_sigma", "rescore_ou_p_over", "rescore_ou_rest_tot",
           "rescore_ou_report_utc", "rescored_utc"]
FIRST_SLATE = "2026-10-29"       # the O/U test's first slate (ou3 rows start here)
EARLIER = ("_ou1", "_ou2")       # lock tags rescored


def load(path=PATH):
    """The rescore file (string game ids), or an empty frame."""
    try:
        d = pd.read_csv(path, dtype={"game_id": str})
    except (OSError, ValueError):
        return pd.DataFrame(columns=COLUMNS)
    return d[[c for c in COLUMNS if c in d.columns]]


def pending(native, done):
    """Native rows with an ou1 / ou2 lock, from FIRST_SLATE, not void and not
    yet rescored."""
    if native is None or not len(native) or "lock_ou_tag" not in native:
        return native.iloc[:0] if native is not None else pd.DataFrame()
    tag = native["lock_ou_tag"].astype(str)
    gid = native["game_id"].astype(str)
    keep = (tag.str.endswith(EARLIER)
            & native["lock_ou_snapshot_utc"].notna()
            & (native["slate_date"].astype(str) >= FIRST_SLATE)
            & ~gid.isin(analysis.VOID_LOCKS)
            & ~gid.isin(set(done["game_id"].astype(str))))
    return native[keep]


def is_ou3(model):
    return model is not None and model.get("report") is not None \
        and totals.model_tags(model)[0] == totals.TAG


def rescore_row(row, model, season, box, values, q_rates, archive):
    """One lock's ou3 values (dict of COLUMNS) from lock-time information.
    `season` is the slate's totals.Season (its features read games strictly
    before the slate)."""
    slate = str(row["slate_date"])
    f = season.features(row["home"], row["away"], slate)
    if f is None:
        raise ValueError("no totals inputs before the slate")
    lock = ledger.parse_utc(row["lock_ou_snapshot_utc"])
    lock_et = lock.astimezone(pav.ET).replace(tzinfo=None)
    rdt, path = archive.latest_before(lock_et)
    report = pav.parse_report(path) if path else None
    before = box[box["date"] < pd.Timestamp(slate)]
    av = pav.totals_terms(report, rdt, before, str(row["game_id"]), row["home"], row["away"],
                          slate, row["tip_utc"], values, q_rates)
    T, sigma, tag = totals.forecast(model, f, av)
    p = totals.p_over(T, sigma, float(row["lock_ou_line"]))
    return dict(game_id=str(row["game_id"]), slate_date=slate,
                lock_ou_snapshot_utc=row["lock_ou_snapshot_utc"], rescore_ou_tag=tag,
                rescore_ou_total=round(T, 2), rescore_ou_sigma=round(sigma, 3),
                rescore_ou_p_over=round(p, 5), rescore_ou_rest_tot=f["rest_tot"],
                rescore_ou_report_utc=(ledger.fmt_utc(pd.Timestamp(rdt).tz_localize(pav.ET)
                                                      .to_pydatetime())
                                       if rdt is not None and av is not None else ""),
                rescored_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))


def update(native, log=print, path=PATH, model=None, archive=None, load_logs=None):
    """Rescore every pending lock and append it to the file (write once).
    Returns the number added; never raises. Does nothing without an ou3
    model file."""
    try:
        done = load(path)
        todo = pending(native, done)
        if not len(todo):
            return 0
        model = model or totals.load()
        if not is_ou3(model):
            log("ou3 rescore: model/totals.json is not ou3 yet; skipped")
            return 0
        load_logs = load_logs or nc.load_logs
        q = json.loads((ROOT / "model" / "logit_avail.json").read_text()).get("q_rates")
        archive = archive or pav.ReportArchive(str(ROOT / "bbr_cache" / "injury_reports"))
        rows = []
        for season, g in todo.groupby("season"):
            y = int(season)
            try:
                prev = load_logs(y - 1)
            except Exception:  # noqa: BLE001 - prior = league to date
                prev = None
            s = totals.season_inputs(load_logs(y), prev)
            box = pav.fetch_box(y)
            values = pav.stat_values(box, y - 1)
            for _, row in g.iterrows():
                try:
                    rows.append(rescore_row(row, model, s, box, values, q, archive))
                except Exception as e:  # noqa: BLE001 - that game stays pending
                    log(f"ou3 rescore {row['game_id']}: {e!r}")
        archive.save()
        if rows:
            out = pd.concat([done, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
            out = out.drop_duplicates("game_id", keep="first")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(path, index=False)
        log(f"ou3 rescore: {len(rows)} lock(s) added, {len(todo) - len(rows)} pending")
        return len(rows)
    except Exception as e:  # noqa: BLE001 - never stops a build
        log(f"ou3 rescore failed ({e!r}); retry next build")
        return 0


if __name__ == "__main__":
    update(ledger.load(ledger.NATIVE_PATH))
    sys.exit(0)
