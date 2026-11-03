#!/usr/bin/env python3
"""One-time ou2 rescore of the O/U locks taken before ou2 (CLAUDE.md O/U
model ou2 amendment, registered 2026-11-03T23:05:00Z).

Each native 5 h O/U lock whose tag ends `_ou1` (from the 2026-10-29 slate)
is rescored once with the ou2 model at its recorded lock, from what was
known then:
  * the lock's base total (lock_ou_total: T*, whose maths ou2 keeps), line,
    prices and book;
  * the latest NBA injury report at or before the lock snapshot;
  * box scores strictly before the slate, last season's OBPM / DBPM;
  * v11's play-rate table (model/logit_avail.json q_rates).
Nothing after the lock is read. Rows go to data/nba_ou_rescore.csv, written
once per game (an existing row is never changed); the ledger's first_ou_* /
lock_ou_* stay as recorded. analysis.with_ou2 reads them for the ou2 rows.

  python ou_rescore.py            # rescore any ou1 lock not yet in the file

The daily build calls update() after saving the ledger; it never raises.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import ledger
import player_availability as pav
import totals

ROOT = Path(__file__).resolve().parent
PATH = ROOT / "data" / "nba_ou_rescore.csv"
COLUMNS = ["game_id", "slate_date", "lock_ou_snapshot_utc", "rescore_ou_tag", "rescore_ou_total",
           "rescore_ou_sigma", "rescore_ou_p_over", "rescore_ou_report_utc", "rescored_utc"]
FIRST_SLATE = "2026-10-29"       # the O/U test's first slate (ou2 rows start here)


def load(path=PATH):
    """The rescore file (string game ids), or an empty frame."""
    try:
        d = pd.read_csv(path, dtype={"game_id": str})
    except (OSError, ValueError):
        return pd.DataFrame(columns=COLUMNS)
    return d[[c for c in COLUMNS if c in d.columns]]


def pending(native, done):
    """Native rows with an ou1 lock, from FIRST_SLATE, not yet rescored."""
    if native is None or not len(native) or "lock_ou_tag" not in native:
        return native.iloc[:0] if native is not None else pd.DataFrame()
    tag = native["lock_ou_tag"].astype(str)
    keep = (tag.str.endswith("_ou1")
            & native["lock_ou_snapshot_utc"].notna()
            & (native["slate_date"].astype(str) >= FIRST_SLATE)
            & ~native["game_id"].astype(str).isin(set(done["game_id"].astype(str))))
    return native[keep]


def rescore_row(row, model, box, values, q_rates, archive):
    """One lock's ou2 values (dict of COLUMNS) from lock-time information."""
    lock = ledger.parse_utc(row["lock_ou_snapshot_utc"])
    lock_et = lock.astimezone(pav.ET).replace(tzinfo=None)
    rdt, path = archive.latest_before(lock_et)
    report = pav.parse_report(path) if path else None
    slate = str(row["slate_date"])
    before = box[box["date"] < pd.Timestamp(slate)]
    av = pav.totals_terms(report, rdt, before, str(row["game_id"]), row["home"], row["away"],
                          slate, row["tip_utc"], values, q_rates)
    gp = min(float(row["gp_home"]), float(row["gp_away"]))
    T, sigma, tag = totals.adjust(model, float(row["lock_ou_total"]), gp, av)
    p = totals.p_over(T, sigma, float(row["lock_ou_line"]))
    return dict(game_id=str(row["game_id"]), slate_date=slate,
                lock_ou_snapshot_utc=row["lock_ou_snapshot_utc"], rescore_ou_tag=tag,
                rescore_ou_total=round(T, 2), rescore_ou_sigma=round(sigma, 3),
                rescore_ou_p_over=round(p, 5),
                rescore_ou_report_utc=(ledger.fmt_utc(pd.Timestamp(rdt).tz_localize(pav.ET)
                                                      .to_pydatetime())
                                       if rdt is not None and av is not None else ""),
                rescored_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))


def update(native, log=print, path=PATH, model=None, archive=None):
    """Rescore every pending ou1 lock and append it to the file (write once).
    Returns the number added; never raises."""
    try:
        done = load(path)
        todo = pending(native, done)
        if not len(todo):
            return 0
        model = model or totals.load()
        if model is None or not model.get("report"):
            log("ou2 rescore: no ou2 model with a report stage; skipped")
            return 0
        q = json.loads((ROOT / "model" / "logit_avail.json").read_text()).get("q_rates")
        # the live build's report archive (LiveAvailability); missing editions
        # are fetched from the NBA's CDN
        archive = archive or pav.ReportArchive(str(ROOT / "bbr_cache" / "injury_reports"))
        rows = []
        for season, g in todo.groupby("season"):
            box = pav.fetch_box(int(season))
            values = pav.stat_values(box, int(season) - 1)
            for _, row in g.iterrows():
                try:
                    rows.append(rescore_row(row, model, box, values, q, archive))
                except Exception as e:  # noqa: BLE001 - that game stays pending
                    log(f"ou2 rescore {row['game_id']}: {e!r}")
        archive.save()
        if rows:
            out = pd.concat([done, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
            out = out.drop_duplicates("game_id", keep="first")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(path, index=False)
        log(f"ou2 rescore: {len(rows)} lock(s) added, {len(todo) - len(rows)} pending")
        return len(rows)
    except Exception as e:  # noqa: BLE001 - never stops a build
        log(f"ou2 rescore failed ({e!r}); retry next build")
        return 0


if __name__ == "__main__":
    n = update(ledger.load(ledger.NATIVE_PATH))
    sys.exit(0)
