"""Retired historical Kalshi backfill, research only; no production workflow."""
import pandas as pd
import kalshi
import ledger

def backfill_kalshi(seasons, spacing=0.2, every=200):
    """kalshi_* grade columns for graded reconstructed rows of `seasons` (from
    ledger.KALSHI_FROM_SEASON) that lack a Kalshi close. Saves every `every`
    rows, so a stopped run keeps its progress; reruns skip filled rows."""
    kalshi.SPACING = spacing
    df = ledger.load(ledger.RECON_PATH)
    won = pd.to_numeric(df["home_won"], errors="coerce").isin([0, 1])
    season = pd.to_numeric(df["season"], errors="coerce")
    todo = df.index[won & season.isin(seasons) & (season >= ledger.KALSHI_FROM_SEASON)
                    & pd.to_numeric(df["kalshi_close_q_home"], errors="coerce").isna()]
    print(f"Kalshi: {len(todo)} reconstructed rows to fill", flush=True)
    found = 0
    for k, i in enumerate(todo, 1):
        r = df.loc[i]
        try:
            o = kalshi.grade_odds(str(r["slate_date"]), r["home"], r["away"], r["tip_utc"])
        except Exception as e:  # noqa: BLE001 - one game; rerun fills it
            print(f"{r['game_id']}: {e!r}", flush=True)
            continue
        for c in ledger.KALSHI_GRADE_COLUMNS:
            if o.get(c) is not None:
                df.at[i, c] = o[c]
        found += bool(o)
        if k % every == 0:
            ledger.save(df, ledger.RECON_PATH)
            print(f"Kalshi: {k}/{len(todo)} read, {found} with a close", flush=True)
    ledger.save(df, ledger.RECON_PATH)
    print(f"Kalshi: {found}/{len(todo)} rows filled; wrote {ledger.RECON_PATH}",
          flush=True)
    return 0


