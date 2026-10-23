"""Inventory missing-input fallback cases before replacing their scoring route.

No forced missing values or ordinary v9 predictions count as fallback evidence.
The committed seasons currently contain zero such cases. If any are found,
their feature reconstruction and paired rescoring are required before activation.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import ledger


def inventory(paths):
    lines = ["V9 NEUTRAL-INPUT CHECK — BEFORE IMPLEMENTATION", "Same-game fallback inventory; hindsight only."]
    count = 0
    for path in paths:
        g = ledger.graded(ledger.load(path))
        scored = pd.to_numeric(g.p_home, errors="coerce").notna()
        fallback = scored & ~g.model_tag.astype(str).str.contains("margin.*v9", regex=True)
        n = int(fallback.sum())
        count += n
        lines.append(f"{Path(path).name}: {int(scored.sum())} scored, {n} non-v9 fallback cases")
        for _, row in g[fallback].iterrows():
            lines.append(f"  {row.game_id} {row.slate_date} {row.away}@{row.home}: {row.model_tag}")
    if count:
        lines.append("BLOCKED: reconstruct inputs and compare neutral v9 with each same-game fallback before activation.")
    else:
        lines.append("No missing-input cases in these rebuilt seasons: paired comparison n=0.")
        lines.append("No empirical improvement can be claimed. Controlled missing-input regressions are required.")
    return "\n".join(lines) + "\n", count


def main():
    text, n = inventory([ROOT / "data/nba_reconstructed.csv", ROOT / "data/nba_rebuilt.csv"])
    out = ROOT / "diagnostics/neutral_fill"
    out.mkdir(parents=True, exist_ok=True)
    (out / "BEFORE.txt").write_text(text)
    print(text)
    return bool(n)


if __name__ == "__main__":
    raise SystemExit(main())
