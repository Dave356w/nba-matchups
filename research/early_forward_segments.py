"""Print the registered early segments on v9 ESPN BET rows: hindsight only."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis
import ledger
import report


def main():
    g = ledger.graded(ledger.load(ROOT / "data/nba_reconstructed.csv"))
    early = analysis.on_book(g, "close", "espnbet")
    early = early[analysis.stage(early)[1]]
    lines = ["EARLY SEGMENTS — HINDSIGHT ONLY, NEVER FORWARD EVIDENCE",
             f"Source: {len(early)} ESPN BET early games; primary v9 P.",
             "Frozen thresholds. Injury-timing caveat applies. Games 10+ never pooled.",
             "Intervals below are ± 1 SE, as on the Ledger; not 95% intervals."]
    for h, r in analysis.early_hindsight(g):
        lines.append(h["key"] + " " + report.roi_line("@ open", r, 12))
    text = "\n".join(lines) + "\n"
    out = ROOT / "diagnostics/first_snapshot_registration/early_hindsight.txt"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
