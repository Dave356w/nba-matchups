"""Export portable, pregame-only v6 folds and raw team logs from Actions caches.

Only writes research output; no model, ledger, or site changes. CSVs avoid
requiring consumers to load executable pickle objects.
"""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.v5_audit import common as c
import nba_composite as nc


def main():
    out = Path('research/output/v6_inputs')
    out.mkdir(parents=True, exist_ok=True)
    folds = c.build_folds([2025, 2026], with_e=True)
    years = set()
    for y, f in folds.items():
        (out/f'weights_{y}.json').write_text(json.dumps(f.weights))
        for name in ['base_train', 'avail_train', 'v4_train', 'test']:
            t = getattr(f, name)
            if t is not None:
                t.to_csv(out/f'{name}_{y}.csv', index=False)
                years.update(t.year.astype(int).unique())
    for y in sorted(years):
        for team, g in nc.load_logs(y).items():
            g.to_csv(out/f'log_{y}_{team}.csv', index=False)
    print('Portable inputs exported:', len(list(out.iterdir())), 'files', flush=True)


if __name__ == '__main__':
    main()
