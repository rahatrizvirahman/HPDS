#!/usr/bin/env python3
"""Move the completed baseline evidence into purpose-specific directories."""
import csv
import hashlib
import io
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
DEST = RESULTS / 'maple-baseline'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if DEST.exists():
        raise SystemExit('Organized destination already exists; refusing to overwrite.')
    bundle = RESULTS / 'maple-baseline-evidence-20261022'
    archive = ROOT.parent / 'maple-baseline-evidence-20261022.tar.gz'
    # Remove a derived extraction only after verifying every file against its archive.
    if bundle.exists():
        with tarfile.open(archive) as saved:
            for path in bundle.rglob('*'):
                if path.is_file():
                    member = saved.extractfile(str(path.relative_to(RESULTS)))
                    if hashlib.sha256(member.read()).hexdigest() != digest(path):
                        raise RuntimeError(f'Extraction differs from archive, preserving {path}')
    moves = []
    for path in sorted(RESULTS.rglob('*')):
        if not path.is_file() or bundle in path.parents:
            continue
        parent = path.parent.name
        if parent.startswith('baseline-'):
            category = 'predictions' if path.name.endswith('.predictions.txt') else 'logs/measurements' if path.suffix == '.log' else 'summary' if path.name in ('summary.csv', 'kernel-summary.csv') else 'measurements'
            relative = Path(category) / path.name
        elif parent.startswith(('verify.', 'correctness-')):
            relative = Path('verification') / parent / path.name
        elif path.suffix == '.patch':
            relative = Path('environment/patches') / path.name
        elif path.name in ('assignment-text.txt', 'cluster-maple.txt', 'queue-latest.txt'):
            relative = Path('environment/diagnostics') / path.name
        else:
            relative = Path('environment') / path.name
        moves.append((path, DEST / relative, digest(path)))
    assert len({target for _, target, _ in moves}) == len(moves)
    manifest = io.StringIO()
    writer = csv.writer(manifest)
    writer.writerow(['original_path', 'current_path', 'sha256'])
    for source, target, checksum in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        assert digest(target) == checksum
        writer.writerow([source.relative_to(ROOT.parent), target.relative_to(ROOT.parent), checksum])
    # Slurm's original log location stays valid; expose copies beside measurement logs.
    for source in (ROOT / 'slurm/logs').glob('baseline-*.log'):
        target = DEST / 'logs/slurm' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert digest(target) == digest(source)
    (DEST / 'manifest.csv').write_text(manifest.getvalue())
    if bundle.exists():
        shutil.rmtree(bundle)
    # Only empty directories are removed. Unexpected files are retained.
    for path in sorted(RESULTS.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and DEST not in path.parents and path != DEST:
            try:
                path.rmdir()
            except OSError:
                pass
    rows = list(csv.DictReader((DEST / 'measurements/raw.csv').open()))
    assert len(rows) == 36
    for row in rows:
        stem = f"{row['dataset']}-{row['strategy']}-b{row['block_size']}-r{row['rep']}"
        assert (DEST / 'predictions' / (stem + '.predictions.txt')).read_bytes() == (DEST / 'predictions' / f"{row['dataset']}-serial-b0-r1.predictions.txt").read_bytes()
    (RESULTS / 'README.md').write_text('''# Results

The completed Maple baseline is in maple-baseline/.

- summary/ contains average runtimes, speedups, and kernel averages.
- measurements/ contains raw.csv and run metadata.
- predictions/ contains all 36 measured prediction files.
- logs/ contains individual measurement logs and copies of the Slurm logs.
- verification/ preserves dataset and synthetic correctness evidence.
- environment/ preserves cluster settings, accounting, patches, and diagnostics.

maple-baseline/manifest.csv maps old paths to current paths and records checksums. Historical logs and patches retain their original path text. The tracked transfer archive is unchanged. No workloads were rerun.
''')
    print(f'Moved and verified {len(moves)} evidence files; all 36 predictions still match serial.')

if __name__ == '__main__':
    main()
