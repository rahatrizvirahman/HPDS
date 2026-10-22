#!/usr/bin/env python3
"""Package the completed Maple baseline without changing original evidence."""
import csv
import hashlib
import io
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT = ROOT.parent
RESULTS = ROOT / 'results'
NAME = 'maple-baseline-evidence-20261022'
DEST = RESULTS / NAME
ARCHIVE = ASSIGNMENT / (NAME + '.tar.gz')
BENCHMARK = RESULTS / 'baseline-20261022T195803Z-ag9hh8np'


def main():
    if DEST.exists() or ARCHIVE.exists():
        raise SystemExit('Destination already exists. Preserve it or select a new name.')
    mapping = []

    def copy(source, relative):
        destination = DEST / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        destination.write_bytes(data)
        mapping.append((str(relative), str(source.relative_to(ASSIGNMENT)), hashlib.sha256(data).hexdigest()))

    for source in sorted(RESULTS.rglob('*')):
        if not source.is_file():
            continue
        if source.parent == BENCHMARK:
            category = 'predictions' if source.name.endswith('.predictions.txt') else 'logs' if source.suffix == '.log' else ''
            relative = Path('measurements') / category / source.name
        elif source.parent.name.startswith(('verify.', 'correctness-')):
            relative = Path('verification') / source.relative_to(RESULTS)
        elif source.suffix == '.patch':
            relative = Path('provenance') / source.name
        elif source.name in ('assignment-text.txt', 'cluster-maple.txt', 'queue-latest.txt'):
            relative = Path('diagnostics') / source.name
        else:
            relative = Path('environment') / source.relative_to(RESULTS)
        copy(source, relative)
    for source in sorted((ROOT / 'slurm/logs').glob('baseline-*.log')):
        copy(source, Path('slurm') / source.name)
    copy(ASSIGNMENT / 'MAPLE_BASELINE_RESULTS.md', Path('SUMMARY.md'))
    for source in sorted(ROOT.rglob('*')):
        if source.is_file() and source.suffix in ('.cpp', '.cu', '.h', '.py', '.sh', '.arff', '.md') and not any(p in ('results', '__pycache__') for p in source.relative_to(ROOT).parts):
            copy(source, Path('source') / source.relative_to(ROOT))
    copy(ROOT / 'Makefile', Path('source/Makefile'))
    manifest = io.StringIO()
    writer = csv.writer(manifest)
    writer.writerow(['bundle_path', 'original_path_relative_to_Assignment2_folder', 'sha256'])
    writer.writerows(mapping)
    (DEST / 'manifest.csv').write_text(manifest.getvalue())
    (DEST / 'README.md').write_text('''# Maple baseline evidence

Start with SUMMARY.md, then measurements/summary.csv and measurements/raw.csv.
The summary retains original Maple paths. manifest.csv maps every copied file to its original path and SHA-256 digest.

- measurements/ contains CSVs and metadata, with individual logs/ and predictions/ subdirectories.
- verification/ contains complete dataset and synthetic verification evidence.
- slurm/ contains the three build and execution logs.
- environment/ contains cluster settings, accounting, and the independent audit.
- provenance/ contains patches identifying changes after the tested revision.
- diagnostics/ preserves discovery issues and supporting inspection output.
- source/ contains source, datasets, and scripts for interpreting or reproducing the measurements.

Empty compiler temporary directories, binaries, and Python caches are omitted. Original evidence is unchanged. No measurements were rerun or altered.

All 36 measured prediction files should match the corresponding dataset's serial-b0-r1.predictions.txt file. Each dataset has three serial runs and three CUDA runs at each of 64, 128, and 256 threads per block. Headline speedup uses mean serial classification time divided by mean GPU end-to-end classification time.
''')
    for relative, original, digest in mapping:
        assert hashlib.sha256((DEST / relative).read_bytes()).hexdigest() == digest
    rows = list(csv.DictReader((DEST / 'measurements/raw.csv').open()))
    assert len(rows) == 36
    for row in rows:
        name = f"{row['dataset']}-{row['strategy']}-b{row['block_size']}-r{row['rep']}.predictions.txt"
        predictions = DEST / 'measurements/predictions'
        assert (predictions / name).read_bytes() == (predictions / f"{row['dataset']}-serial-b0-r1.predictions.txt").read_bytes()
    with tarfile.open(ARCHIVE, 'w:gz') as archive:
        archive.add(DEST, arcname=NAME)
    with tarfile.open(ARCHIVE) as archive:
        for relative, original, digest in mapping:
            assert hashlib.sha256(archive.extractfile(f'{NAME}/{relative}').read()).hexdigest() == digest
    checksum = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    ARCHIVE.with_name(ARCHIVE.name + '.sha256').write_text(f'{checksum}  {ARCHIVE.name}\n')
    print(f'Verified {len(mapping)} copied files and 36 predictions. Archive {ARCHIVE}, {ARCHIVE.stat().st_size} bytes')


if __name__ == '__main__':
    main()
