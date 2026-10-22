#!/usr/bin/env python3
"""Package the Athena baseline evidence without changing the original files."""
import csv
import hashlib
import io
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT = ROOT.parent
# Usage: package_athena_results.py [baseline|shared]. The default packages the original baseline evidence.
MILESTONES = {
    'baseline': dict(results='athena', logs='athena', logglob='baseline-*.log', name='athena-baseline-evidence-20261022',
                     summary='ATHENA_BASELINE_RESULTS.md', title='Athena baseline evidence',
                     jobs='jobs 3382502, 3382504 and 3382505', runs=36, strategies=1),
    'shared': dict(results='athena-shared', logs='athena-shared', logglob='job-*.log', name='athena-shared-memory-evidence-20261022',
                   summary='ATHENA_SHARED_MEMORY_RESULTS.md', title='Athena shared-memory evidence',
                   jobs='jobs 3382580, 3382581 and 3382591', runs=63, strategies=2),
}
CONFIG = MILESTONES[sys.argv[1] if len(sys.argv) > 1 else 'baseline']
RESULTS = ROOT / 'results' / CONFIG['results']
NAME = CONFIG['name']
DEST = RESULTS / NAME
ARCHIVE = ASSIGNMENT / (NAME + '.tar.gz')
BENCHMARK = sorted(RESULTS.glob('baseline-*-job*'))


def main():
    if DEST.exists() or ARCHIVE.exists():
        raise SystemExit('Destination already exists. Preserve it or select a new name.')
    if len(BENCHMARK) != 1:
        raise SystemExit('Expected exactly one Athena benchmark directory')
    benchmark = BENCHMARK[0]
    mapping = []

    def copy(source, relative):
        destination = DEST / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        destination.write_bytes(data)
        mapping.append((str(relative), str(source.relative_to(ASSIGNMENT)), hashlib.sha256(data).hexdigest()))

    for source in sorted(RESULTS.rglob('*')):
        if not source.is_file() or NAME in source.parts or any(p.startswith('tmp-') for p in source.relative_to(RESULTS).parts):
            continue
        if source.parent == benchmark:
            category = 'predictions' if source.name.endswith('.predictions.txt') else 'logs' if source.suffix == '.log' else ''
            relative = Path('measurements') / category / source.name
        elif source.parent.name.startswith(('verify.', 'correctness-')):
            relative = Path('verification') / source.relative_to(RESULTS)
        elif source.suffix == '.patch' or source.name.endswith('.status.txt'):
            relative = Path('provenance') / source.name
        else:
            relative = Path('environment') / source.relative_to(RESULTS)
        copy(source, relative)
    for source in sorted((ROOT / 'slurm/logs' / CONFIG['logs']).glob(CONFIG['logglob'])):
        copy(source, Path('slurm') / source.name)
    copy(ASSIGNMENT / CONFIG['summary'], Path('SUMMARY.md'))
    for source in sorted(ROOT.rglob('*')):
        parts = source.relative_to(ROOT).parts
        if source.is_file() and source.suffix in ('.cpp', '.cu', '.h', '.py', '.sh', '.arff', '.md') and not any(p in ('results', '__pycache__', 'logs') for p in parts):
            copy(source, Path('source') / source.relative_to(ROOT))
    copy(ROOT / 'Makefile', Path('source/Makefile'))
    manifest = io.StringIO()
    writer = csv.writer(manifest)
    writer.writerow(['bundle_path', 'original_path_relative_to_Assignment2_folder', 'sha256'])
    writer.writerows(mapping)
    (DEST / 'manifest.csv').write_text(manifest.getvalue())
    (DEST / 'README.md').write_text(f'''# {CONFIG['title']}

Start with SUMMARY.md, then measurements/summary.csv, kernel-summary.csv and raw.csv.
The summary retains original Athena paths. manifest.csv maps every copied file to its original path and SHA-256 digest.

- measurements/ contains CSVs and metadata, with individual logs/ and predictions/ subdirectories.
- verification/ contains complete dataset and synthetic verification evidence.
- slurm/ contains the build and execution logs of {CONFIG['jobs']}.
- environment/ contains discovery output, discovery job logs, Slurm settings, accounting, submissions and the independent audit.
- provenance/ contains per-job git status and patches, and the combined change patch relative to the tested revision.
- source/ contains the tested source, scripts, tools, datasets and the Maple/Athena documentation snapshot.

Empty compiler temporary directories, binaries and Python caches are omitted. Original evidence is unchanged and nothing was rerun or altered.

All {CONFIG['runs']} measured prediction files should match the corresponding dataset's serial-b0-r1.predictions.txt file. Each dataset has three serial runs and three CUDA runs per strategy at each of 64, 128 and 256 threads per block ({CONFIG['strategies']} CUDA strateg{'y' if CONFIG['strategies'] == 1 else 'ies'}), all on one NVIDIA A100-SXM4-80GB. Headline speedup uses mean Athena serial classification time divided by mean Athena GPU end-to-end classification time.
''')
    for relative, original, digest in mapping:
        assert hashlib.sha256((DEST / relative).read_bytes()).hexdigest() == digest
    rows = list(csv.DictReader((DEST / 'measurements/raw.csv').open()))
    assert len(rows) == CONFIG['runs']
    predictions = DEST / 'measurements/predictions'
    for row in rows:
        name = f"{row['dataset']}-{row['strategy']}-b{row['block_size']}-r{row['rep']}.predictions.txt"
        assert (predictions / name).read_bytes() == (predictions / f"{row['dataset']}-serial-b0-r1.predictions.txt").read_bytes()
    with tarfile.open(ARCHIVE, 'w:gz') as archive:
        archive.add(DEST, arcname=NAME)
    with tarfile.open(ARCHIVE) as archive:
        for relative, original, digest in mapping:
            assert hashlib.sha256(archive.extractfile(f'{NAME}/{relative}').read()).hexdigest() == digest
        archive_manifest = archive.extractfile(f'{NAME}/manifest.csv').read().decode()
        assert archive_manifest == manifest.getvalue()
    checksum = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    ARCHIVE.with_name(ARCHIVE.name + '.sha256').write_text(f'{checksum}  {ARCHIVE.name}\n')
    print(f'Verified {len(mapping)} copied files and {CONFIG["runs"]} predictions. Archive {ARCHIVE}, {ARCHIVE.stat().st_size} bytes')


if __name__ == '__main__':
    main()
