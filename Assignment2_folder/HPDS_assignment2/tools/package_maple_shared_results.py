#!/usr/bin/env python3
"""Package audited Maple shared-memory evidence without duplicating the working tree."""
import csv
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT = ROOT.parent
RESULTS = ROOT / 'results/maple-shared'


def main():
    name = 'maple-shared-memory-evidence-' + datetime.now(timezone.utc).strftime('%Y%m%d')
    archive_path = ASSIGNMENT / (name + '.tar.gz')
    if archive_path.exists():
        raise SystemExit('Archive exists; refusing to overwrite evidence.')
    benchmarks = list(RESULTS.glob('baseline-*-job*'))
    if len(benchmarks) != 1:
        raise SystemExit('Expected exactly one completed Maple paired benchmark.')
    benchmark = benchmarks[0]
    if 'AUDIT PASSED' not in (RESULTS / 'audit.txt').read_text():
        raise SystemExit('Run the completed benchmark audit before packaging.')
    rows = list(csv.DictReader((benchmark / 'raw.csv').open()))
    assert len(rows) == 63
    for row in rows:
        stem = f"{row['dataset']}-{row['strategy']}-b{row['block_size']}-r{row['rep']}"
        assert (benchmark / (stem + '.predictions.txt')).read_bytes() == (benchmark / f"{row['dataset']}-serial-b0-r1.predictions.txt").read_bytes()
    files = {}

    def include(source, target):
        if target in files:
            raise RuntimeError(f'Duplicate archived path {target}')
        files[target] = source

    for source in sorted(RESULTS.rglob('*')):
        if not source.is_file() or any(p.startswith('tmp-') for p in source.relative_to(RESULTS).parts):
            continue
        if source.parent == benchmark:
            category = 'predictions' if source.name.endswith('.predictions.txt') else 'logs/measurements' if source.suffix == '.log' else 'summary' if source.name in ('summary.csv', 'kernel-summary.csv') else 'measurements'
            target = f'{category}/{source.name}'
        elif source.parent.name.startswith(('verify.', 'correctness-')):
            target = f'verification/{source.relative_to(RESULTS)}'
        else:
            category = 'provenance' if source.suffix == '.patch' or 'revision' in source.name or 'status' in source.name else 'environment'
            target = f'{category}/{source.relative_to(RESULTS)}'
        include(source, target)
    for source in sorted((ROOT / 'slurm/logs/maple-shared').glob('*.log')):
        include(source, f'logs/slurm/{source.name}')
    include(ASSIGNMENT / 'MAPLE_SHARED_MEMORY_RESULTS.md', 'SUMMARY.md')
    for source in sorted(ROOT.rglob('*')):
        parts = source.relative_to(ROOT).parts
        if source.is_file() and not any(p in ('results', 'logs', '__pycache__') for p in parts) and (source.suffix in ('.cpp', '.cu', '.cuh', '.h', '.py', '.sh', '.arff', '.md') or source.name == 'Makefile'):
            include(source, f'source/{source.relative_to(ROOT)}')
    for source in sorted((ASSIGNMENT / 'Explanations').glob('*.md')):
        include(source, f'explanations/{source.name}')
    payloads = {target: source.read_bytes() for target, source in files.items()}
    manifest = io.StringIO()
    writer = csv.writer(manifest)
    writer.writerow(['bundle_path', 'original_path_relative_to_Assignment2_folder', 'sha256'])
    for target, source in files.items():
        writer.writerow([target, source.relative_to(ASSIGNMENT), hashlib.sha256(payloads[target]).hexdigest()])
    payloads['manifest.csv'] = manifest.getvalue().encode()
    payloads['README.md'] = b'''# Maple shared-memory evidence

Start with SUMMARY.md and summary/summary.csv. measurements/raw.csv contains all 63 measurements. predictions/ and logs/measurements/ contain individual evidence. verification/ preserves synthetic and full-dataset checks. logs/slurm/, environment/, and provenance/ contain build settings, accounting, commands, revision, and patches. source/ includes the CUDA header and datasets. manifest.csv maps each copied file to its original path and SHA-256 hash. Historical paths printed in logs are unchanged.

Three repetitions were measured per dataset/strategy/block group. Headline speedup uses serial and GPU end-to-end means from this sweep. No old measurements were substituted. Empty temporary directories, binaries, and Python caches are omitted.
'''
    with tarfile.open(archive_path, 'x:gz') as archive:
        for target, data in payloads.items():
            member = tarfile.TarInfo(f'{name}/{target}')
            member.size = len(data)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(data))
    with tarfile.open(archive_path) as archive:
        assert len(archive.getmembers()) == len(payloads)
        for target, data in payloads.items():
            assert archive.extractfile(f'{name}/{target}').read() == data
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_path.with_name(archive_path.name + '.sha256').write_text(f'{checksum}  {archive_path.name}\n')
    print(f'Archive and internal manifest verified, {len(files)} evidence/source files, 63 predictions, {archive_path.stat().st_size} bytes')
    print(archive_path)


if __name__ == '__main__':
    main()
