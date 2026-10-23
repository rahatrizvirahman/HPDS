#!/usr/bin/env python3
"""Package the Athena H100 single-GPU, MPI+CUDA verification and CPU evidence without changing the originals.

Usage: package_athena_extra_results.py h100|mpi|cpu [TAG]. TAG (default 20261023) is part of the archive name, so a later
package of new evidence (for example a completed scaling job) never overwrites an earlier archive.
"""
import csv
import hashlib
import io
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT = ROOT.parent
SKIP_PREFIXES = ('tmp-', 'work-')
MILESTONES = {
    'h100': dict(name='athena-h100-single-gpu-evidence', summary='ATHENA_H100_SINGLE_GPU_RESULTS.md', title='Athena H100 single-GPU evidence',
                 results=['athena-shared-h100'], logs=['athena-shared-h100'], bench='baseline-*-job*', reference='{dataset}-serial-b0-r1.predictions.txt',
                 stem='{dataset}-{strategy}-b{block_size}-r{rep}', rows=63),
    'mpi': dict(name='athena-mpi-cuda-evidence', summary='ATHENA_MPI_CUDA_RESULTS.md', title='Athena MPI+CUDA verification evidence',
                results=['athena-mpi-h100', 'athena-mpi-v100', 'athena-mpi'], logs=['athena-mpi-h100', 'athena-mpi-v100'],
                bench='mpi-benchmark-*', reference='{dataset}-serial---r1.predictions.txt', stem='{dataset}-{program}-{layout}-r{rep}', rows=None),
    'cpu': dict(name='athena-cpu-evidence', summary='ATHENA_CPU_RESULTS.md', title='Athena CPU comparison evidence',
                results=['athena-cpu'], logs=['athena-cpu'], bench='cpu-benchmark-*', reference='{dataset}-serial-w1-r1.predictions.txt',
                stem='{dataset}-{program}-w{workers}-r{rep}', rows=144),
}


def main():
    config = MILESTONES[sys.argv[1]]
    tag = sys.argv[2] if len(sys.argv) > 2 else '20261023'
    name = f"{config['name']}-{tag}"
    archive_path = ASSIGNMENT / (name + '.tar.gz')
    scratch = ROOT / 'results' / '_packaging' / name
    if archive_path.exists() or scratch.exists():
        raise SystemExit('Destination already exists. Preserve it or select a new tag.')
    mapping = []

    def copy(source, relative):
        destination = scratch / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        destination.write_bytes(data)
        mapping.append((str(relative), str(source.relative_to(ASSIGNMENT)), hashlib.sha256(data).hexdigest()))

    benchmarks = []
    for label in config['results']:
        base = ROOT / 'results' / label
        found = sorted(base.glob(config['bench']))
        benchmarks += found
        for source in sorted(base.rglob('*')):
            parts = source.relative_to(base).parts
            if not source.is_file() or any(p.startswith(SKIP_PREFIXES) for p in parts):
                continue
            if source.parent in found:
                category = 'predictions' if source.name.endswith('.predictions.txt') else 'logs' if source.suffix == '.log' else ''
                relative = Path('measurements') / label / source.parent.name / category / source.name
            elif source.parent.name.startswith(('verify.', 'correctness-', 'mpi-verify.', 'mpi-correctness-', 'cpu-verify.', 'cpu-correctness-')):
                relative = Path('verification') / label / source.relative_to(base)
            elif source.suffix == '.patch' or source.name.endswith(('.status.txt', '.scontrol.txt')):
                relative = Path('provenance') / label / source.name
            else:
                relative = Path('environment') / label / source.relative_to(base)
            copy(source, relative)
    for label in config['logs']:
        for source in sorted((ROOT / 'slurm/logs' / label).glob('*.log')):
            copy(source, Path('slurm') / label / source.name)
    copy(ASSIGNMENT / config['summary'], Path('SUMMARY.md'))
    for source in sorted(ROOT.rglob('*')):
        parts = source.relative_to(ROOT).parts
        if source.is_file() and source.suffix in ('.cpp', '.cu', '.cuh', '.h', '.py', '.sh', '.arff', '.md') and not any(p in ('results', '__pycache__', 'logs') for p in parts):
            copy(source, Path('source') / source.relative_to(ROOT))
    copy(ROOT / 'Makefile', Path('source/Makefile'))
    manifest = io.StringIO()
    writer = csv.writer(manifest)
    writer.writerow(['bundle_path', 'original_path_relative_to_Assignment2_folder', 'sha256'])
    writer.writerows(mapping)
    (scratch / 'manifest.csv').write_text(manifest.getvalue())
    (scratch / 'README.md').write_text(f'''# {config['title']}

Start with SUMMARY.md. manifest.csv maps every copied file to its original path and SHA-256 digest.

- measurements/ holds benchmark CSVs, metadata, individual logs and prediction files, if the evidence includes a benchmark.
- verification/ holds dataset and synthetic verification output.
- slurm/ holds the Slurm logs, including failed and diagnostic jobs.
- environment/ holds discovery output, submissions, audits and other supporting files.
- provenance/ holds per-job git status, patches and job descriptions.
- source/ holds the tested sources, scripts, tools and datasets.

Empty compiler temporary directories, private build work directories, binaries and Python caches are omitted. Originals are unchanged and nothing was rerun or altered.
''')
    for relative, original, digest in mapping:
        assert hashlib.sha256((scratch / relative).read_bytes()).hexdigest() == digest
    checked = 0
    for directory in benchmarks:
        rows = list(csv.DictReader((directory / 'raw.csv').open()))
        if config['rows']:
            assert len(rows) == config['rows']
        target = scratch / 'measurements' / directory.parent.name / directory.name / 'predictions'
        for row in rows:
            ref = target / config['reference'].format(**row)
            assert (target / (config['stem'].format(**row) + '.predictions.txt')).read_bytes() == ref.read_bytes()
            checked += 1
    with tarfile.open(archive_path, 'w:gz') as archive:
        archive.add(scratch, arcname=name)
    with tarfile.open(archive_path) as archive:
        for relative, original, digest in mapping:
            assert hashlib.sha256(archive.extractfile(f'{name}/{relative}').read()).hexdigest() == digest
        assert archive.extractfile(f'{name}/manifest.csv').read().decode() == manifest.getvalue()
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_path.with_name(archive_path.name + '.sha256').write_text(f'{checksum}  {archive_path.name}\n')
    print(f'Verified {len(mapping)} copied files and {checked} measured predictions. Archive {archive_path}, {archive_path.stat().st_size} bytes')


if __name__ == '__main__':
    main()
