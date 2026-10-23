#!/usr/bin/env python3
"""Check Maple verification artifacts and provenance in addition to the numeric audit."""
import csv
import hashlib
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/maple-shared'


def main():
    benchmarks = list(RESULTS.glob('baseline-*-job1561-*'))
    assert len(benchmarks) == 1
    benchmark = benchmarks[0]
    rows = list(csv.DictReader((benchmark / 'raw.csv').open()))
    expected = {(d, s, str(b), str(r)) for d in ('small', 'medium', 'large')
                for s, b in [('serial', 0)] + [(s, b) for s in ('global', 'shared') for b in (64, 128, 256)]
                for r in (1, 2, 3)}
    assert len(rows) == 63
    assert {(r['dataset'], r['strategy'], r['block_size'], r['rep']) for r in rows} == expected
    for row in rows:
        stem = f"{row['dataset']}-{row['strategy']}-b{row['block_size']}-r{row['rep']}"
        text = (benchmark / f'{stem}.log').read_text()
        if row['strategy'] != 'serial':
            assert f"threads_per_block={row['block_size']}" in text
        if row['strategy'] == 'shared':
            assert 'tile_rows=256' in text
        size = len((benchmark / f'{stem}.predictions.txt').read_text().splitlines())
        assert size == {'small': 160, 'medium': 740, 'large': 3436}[row['dataset']]
    verification = list(RESULTS.glob('verify.job1560.*'))
    assert len(verification) == 1
    for d in ('small', 'medium', 'large'):
        ref = (verification[0] / f'{d}-serial.txt').read_bytes()
        for s in ('cuda', 'shared'):
            for b in (64, 128, 256):
                assert (verification[0] / f'{d}-{s}-b{b}.txt').read_bytes() == ref
    synthetic = list(RESULTS.glob('correctness-job1560.*'))
    assert len(synthetic) == 1
    commands = list(synthetic[0].glob('case-*.command.txt'))
    assert len(commands) == 165
    assert len(list(synthetic[0].glob('case-*.predictions.txt'))) == 61
    for command in commands:
        text = command.read_text()
        assert re.search(r'exit=\d+\n$', text)
        for stream in ('stdout', 'stderr'):
            assert command.with_name(command.name.replace('.command.', f'.{stream}.')).exists()
    logs = ROOT / 'slurm/logs/maple-shared'
    for filename in ('verify-1560.log', 'benchmark-1561.log'):
        text = (logs / filename).read_text()
        assert 'GeForce GTX 1080 Ti' in text and 'V12.3.107' in text and '11.4.0' in text
        assert 'nvcc -I libarff -O2 -std=c++11 --fmad=false cuda.cu' in text
        assert 'g++ -I libarff -O2 -std=c++11 -ffp-contract=off serial.cpp' in text
        assert 'Ran 3 tests' in text and '\nOK\n' in text
    assert 'ALL PREDICTIONS MATCH SERIAL' in (logs / 'verify-1560.log').read_text()
    for line in (RESULTS / 'tested-source.sha256').read_text().splitlines():
        checksum, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == checksum
    accounting = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', '1560,1561', '--format=JobID,State,ExitCode'], text=True)
    states = {parts[0]: parts[1:] for parts in (line.split('|') for line in accounting.splitlines())}
    assert states == {'1560': ['COMPLETED', '0:0'], '1561': ['COMPLETED', '0:0']}
    (RESULTS / 'accounting-machine.txt').write_text(accounting)
    message = ('ADDITIONAL AUDIT PASSED\n63 expected measurements, correct block sizes and tile_rows=256.\n'
               '18 verification prediction files match serial.\n165 synthetic command records, 61 prediction files, and stdout/stderr retained.\n'
               'GPU, compiler, flags, source hashes and both requested final job records verified.\n')
    (RESULTS / 'additional-audit.txt').write_text(message)
    print(message, end='')


if __name__ == '__main__':
    main()
