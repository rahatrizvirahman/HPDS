#!/usr/bin/env python3
"""Averages repeated runs and computes speedup and parallel-fraction metrics
from results/maple.csv and/or results/athena.csv.

Run this yourself after run_experiments.sh (maple) and/or
slurm/collect_athena_results.py (athena) have produced their CSVs. It does
not run any experiment.

Definitions used (must match what the report states):
  absolute speedup S_abs(p) = mean(serial time)      / mean(parallel time at p)
  relative speedup S_rel(p) = mean(parallel time at 1) / mean(parallel time at p)
  Karp-Flatt serial fraction e = (1/S_abs - 1/p) / (1 - 1/p),  parallel fraction = 1 - e
  (e is left blank at p = 1, where the formula divides by zero)

Outputs:
  results/summary.csv                          one row per (platform, version,
                                                 node_policy, dataset, workers)
  results/pgfplots/<platform>_<version>[_<node_policy>]_<dataset>.dat
                                                 whitespace-separated data files
                                                 ready for \\addplot table{...}
                                                 in the report figures

Usage:
    python3 analyze_results.py
    python3 analyze_results.py --inputs results/maple.csv results/athena.csv
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(paths):
    rows = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  (skipping {p}, not found)")
            continue
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                row.setdefault("node_policy", "n/a")
                row.setdefault("nodes_used", "1")
                rows.append(row)
    return rows


def key_of(row):
    return (row["platform"], row["version"], row.get("node_policy", "n/a"),
            row["dataset"], int(row["workers"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=["results/maple.csv", "results/athena.csv"])
    ap.add_argument("--out", default="results/summary.csv")
    ap.add_argument("--pgfdir", default="results/pgfplots")
    args = ap.parse_args()

    rows = read_rows(args.inputs)
    if not rows:
        print("No input rows found. Run run_experiments.sh and/or collect_athena_results.py first.")
        return

    # Group repetitions and average both runtime and accuracy.
    groups = defaultdict(list)
    for row in rows:
        groups[key_of(row)].append(row)

    means = {}
    for k, grp in groups.items():
        times = [float(r["time_ms"]) for r in grp]
        accs = [float(r["accuracy"]) for r in grp if r.get("accuracy")]
        means[k] = {
            "mean_time_ms": statistics.mean(times),
            "n_reps": len(times),
            "accuracy": statistics.mean(accs) if accs else None,
        }

    # Serial baseline per (platform, dataset): workers column is meaningless
    # for serial, so it is always stored at workers=1 by run_experiments.sh
    # and run_serial.sh.
    serial_time = {}
    for (platform, version, _np, dataset, workers), v in means.items():
        if version == "serial":
            serial_time[(platform, dataset)] = v["mean_time_ms"]

    # Relative baseline: each version's own workers=1 run.
    own_base_time = {}
    for (platform, version, node_policy, dataset, workers), v in means.items():
        if workers == 1:
            own_base_time[(platform, version, node_policy, dataset)] = v["mean_time_ms"]

    summary_rows = []
    for k in sorted(means.keys(), key=lambda k: (k[0], k[1], k[2], k[3], k[4])):
        platform, version, node_policy, dataset, workers = k
        v = means[k]
        t = v["mean_time_ms"]

        s_abs = serial_time.get((platform, dataset))
        speedup_abs = (s_abs / t) if s_abs else ""

        s_own = own_base_time.get((platform, version, node_policy, dataset))
        speedup_rel = (s_own / t) if s_own else ""

        if isinstance(speedup_abs, float) and workers > 1:
            e = (1.0 / speedup_abs - 1.0 / workers) / (1.0 - 1.0 / workers)
            parallel_fraction = 1.0 - e
        else:
            parallel_fraction = ""

        summary_rows.append({
            "platform": platform, "version": version, "node_policy": node_policy,
            "dataset": dataset, "workers": workers, "n_reps": v["n_reps"],
            "mean_time_ms": round(t, 3),
            "accuracy": v["accuracy"],
            "speedup_abs": round(speedup_abs, 4) if isinstance(speedup_abs, float) else "",
            "speedup_rel": round(speedup_rel, 4) if isinstance(speedup_rel, float) else "",
            "parallel_fraction_karp_flatt": round(parallel_fraction, 4) if isinstance(parallel_fraction, float) else "",
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        fieldnames = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {args.out} ({len(summary_rows)} rows)")

    # One .dat file per (platform, version, node_policy, dataset) curve, for
    # \addplot table {file.dat}; in pgfplots. Skip the serial rows: they have
    # no workers axis to plot against.
    pgfdir = Path(args.pgfdir)
    pgfdir.mkdir(parents=True, exist_ok=True)
    curves = defaultdict(list)
    for r in summary_rows:
        if r["version"] == "serial":
            continue
        curves[(r["platform"], r["version"], r["node_policy"], r["dataset"])].append(r)

    for (platform, version, node_policy, dataset), pts in curves.items():
        tag = version if node_policy == "n/a" else f"{version}_{node_policy}"
        fname = pgfdir / f"{platform}_{tag}_{dataset}.dat"
        with open(fname, "w") as f:
            f.write("workers time_ms speedup_abs speedup_rel parallel_fraction\n")
            for r in sorted(pts, key=lambda r: r["workers"]):
                f.write(f"{r['workers']} {r['mean_time_ms']} "
                        f"{r['speedup_abs'] or 'nan'} {r['speedup_rel'] or 'nan'} "
                        f"{r['parallel_fraction_karp_flatt'] or 'nan'}\n")
    print(f"Wrote {len(curves)} pgfplots data files to {pgfdir}/")


if __name__ == "__main__":
    main()
