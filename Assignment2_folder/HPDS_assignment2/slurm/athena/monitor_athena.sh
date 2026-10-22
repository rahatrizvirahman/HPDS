#!/usr/bin/env bash
# Monitor Athena jobs. Usage: bash slurm/athena/monitor_athena.sh JOBID [JOBID ...]
set -uo pipefail
ids=$(IFS=,; echo "$*")
squeue -u "$USER" -o '%i %P %j %T %M %R'
sacct -X -j "$ids" --format=JobID,JobName%22,Partition,NodeList,State,ExitCode,Elapsed,AllocTRES%60
