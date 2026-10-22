#!/usr/bin/env bash
# Read-only discovery, safe to run on a login node. No GPU work is launched.
set -uo pipefail
printf 'Host and date\n'
hostname
date -u
printf '\nSlurm partitions and GPU resources\n'
if command -v sinfo >/dev/null 2>&1; then
    sinfo -o '%P %a %l %D %G'
    sinfo -N -o '%N %P %G %c %m'
else
    echo 'sinfo is unavailable in this shell'
fi
printf '\nAvailable modules\n'
if type module >/dev/null 2>&1; then
    module list 2>&1
    module avail cuda 2>&1
    module avail gcc 2>&1
    module avail mpi 2>&1
else
    echo 'The module command is unavailable in this shell'
fi
printf '\nCompilers currently on PATH\n'
command -v nvcc g++ mpicxx || true
if command -v nvcc >/dev/null 2>&1; then nvcc --version; fi
if command -v g++ >/dev/null 2>&1; then g++ --version; fi
printf '\nAccount and QOS associations, if readable\n'
if command -v sacctmgr >/dev/null 2>&1; then
    sacctmgr -n show assoc where user="$USER" format=Cluster,Account,Partition,QOS,DefaultQOS,MaxJobs,MaxSubmitJobs
    sacctmgr show qos format=Name,MaxTRESPerJob,MaxWall
fi
