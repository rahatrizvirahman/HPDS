#!/usr/bin/env bash
#SBATCH --job-name=athena-mpi-diagnose
#SBATCH --time=00:10:00
#SBATCH --mem=4G
# Diagnose a two-node MPI step that hung. Each attempt has a 60 s timeout and its stderr is kept.
# Run from HPDS_assignment2 with an already built ./mpi_cuda and the same modules as the failing job.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda/12.3 mpi/openmpi-4.1.6
echo "job $SLURM_JOB_ID nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
args=(datasets/small-train.arff datasets/small-test.arff 3 128)
attempt() {
    local label=$1; shift
    echo "=== $label"
    echo "+ $*"
    local start=$SECONDS
    timeout 60 "$@" 2>&1 | head -30
    echo "--- exit ${PIPESTATUS[0]} after $((SECONDS - start)) s"
}
# ATTEMPTS selects which attempts run (default a b c d e f). Attempts g and h use mpirun inside the allocation.
want() { [[ " ${ATTEMPTS:-a b c d e f} " == *" $1 "* ]]; }
want a && attempt 'a. plain srun hostname' srun --nodes=2 --ntasks-per-node=1 hostname
want b && attempt 'b. pmix, no gpu binding' srun --mpi=pmix --nodes=2 --ntasks-per-node=1 ./mpi_cuda "${args[@]}"
want c && attempt 'c. pmix, gpus-per-task' srun --mpi=pmix --nodes=2 --ntasks-per-node=1 --gpus-per-task=1 ./mpi_cuda "${args[@]}"
want d && attempt 'd. pmi2' srun --mpi=pmi2 --nodes=2 --ntasks-per-node=1 ./mpi_cuda "${args[@]}"
want e && attempt 'e. pmix, UCX off (ob1 over tcp)' env OMPI_MCA_pml=ob1 OMPI_MCA_btl=tcp,self srun --mpi=pmix --nodes=2 --ntasks-per-node=1 ./mpi_cuda "${args[@]}"
want f && attempt 'f. pmix, UCX transports limited to tcp' env UCX_TLS=tcp,self,sm srun --mpi=pmix --nodes=2 --ntasks-per-node=1 ./mpi_cuda "${args[@]}"
want g && attempt 'g. mpirun, one rank per node' mpirun -n 2 --map-by ppr:1:node ./mpi_cuda "${args[@]}"
want h && attempt 'h. mpirun, two ranks per node (4 ranks), needs --ntasks-per-node=2' mpirun -n 4 --map-by ppr:2:node ./mpi_cuda "${args[@]}"
want i && { echo '=== i. IPv4 addresses per node'; srun --nodes=2 --ntasks-per-node=1 bash -c 'echo "--- $(hostname)"; ip -4 -o addr show | awk "{print \$2, \$4}"'; }
want j && attempt 'j. mpirun excluding lo and virbr0' mpirun -n 2 --map-by ppr:1:node --mca oob_tcp_if_exclude lo,virbr0 --mca btl_tcp_if_exclude lo,virbr0 ./mpi_cuda "${args[@]}"
want k && attempt 'k. mpirun using ibp103s0 only' mpirun -n 2 --map-by ppr:1:node --mca oob_tcp_if_include ibp103s0 --mca btl_tcp_if_include ibp103s0 ./mpi_cuda "${args[@]}"
want l && attempt 'l. mpirun using ibp103s0 only, UCX over that device' env UCX_NET_DEVICES=mlx5_0:1 mpirun -n 2 --map-by ppr:1:node --mca oob_tcp_if_include ibp103s0 --mca btl_tcp_if_include ibp103s0 ./mpi_cuda "${args[@]}"
want m && { echo '=== m. verbose mpirun launch'; timeout 60 mpirun -n 2 --map-by ppr:1:node --debug-daemons --mca plm_base_verbose 10 ./mpi_cuda "${args[@]}" 2>&1 | head -60; }
want n && { echo '=== n. one node mpirun, two ranks (control)'; timeout 60 mpirun -n 1 ./mpi_cuda "${args[@]}" 2>&1 | head -10; }
want o && { echo '=== o. srun can start orted on remote node?'; srun --nodes=2 --ntasks-per-node=1 bash -c 'echo "$(hostname): $(command -v orted) LD=$LD_LIBRARY_PATH"; orted --version 2>&1 | head -2'; }
echo '=== network interfaces on each node'
srun --nodes=2 --ntasks-per-node=1 bash -c 'echo "$(hostname): $(ls /sys/class/net | tr "\n" " ")"; ibstat 2>/dev/null | grep -E "State|Rate" | head -4'
