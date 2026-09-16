#!/bin/bash
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 4
#SBATCH --mem 1G
#SBATCH --output mpi_openmp_threads_4_4.log

module load mpi/openmpi-4.1.6

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpirun -np $SLURM_NTASKS mpi_openmp_threads