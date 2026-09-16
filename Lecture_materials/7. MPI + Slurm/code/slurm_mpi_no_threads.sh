#!/bin/bash
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 1
#SBATCH --mem 1G
#SBATCH --output mpi_no_threads_4.log

module load mpi/openmpi-4.1.6

mpirun -np $SLURM_NTASKS mpi_no_threads