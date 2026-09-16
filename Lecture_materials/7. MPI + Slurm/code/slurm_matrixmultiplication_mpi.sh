#!/bin/bash
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH --output matrixmultiplication_mpi_4.log

module load mpi/openmpi-4.1.6

mpirun -np $SLURM_NTASKS matrixmultiplication_mpi