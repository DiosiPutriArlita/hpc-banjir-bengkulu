#!/bin/bash
#SBATCH --job-name=floodcast
#SBATCH --account=PROJECT_ID
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load Python/3.11 OpenMPI

source .venv/bin/activate
mkdir -p logs results
export PYTHONPATH=$PWD/src

srun python -m floodcast.sweep --preset full --backend mpi --out results/sweep_full.csv
