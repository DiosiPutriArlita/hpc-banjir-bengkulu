#!/bin/bash
#SBATCH --job-name=floodcast
#SBATCH --account=PROJECT_ID          # GANTI: ID proyek/alokasi dari pusat HPC Anda
#SBATCH --partition=compute           # GANTI: nama partisi (cek `sinfo`)
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Nama modul berbeda di tiap pusat HPC -> cek `module avail`
module purge
module load Python/3.11 OpenMPI       # GANTI sesuai sistem

source .venv/bin/activate
mkdir -p logs results
export PYTHONPATH=$PWD/src

# 1 rank MPI = 1 proses Python; kejadian dibagi round-robin antar rank
srun python -m floodcast.sweep --preset full --backend mpi --out results/sweep_full.csv
