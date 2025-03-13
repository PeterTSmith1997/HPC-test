#!/bin/bash
#SBATCH --job-name=Test_gan_training
#SBATCH --output=test_gan_%j.txt
#SBATCH --time=08:00:00
#SBATCH --nodes=2                   # Use 2 compute nodes
#SBATCH --ntasks-per-node=4         # Use 4 tasks per node
#SBATCH --gres=gpu:1                # Request 1 GPU per node
#SBATCH --cpus-per-task=4           # Use 4 CPU cores per task
#SBATCH --mem=32GB                  # Request 32GB memory
#SBATCH --partition=gpu              # Use GPU partition (check `sinfo`)
#SBATCH --mail-user=peter.t.smith@northumbria.ac.uk
#SBATCH --mail-type=ALL

# Load modules
module load anaconda3/5.3.1
module load CUDA/10.0.130_410.79
module load openmpi/gcc/64/1.10.4

# Activate Virtual Environment
source ~/my_project/.venv/bin/activate

# Run with MPI
mpirun -np 8 python main.py
