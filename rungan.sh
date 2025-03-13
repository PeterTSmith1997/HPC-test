#!/bin/bash
#SBATCH --job-name=Test_gan_training
#SBATCH --output=test_gan_%j.txt
#SBATCH --time=08:00:00
#SBATCH --nodes=2                   # Use 2 compute nodes
#SBATCH --ntasks-per-node=4         # Use 4 tasks per node
#SBATCH --cpus-per-task=4           # Use 4 CPU cores per task
#SBATCH --mem=32GB                  # Request 32GB memory
#SBATCH --partition=48hour          # Use GPU partition (check `sinfo`)
#SBATCH --mail-user=peter.t.smith@northumbria.ac.uk
#SBATCH --mail-type=ALL

# Load Modules (Remove Anaconda if using Miniconda)
module purge  # Clear all loaded modules to prevent conflicts
module load CUDA/10.0.130_410.79
module load openmpi/gcc/64/1.10.4

# Use Miniconda installed in your home directory
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your HPC Conda Environment
conda activate hpc_env

# Debugging Info (Optional)
echo "Using Python: $(which python)"
echo "Using Conda Env: $(conda info --envs)"

# Run with MPI
mpirun -np 8 python main.py  # Use 8 total processes across nodes
