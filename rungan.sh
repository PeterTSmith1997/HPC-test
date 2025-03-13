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

# Purge modules to prevent conflicts
module purge
module load CUDA/10.0.130_410.79  # Load CUDA if needed

# Activate Miniconda
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hpc_env

# Debugging Info
echo "Using Python: $(which python)"
echo "Using Conda Env: $(conda info --envs)"

# Run Python script (No MPI)
python main.py
