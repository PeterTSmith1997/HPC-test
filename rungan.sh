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

# Set number of processes (one per task)
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(hostname)  # Set the master node address
export MASTER_PORT=12355        # Port for communication

# Run the training script using torchrun (PyTorch's utility for multi-node)
# We set the rank for each node automatically using SLURM's task environment variables
torchrun --nproc_per_node=$SLURM_NTASKS --nnodes=$SLURM_NODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py
