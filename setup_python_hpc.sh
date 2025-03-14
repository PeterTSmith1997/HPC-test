#!/bin/bash

# Ensure Miniconda is initialized for your shell
echo "Initializing Miniconda..."
~/miniconda3/bin/conda init

# Reload shell to apply changes
exec $SHELL

# Define environment name and Python version
ENV_NAME="hpc_env"
PYTHON_VERSION="3.9"

# Create a Conda environment
echo "Creating Conda environment: $ENV_NAME with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate the environment
echo "Activating Conda environment: $ENV_NAME..."
conda activate $ENV_NAME

# Install necessary packages
echo "Installing required packages..."
conda install numpy scipy matplotlib -y

echo "Setup complete. You are now using Conda environment: $ENV_NAME"
