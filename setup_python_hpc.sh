#!/bin/bash

# Exit on error
set -e

# Define environment name and Python version
ENV_NAME="hpc_env"
PYTHON_VERSION="3.9"
ENV_PATH="$(pwd)/$ENV_NAME"

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in your PATH."
    exit 1
fi

# Check if the environment already exists
if conda env list | grep -q "$ENV_PATH"; then
    echo "Conda environment '$ENV_NAME' already exists at $ENV_PATH."
else
    # Create the Conda environment
    echo "Creating Conda environment: $ENV_NAME at $ENV_PATH with Python $PYTHON_VERSION..."
    conda create --prefix "$ENV_PATH" python=$PYTHON_VERSION -y
fi

# Ensure Conda is properly initialized
eval "$(conda shell.bash hook)"

# Activate the environment
echo "Activating Conda environment: $ENV_PATH..."
conda activate "$ENV_PATH" || { echo "Error: Failed to activate Conda environment."; exit 1; }

# Install necessary packages
echo "Installing required packages..."
conda install numpy scipy matplotlib pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia -y

echo "Setup complete. You are now using Conda environment: $ENV_PATH"
