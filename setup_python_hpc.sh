#!/bin/bash


# Define environment name and Python version
ENV_NAME="hpc_env"
PYTHON_VERSION="3.9"
ENV_PATH="$(pwd)/$ENV_NAME"

# Create a Conda environment in the current directory
echo "Creating Conda environment: $ENV_NAME in $(pwd) with Python $PYTHON_VERSION..."
conda create --prefix "$ENV_PATH" python=$PYTHON_VERSION -y

# Activate the environment
echo "Activating Conda environment: $ENV_PATH..."
source "$ENV_PATH/bin/activate"

# Install necessary packages
echo "Installing required packages..."
conda install numpy scipy matplotlib pytorch torchvision torchaudio nltk pytorch-cuda -c pytorch -c nvidia -y


echo "Setup complete. You are now using Conda environment: $ENV_PATH"
