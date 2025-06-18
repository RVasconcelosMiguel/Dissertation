#!/bin/bash

# Initialize conda for shell compatibility
eval "$(conda shell.bash hook)"
conda activate rm_dermo_env

# Run the training and evaluation scripts
#python train.py
python evaluate.py