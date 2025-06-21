#!/bin/bash

# Initialize conda for shell compatibility
eval "$(conda shell.bash hook)"
conda activate rm_dermo_env

# Run the training and evaluation scripts
echo "=== TRAINING ==="
python train.py
echo "=== EVALUATION ==="
python evaluate.py
echo "=== THRESHOLD TUNING (ROC-based) ==="
python threshold_tuning.py