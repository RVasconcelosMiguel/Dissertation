#!/bin/bash

# Initialize conda for shell compatibility
eval "$(conda shell.bash hook)"
conda activate rm_dermo_env



#Initialization
echo "=== Preprocess ==="
#python 2_preprocess.py

echo "=== Augmentation ==="
#python 3_augmentation.py



# Run the training and evaluation scripts
echo "=== TRAINING ==="
python train.py

echo "=== EVALUATION ==="
python evaluate.py
