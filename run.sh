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
echo "=== TRAINING HEAD ==="
python train_head.py | tee ../files_to_transfer/efficientnetb4/head/train_head_log.txt

echo "=== EVALUATION ==="
python evaluate_head.py | tee ../files_to_transfer/efficientnetb4/head/evaluate_head_log.txt
