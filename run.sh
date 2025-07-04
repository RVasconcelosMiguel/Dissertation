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
python train.py | tee ../files_to_transfer/efficientnetb3/train_log.txt

echo "=== EVALUATION ==="
python evaluate.py | tee ../files_to_transfer/efficientnetb3/evaluate_log.txt
