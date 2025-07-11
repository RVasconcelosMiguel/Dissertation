#!/bin/bash

# Initialize conda for shell compatibility
eval "$(conda shell.bash hook)"
conda activate rm_dermo_env



#Initialization
#echo "=== Preprocess ==="
#python 2_preprocess.py | tee ../files_to_transfer/preprocess_log.txt

echo "=== Augmentation ==="
python 3_augmentation.py



# Run the HEAD training and evaluation scripts

#echo "=== TRAINING HEAD ==="
#python train_head.py | tee ../files_to_transfer/efficientnetb4/head/train_head_log.txt

#echo "=== EVALUATION HEAD ==="
#python evaluate_head.py | tee ../files_to_transfer/efficientnetb4/head/evaluate_head_log.txt



# Run the FINE training and evaluation scripts
#echo "=== TRAINING FINE ==="
#python train_fine.py | tee ../files_to_transfer/efficientnetb4/fine/train_fine_log.txt

#echo "=== EVALUATION HEAD ==="
#python evaluate_fine.py | tee ../files_to_transfer/efficientnetb4/fine/evaluate_fine_log.txt
