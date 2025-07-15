#!/bin/bash

# Initialize conda for shell compatibility
eval "$(conda shell.bash hook)"
conda activate rm_dermo_env

python dir_creator.py

#Data Load
#echo "=== Data Load ==="
#python 1_dataset_load.py | tee outputs/load_log.txt

#Initialization
#echo "=== Preprocess ==="
#python 2_preprocess.py | tee outputs/preprocess_log.txt

#echo "=== Augmentation ==="
#python 3_augmentation.py | tee outputs/aug_log.txt



# Run the HEAD training and evaluation scripts
#echo "=== TRAINING HEAD ==="
#python train_head.py | tee outputs/head/results/train_head_log.txt

#echo "=== EVALUATION HEAD ==="
#python evaluate_head.py | tee outputs/head/results/evaluate_head_log.txt



# Run the FINE training and evaluation scripts
#echo "=== TRAINING FINE ==="
#python train_fine.py | tee outputs/fine/results/train_fine_log.txt

#echo "=== POSTPROCESSING ==="
python postprocessing.py | tee outputs/postprocessing_log.txt

echo "=== EVALUATION FINE ==="
python evaluate_fine.py | tee outputs/fine/results/evaluate_fine_log.txt