#!/bin/bash
source activate rm_dermo_env
python train.py
python evaluate.py