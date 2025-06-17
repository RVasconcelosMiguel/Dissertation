import os
import time
import random
import zipfile
import json
import logging
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import cv2

from PIL import Image, ImageFilter, ImageEnhance, ImageChops, ImageOps


# Define paths
labels_path_train = '/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Training_labels.csv'
labels_path_test = '/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Testing_labels.csv'

# === Load CSVs ===
df_train = pd.read_csv(labels_path_train, header=None, names=['image', 'label'])
df_test  = pd.read_csv(labels_path_test,  header=None, names=['image', 'label'])

# === Encode training labels (benign -> 0, malignant -> 1) ===
df_train['label'] = df_train['label'].map({'benign': 0, 'malignant': 1})
df_test['label'] = df_test['label'].astype(int)

# === Add '.jpg' extension to all image names ===
df_train['image'] = df_train['image'].astype(str) + '.jpg'
df_test['image']  = df_test['image'].astype(str)  + '.jpg'

df_preprocessed_train = df_train.copy()
df_preprocessed_test = df_train.copy()

print(df_preprocessed_train.head())

print(df_preprocessed_test.head())

label_counts_train = df_preprocessed_train['label'].value_counts()

print("Number of images per class (train):")
for label, count in label_counts_train.items():
    print(f"{label}: {count}")

label_counts_test = df_preprocessed_test['label'].value_counts()

print("Number of images per class (test):")
for label, count in label_counts_test.items():
    print(f"{label}: {count}")