import os
import random
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
labels_path_train = '/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Training_labels.csv'
labels_path_test = '/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Testing_labels.csv'
train_image_dir = '/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Training_Data'
test_image_dir = '/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Testing_Data'

# === Load CSVs ===
df_train = pd.read_csv(labels_path_train, header=None, names=['image', 'label'])
df_test  = pd.read_csv(labels_path_test,  header=None, names=['image', 'label'])

# === Encode training labels (benign -> 0, malignant -> 1) ===
df_train['label'] = df_train['label'].map({'benign': 0, 'malignant': 1})
df_test['label'] = df_test['label'].astype(int)

# === Add '.jpg' extension to all image names ===
df_train['image'] = df_train['image'].astype(str) + '.jpg'
df_test['image']  = df_test['image'].astype(str)  + '.jpg'

# === Create copies for manipulation ===
df_preprocessed_train = df_train.copy()
df_preprocessed_test = df_test.copy()

# === TRAIN STATS ===
print("\n\n-----------------Train-----------------")
print("\nDataframe head:")
print(df_preprocessed_train.head())

label_counts_train = df_preprocessed_train['label'].value_counts()
print("\nNumber of images per class:")
for label, count in label_counts_train.items():
    print(f"{label}: {count}")

# === TEST STATS ===
print("\n\n-----------------Test-----------------")
print("\nDataframe head:")
print(df_preprocessed_test.head())

label_counts_test = df_preprocessed_test['label'].value_counts()
print("\nNumber of images per class:")
for label, count in label_counts_test.items():
    print(f"{label}: {count}")

print("\n---------------Data split---------------\n")

# === Stratified train/val split from training set ===
df_preprocessed_train_train, df_preprocessed_train_val = train_test_split(
    df_preprocessed_train,
    stratify=df_preprocessed_train['label'],
    test_size=0.15,
    random_state=42
)

print("Train:", len(df_preprocessed_train_train),
      "Validation:", len(df_preprocessed_train_val),
      "Test:", len(df_preprocessed_test))

print("\n---------------------------------------\n")
