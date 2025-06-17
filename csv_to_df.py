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
from google.colab.patches import cv2_imshow

from PIL import Image, ImageFilter, ImageEnhance, ImageChops, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.optim import Adam, SGD, NAdam
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

import torchvision
from torchvision import models, transforms
from torchvision.io import read_image

from torchsummary import summary

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

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

df_preprocessed_train = df_train.copy()
df_preprocessed_test = df_test.copy()


######### Train preprocessed #########
print("\n\n -----------------Train-----------------")
print("\nDataframe head:")
print(df_preprocessed_train.head())

label_counts_train = df_preprocessed_train['label'].value_counts()

print("\nNumber of images per class:")
for label, count in label_counts_train.items():
    print(f"{label}: {count}")


######### Test preprocessed #########
print("\n\n -----------------Test-----------------")
print("\nDataframe head:")
print(df_preprocessed_test.head())

label_counts_test = df_preprocessed_test['label'].value_counts()

print("\nNumber of images per class:")
for label, count in label_counts_test.items():
    print(f"{label}: {count}")
print("\n---------------------------------------\n")


# === Split only the training dataframe ===
df_preprocessed_train_train, df_preprocessed_train_val = train_test_split(df_preprocessed_train, stratify=df_preprocessed_train['label'], test_size=0.15, random_state=42)

print("Train:", len(df_preprocessed_train_train), "Validation:", len(df_preprocessed_train_val), "Test:", len(df_preprocessed_test))