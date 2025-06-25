import os
import random
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
import cv2

# === Paths ===
df_path = "/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Training_labels.csv"
preprocessed_folder = "/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Training_Data"
augmented_folder = "/raid/DATASETS/rmiguel_datasets/ISIC16/Augmented_Training_Data"
out_csv = "/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Augmented_Training_labels.csv"

# === Safety Check for Output Path ===
assert "rmiguel_datasets" in augmented_folder, "Unsafe output path! Aborting."

# === Create Output Directory ===
if os.path.exists(augmented_folder):
    print(f"Folder {augmented_folder} already exists. It will be deleted.")
    shutil.rmtree(augmented_folder)
os.makedirs(augmented_folder, exist_ok=True)

# === Load CSV ===
df = pd.read_csv(df_path, header=None, names=["image", "label"])
df['image'] = df['image'].astype(str).apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')
if df['label'].dtype == object:
    df['label'] = df['label'].map({'benign': '0', 'malignant': '1'}).astype(str)

# === Copy all preprocessed images to augmented folder ===
for img_file in os.listdir(preprocessed_folder):
    shutil.copy2(os.path.join(preprocessed_folder, img_file), os.path.join(augmented_folder, img_file))

# === Count class frequencies ===
label_counts = df['label'].value_counts()
print("Initial class distribution:")
print(label_counts)

# === Set unified target count for both classes ===
target_count = 1500

# === Augmentation pipeline ===
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.ElasticTransform(alpha=0.5, sigma=20, p=0.1),  
    A.ISONoise(color_shift=(0.01, 0.01), intensity=(0.01, 0.03), p=0.1),  
    A.Resize(224, 224)
])

# === Perform Augmentation to reach target_count for each class ===
new_rows = []

for cls, count in label_counts.items():
    if count >= target_count:
        print(f"Class '{cls}' already has {count} samples. Skipping.")
        continue

    samples = df[df['label'] == cls]
    augment_needed = target_count - count
    augment_times = augment_needed // len(samples)
    remainder = augment_needed % len(samples)

    print(f"Class '{cls}': augmenting {len(samples)} samples to generate {augment_needed} new images.")
    print(f"Each image will be augmented {augment_times} times, with {remainder} extra augmentations.")

    for idx, (_, row) in enumerate(tqdm(samples.iterrows(), total=len(samples), desc=f"Augmenting class {cls}")):
        img_name = row['image']
        img_path = os.path.join(preprocessed_folder, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            continue

        reps = augment_times + (1 if idx < remainder else 0)

        for i in range(reps):
            augmented = augment(image=img_np)
            aug_img = Image.fromarray(augmented['image'])
            base = os.path.splitext(img_name)[0]
            new_name = f"{base}_aug_{i}.jpg"
            save_path = os.path.join(augmented_folder, new_name)
            aug_img.save(save_path)
            new_rows.append({'image': new_name, 'label': cls})

# === Create Updated DataFrame ===
df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
print(f"\nTotal images after augmentation: {len(df_aug)}")
print(df_aug['label'].value_counts())

# === Save New CSV ===
df_aug.to_csv(out_csv, index=False, header=False)
print(f"New CSV saved to: {out_csv}")
