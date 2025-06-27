import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split

# === Paths ===
csv_path = "/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Training_labels.csv"
preprocessed_folder = "/raid/DATASETS/rmiguel_datasets/ISIC16/Classification/Preprocessed_Training_Data"
split_base = "/raid/DATASETS/rmiguel_datasets/ISIC16/Classification/Split"
train_folder = os.path.join(split_base, "train")
val_folder = os.path.join(split_base, "val")

# === Safety Check ===
assert "rmiguel_datasets" in split_base, "Unsafe output path. Aborting."

# === Create folders ===
for folder in [train_folder, val_folder]:
    if os.path.exists(folder):
        print(f"[INFO] Deleting existing folder: {folder}")
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# === Load CSV ===
df = pd.read_csv(csv_path, header=None, names=["image", "label"])
df['image'] = df['image'].astype(str).apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')
df['label'] = df['label'].map({'benign': 0, 'malignant': 1}).astype(int)

# === Split before augmentation ===
train_df, val_df = train_test_split(df, stratify=df['label'], test_size=0.15, random_state=42)

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

# === Augment training set to balance classes ===
target_count = 1500
train_aug_rows = []

label_counts = train_df['label'].value_counts()
print("Initial TRAIN class distribution:")
print(label_counts)

for cls in label_counts.index:
    samples = train_df[train_df['label'] == cls]
    current_count = len(samples)

    # Copy originals
    for _, row in samples.iterrows():
        img_name = row['image']
        src = os.path.join(preprocessed_folder, img_name)
        dst = os.path.join(train_folder, img_name)
        shutil.copy2(src, dst)
        train_aug_rows.append({'image': img_name, 'label': cls})

    # Determine augmentations needed
    augment_needed = target_count - current_count
    if augment_needed <= 0:
        print(f"Class {cls} already has {current_count} samples. No augmentation needed.")
        continue

    augment_times = augment_needed // current_count
    remainder = augment_needed % current_count

    print(f"Class {cls}: augmenting {current_count} images to generate {augment_needed} new images.")

    for idx, (_, row) in enumerate(tqdm(samples.iterrows(), total=current_count, desc=f"Augmenting class {cls}")):
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
            save_path = os.path.join(train_folder, new_name)
            aug_img.save(save_path)
            train_aug_rows.append({'image': new_name, 'label': cls})

# === Process validation set (copy only, no augmentation) ===
val_rows = []

for _, row in val_df.iterrows():
    img_name = row['image']
    src = os.path.join(preprocessed_folder, img_name)
    dst = os.path.join(val_folder, img_name)
    shutil.copy2(src, dst)
    val_rows.append({'image': img_name, 'label': row['label']})

# === Save CSVs ===
pd.DataFrame(train_aug_rows).to_csv(os.path.join(split_base, "train/train_labels.csv"), index=False, header=False)
pd.DataFrame(val_rows).to_csv(os.path.join(split_base, "val/val_labels.csv"), index=False, header=False)

print(f"\n[INFO] Final TRAIN images: {len(train_aug_rows)}")
print(f"[INFO] Final VAL images: {len(val_rows)}")
print("[INFO] CSV files saved.")
