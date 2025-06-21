import os
import random
import shutil
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

# === Paths ===
df_path = "/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Training_labels.csv"
preprocessed_folder = "/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Training_Data"
augmented_folder = "/raid/DATASETS/rmiguel_datasets/ISIC16/Augmented_Training_Data"

# === Safety Check for Output Path ===
assert "rmiguel_datasets" in augmented_folder, "Unsafe output path! Aborting."

# === Create Output Directory ===
if os.path.exists(augmented_folder):
    print(f"Folder {augmented_folder} already exists. It will be deleted.")
    shutil.rmtree(augmented_folder)
os.makedirs(augmented_folder, exist_ok=True)

# === Load CSV ===
df = pd.read_csv(df_path, header=None, names=["image", "label"])

# Ensure image names end with .jpg
df['image'] = df['image'].astype(str).apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')

# Map 'benign' and 'malignant' to '0' and '1' if needed
if df['label'].dtype == object:
    df['label'] = df['label'].map({'benign': '0', 'malignant': '1'}).astype(str)

# === Copy all preprocessed images to augmented folder ===
for img_file in os.listdir(preprocessed_folder):
    src = os.path.join(preprocessed_folder, img_file)
    dst = os.path.join(augmented_folder, img_file)
    shutil.copy2(src, dst)

# === Count class frequencies ===
label_counts = df['label'].value_counts()
max_count = label_counts.max()

print("Initial class distribution:")
print(label_counts)

# === Perform Augmentation for Underrepresented Classes ===
new_rows = []

for cls, count in label_counts.items():
    if count >= max_count:
        print(f"Class '{cls}' is already balanced. Skipping.")
        continue

    samples = df[df['label'] == cls]
    augment_times = (max_count - count) // count
    print(f"Class '{cls}': augmenting each sample {augment_times} times.")

    for _, row in tqdm(samples.iterrows(), total=len(samples), desc=f"Augmenting class {cls}"):
        img_name = row['image']
        img_path = os.path.join(preprocessed_folder, img_name)

        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            continue

        for i in range(augment_times):
            aug_img = img.copy()

            # Basic augmentations
            if random.random() < 0.5:
                aug_img = ImageOps.mirror(aug_img)
            if random.random() < 0.5:
                aug_img = ImageOps.flip(aug_img)

            angle = random.uniform(-30, 30)
            aug_img = aug_img.rotate(angle)

            # Save with unique name
            base = os.path.splitext(img_name)[0]
            new_name = f"{base}_aug_{i}.jpg"
            save_path = os.path.join(augmented_folder, new_name)
            aug_img.save(save_path)

            new_rows.append({'image': new_name, 'label': cls})

# === Create Updated DataFrame ===
df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

print(f"\nTotal images after augmentation: {len(df_aug)}")
print(df_aug['label'].value_counts())

# === (Optional) Save New CSV ===
out_csv = "/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Augmented_Training_labels.csv"
df_aug.to_csv(out_csv, index=False, header=False)
print(f"New CSV saved to: {out_csv}")
