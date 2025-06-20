# augmentation.py
import os
import shutil
import random
from PIL import Image, ImageOps
from tqdm import tqdm

# === Folders ===
input_folder  = '/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Training_Data'
output_folder = '/raid/DATASETS/rmiguel_datasets/ISIC16/Augmented_Training_Data'

# Safety check
assert "rmiguel_datasets" in output_folder, "Unsafe output path. Aborting!"

# Prepare output folder: clear and recreate
if os.path.exists(output_folder):
    print(f"Folder {output_folder} already exists. It will be deleted and replaced.")
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

def random_augment(img):
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.5:
        img = ImageOps.flip(img)
    angle = random.uniform(-30, 30)
    return img.rotate(angle)

# List input images
all_images = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

# Step 1: Copy original images
for image_name in tqdm(all_images, desc="Copying original images"):
    src_path = os.path.join(input_folder, image_name)
    dst_path = os.path.join(output_folder, image_name)
    shutil.copy2(src_path, dst_path)

# Step 2: Apply basic augmentations
augmentations_per_image = 2  # You can increase this if needed

for image_name in tqdm(all_images, desc="Creating augmented images"):
    img_path = os.path.join(input_folder, image_name)
    try:
        img = Image.open(img_path)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        continue

    base_name = image_name[:-4]  # remove '.jpg'

    for i in range(augmentations_per_image):
        aug_img = img.copy()

        # Random horizontal and vertical flips
        aug_img = random_augment(img.copy())

        # Save with a new filename
        aug_filename = f"{base_name}_aug_{i}.jpg"
        aug_path = os.path.join(output_folder, aug_filename)
        aug_img.save(aug_path)

print(f"Copied {len(all_images)} original images.")
print(f"Created {len(all_images) * augmentations_per_image} augmented images.")
