import os
import shutil
import random
from PIL import Image, ImageOps
from tqdm import tqdm

input_folder = '/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Training_Data'
output_folder = '/raid/DATASETS/rmiguel_datasets/ISIC16/Augmented_Training_Data'

assert "rmiguel_datasets" in output_folder, "Unsafe output path. Aborting!"

# Prepare output folder
if os.path.exists(output_folder):
    print(f"Folder {output_folder} exists. Deleting...")
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Get list of classes (subfolders)
classes = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]

# Count images per class
class_counts = {}
for cls in classes:
    cls_path = os.path.join(input_folder, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith('.jpg')]
    class_counts[cls] = len(images)

max_count = max(class_counts.values())

print("Class counts before augmentation:")
print(class_counts)

def random_augment(img):
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.5:
        img = ImageOps.flip(img)
    angle = random.uniform(-30, 30)
    return img.rotate(angle)

# Copy originals and augment minority classes
for cls in classes:
    src_cls_path = os.path.join(input_folder, cls)
    dst_cls_path = os.path.join(output_folder, cls)
    os.makedirs(dst_cls_path, exist_ok=True)

    images = [f for f in os.listdir(src_cls_path) if f.lower().endswith('.jpg')]
    count = class_counts[cls]

    # Copy original images
    for img_name in tqdm(images, desc=f"Copy originals for class {cls}"):
        shutil.copy2(os.path.join(src_cls_path, img_name), os.path.join(dst_cls_path, img_name))

    # Determine how many times to augment each image
    if count < max_count:
        augment_times = max_count // count - 1
        print(f"Augmenting class '{cls}' {augment_times} times per image.")

        for img_name in tqdm(images, desc=f"Augmenting class {cls}"):
            img_path = os.path.join(src_cls_path, img_name)
            img = Image.open(img_path)
            base_name = img_name[:-4]

            for i in range(augment_times):
                aug_img = random_augment(img.copy())
                aug_filename = f"{base_name}_aug_{i}.jpg"
                aug_img.save(os.path.join(dst_cls_path, aug_filename))

print("Augmentation complete.")