import os
import cv2
import shutil
from PIL import Image, ImageFilter, ImageChops
from tqdm import tqdm

# === CONFIGURATION ===
use_binary_mask = True  # Set to False to disable mask application

# === Folders ===
input_folders = {
    'train': '/raid/DATASETS/rmiguel_datasets/ISIC16/Training_Data',
    'test': '/raid/DATASETS/rmiguel_datasets/ISIC16/Testing_Data'
}
output_folders = {
    'train': '/raid/DATASETS/rmiguel_datasets/ISIC16/Classification/Preprocessed_Training_Data',
    'test': '/raid/DATASETS/rmiguel_datasets/ISIC16/Classification/Preprocessed_Testing_Data'
}
mask_folders = {
    'train': '/raid/DATASETS/rmiguel_datasets/ISIC16/Training_GroundTruth',
    'test': '/raid/DATASETS/rmiguel_datasets/ISIC16/Testing_GroundTruth'
}

# === Preprocessing Steps ===
def apply_clahe_rgb(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

def remove_hairs(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image_rgb, mask, 1, cv2.INPAINT_TELEA)

def normalize_illumination(image_rgb):
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(image_lab)
    l = cv2.equalizeHist(l)
    normalized = cv2.merge((l, a, b))
    return cv2.cvtColor(normalized, cv2.COLOR_LAB2RGB)

def sharpen_channel_pil(channel, blur_radius=2, scale=0.5):
    smoothed = channel.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    edge_mask = ImageChops.subtract(channel, smoothed)
    scaled_edge = edge_mask.point(lambda p: p * scale)
    return ImageChops.add(channel, scaled_edge)

# === Main Processing Loop ===
for key in input_folders:
    original_folder = input_folders[key]
    processed_folder = output_folders[key]
    mask_folder = mask_folders[key]

    # Safety: prevent accidental deletion of important paths
    assert "rmiguel_datasets" in processed_folder, "Unsafe output path! Aborting."

    # Create output directory
    if os.path.exists(processed_folder):
        print(f"Folder {processed_folder} already exists. It will be deleted and replaced.")
        shutil.rmtree(processed_folder)
    os.makedirs(processed_folder, exist_ok=True)

    all_images = [f for f in os.listdir(original_folder) if f.endswith('.jpg')]

    for image_name in tqdm(all_images, desc=f"Processing {key}"):
        img_path = os.path.join(original_folder, image_name)
        processed_path = os.path.join(processed_folder, image_name)

        # Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Error loading: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # === Apply mask if configured ===
        if use_binary_mask:
            mask_name = image_name.replace('.jpg', '_Segmentation.png')
            mask_path = os.path.join(mask_folder, mask_name)

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Ensure mask is binary (0 or 255)
                    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    # Resize mask to image size if needed
                    if mask_bin.shape != img_rgb.shape[:2]:
                        mask_bin = cv2.resize(mask_bin, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # Apply mask
                    img_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_bin)
                else:
                    print(f"Mask could not be loaded for: {image_name}")
            else:
                print(f"Mask not found for: {image_name}")

        # === Preprocessing pipeline ===
        img_rgb = remove_hairs(img_rgb)
        # img_rgb = apply_clahe_rgb(img_rgb)             # Optional
        # img_rgb = normalize_illumination(img_rgb)      # Optional

        # === Channel sharpening ===
        img_pil = Image.fromarray(img_rgb)
        r, g, b = img_pil.split()
        r = sharpen_channel_pil(r)
        g = sharpen_channel_pil(g)
        b = sharpen_channel_pil(b)
        img_sharp = Image.merge('RGB', (r, g, b))

        # === Save processed image ===
        img_sharp.save(processed_path)

    print(f"{key.upper()}: {len(all_images)} images processed.")
