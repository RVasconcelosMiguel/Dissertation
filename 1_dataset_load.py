import os
import subprocess

# === Set root directory for your datasets (under shared disk) ===
dataset_root = "/raid/DATASETS/rmiguel_datasets/ISIC16"
os.makedirs(dataset_root, exist_ok=True)

# Create output folder for model, logs and PNGs
current_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(current_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# === Dataset download + extraction configuration ===
datasets = {
    "train_images": {
        "url": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip",
        "zip_path": os.path.join(dataset_root, "Train_Data.zip"),
        "extract_path": dataset_root,
        "final_name": "Training_Data"
    },
    "train_gt": {
        "url": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip",
        "zip_path": os.path.join(dataset_root, "Train_GT.zip"),
        "extract_path": dataset_root,
        "final_name": "Training_GroundTruth"
    },
    "test_images": {
        "url": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip",
        "zip_path": os.path.join(dataset_root, "Test_Data.zip"),
        "extract_path": dataset_root,
        "final_name": "Testing_Data"
    },
    "test_gt": {
        "url": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip",
        "zip_path": os.path.join(dataset_root, "Test_GT.zip"),
        "extract_path": dataset_root,
        "final_name": "Testing_GroundTruth"
    },
}

# === Clean previous extracted data (if needed, optional) ===
if os.path.exists(dataset_root):
    print(f"[INFO] Cleaning old dataset folder: {dataset_root}")
    subprocess.run(["rm", "-rf", dataset_root])
os.makedirs(dataset_root, exist_ok=True)

# === Download, unzip, rename datasets ===
for key, data in datasets.items():
    print(f"\n[INFO] --- Processing: {key} ---")

    if os.path.exists(data["zip_path"]):
        print(f"[INFO] Removing old zip file: {data['zip_path']}")
        os.remove(data["zip_path"])

    print(f"[INFO] Downloading from: {data['url']}")
    subprocess.run(["wget", "-q", "-O", data["zip_path"], data["url"]])

    print(f"[INFO] Extracting zip to: {data['extract_path']}")
    subprocess.run(["unzip", "-uq", data["zip_path"], "-d", data["extract_path"]])

    extracted_dirs = [
        d for d in os.listdir(data["extract_path"])
        if d.startswith("ISBI2016_ISIC_Part1") and os.path.isdir(os.path.join(data["extract_path"], d))
    ]

    if len(extracted_dirs) == 1:
        old_path = os.path.join(data["extract_path"], extracted_dirs[0])
        new_path = os.path.join(data["extract_path"], data["final_name"])
        os.rename(old_path, new_path)
        print(f"[INFO] Renamed folder: {old_path} → {new_path}")
    else:
        print(f"[WARNING] Unexpected folder structure in {data['extract_path']}. Please check manually.")

# === Download classification CSVs ===
csv_dir = os.path.join(dataset_root, "CSV")
os.makedirs(csv_dir, exist_ok=True)

csvs = {
    "Training_labels.csv": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv",
    "Testing_labels.csv": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.csv"
}

for filename, url in csvs.items():
    dest_path = os.path.join(csv_dir, filename)
    print(f"[INFO] Downloading CSV: {filename} from {url}")
    subprocess.run(["wget", "-q", "-O", dest_path, url])

# === Print final structure ===
print("\n[INFO] Final dataset structure:")
subprocess.run(["ls", "-lh", dataset_root])
