import os
import logging
import subprocess

# === Set root directory for your datasets (under shared disk) ===
dataset_root = "/raid/DATASETS/rmiguel_datasets/ISIC16"
os.makedirs(dataset_root, exist_ok=True)

# === Setup logging to file ===
log_file_path = os.path.join(dataset_root, "dataset_download.log")

# Clear any existing logging handlers (for clean re-runs)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename=log_file_path,
    filemode='w',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Log file created successfully!")

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
    logging.info(f"Cleaning old dataset folder: {dataset_root}")
    subprocess.run(["rm", "-rf", dataset_root])
os.makedirs(dataset_root, exist_ok=True)

# === Download, unzip, rename datasets ===
for key, data in datasets.items():
    logging.info(f"--- Processing: {key} ---")

    if os.path.exists(data["zip_path"]):
        logging.info(f"Removing old zip file: {data['zip_path']}")
        os.remove(data["zip_path"])

    logging.info(f"Downloading from: {data['url']}")
    subprocess.run(["wget", "-q", "-O", data["zip_path"], data["url"]])

    logging.info(f"Extracting zip to: {data['extract_path']}")
    subprocess.run(["unzip", "-uq", data["zip_path"], "-d", data["extract_path"]])

    # Look for default ISIC-named extracted folder
    extracted_dirs = [
        d for d in os.listdir(data["extract_path"])
        if d.startswith("ISBI2016_ISIC_Part1") and os.path.isdir(os.path.join(data["extract_path"], d))
    ]

    if len(extracted_dirs) == 1:
        old_path = os.path.join(data["extract_path"], extracted_dirs[0])
        new_path = os.path.join(data["extract_path"], data["final_name"])
        os.rename(old_path, new_path)
        logging.info(f"Renamed folder: {old_path} → {new_path}")
    else:
        logging.warning(f"Unexpected folder structure found in {data['extract_path']}. Manual check may be needed.")

# === Download classification CSVs ===
csv_dir = os.path.join(dataset_root, "CSV")
os.makedirs(csv_dir, exist_ok=True)

csvs = {
    "Training_labels.csv": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv",
    "Testing_labels.csv": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.csv"
}

for filename, url in csvs.items():
    dest_path = os.path.join(csv_dir, filename)
    logging.info(f"Downloading CSV: {filename} from {url}")
    subprocess.run(["wget", "-q", "-O", dest_path, url])

# === Summary log ===
logging.info("Final dataset structure:")
result = subprocess.run(["ls", "-lh", dataset_root], capture_output=True, text=True)
logging.info("\n" + result.stdout)
