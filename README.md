# ENb4-CBAM: Dermoscopic Image Classifier (ISIC 2016)

This repository contains the code developed for the master's dissertation titled **"Dermoscopic Image Classification Using EfficientNet and Attention Mechanisms"**, which proposes a binary classification pipeline for the **ISIC 2016 Challenge Dataset**.

The proposed model, named **ENb4-CBAM**, leverages **EfficientNet-B4** and a **Convolutional Block Attention Module (CBAM)**. It includes extensive preprocessing, data augmentation, transfer learning, and a two-phase training scheme, achieving competitive results compared to dermatologists and state-of-the-art submissions.

---

## 🧠 Overview

The model classifies dermoscopic images as **benign** or **malignant**, with key components including:

- CLAHE contrast enhancement and hair removal
- Custom class-balanced data partitioning
- Albumentations-based augmentation pipeline
- EfficientNet-B4 + CBAM architecture
- Transfer learning from ImageNet
- Two-phase training (head → fine-tuning)
- Threshold calibration for fixed sensitivity

---

## 📁 Project Structure

├── 1_dataset_load.py # Download and extract ISIC 2016 dataset
├── 2_preprocess.py # Hair removal, CLAHE, and sharpening
├── 3_augmentation.py # Data partitioning and augmentation
├── model.py # ENb4-CBAM model definition
├── train_head.py # Train classification head only
├── evaluate_head.py # Evaluate head-only model
├── train_fine.py # Fine-tune the full model
├── evaluate_fine.py # Evaluate fine-tuned model
├── postprocessing.py # Threshold calibration and metric logging
├── run.sh # Script to execute the full pipeline
├── figures/ # Optional: plots and visualizations
└── outputs/ # Evaluation logs, plots, model weights


---

## ⚙️ Requirements

- Python 3.8+
- TensorFlow 2.11+
- OpenCV
- Albumentations
- scikit-learn
- Pillow
- matplotlib
- numpy
- tqdm

You can install the dependencies using:

```bash
pip install -r requirements.txt

Or using a conda environment (recommended):
conda create -n dermo_env python=3.8
conda activate dermo_env
pip install -r requirements.txt

---

## 🚀 Running the Pipeline

To run the full training and evaluation process:

```bash
bash run.sh

## 📊 Results

**Final metrics obtained on the ISIC 2016 test set:**

- **AUC:** 0.8788  
- **Sensitivity:** 85.33%  
- **Specificity:** 74.67%  
- **Accuracy:** 76.78%  
- **Benign class precision:** 95.38%  
- **Average Precision (AP):** 63.66%

The ENb4-CBAM model achieved results above the average diagnostic performance of dermatologists, and remains competitive with leading models from the ISIC 2016 challenge, without relying on ensembles or external datasets.

---

## 📂 Output Files

After running the pipeline, the following outputs will be generated in the `outputs/` directory:

- `results_head.json` – Evaluation metrics from head-only training  
- `results_fine.json` – Final evaluation metrics after fine-tuning  
- `confusion_matrix.png`, `roc_curve.png` – Diagnostic plots  
- `models/` – Folder with saved model checkpoints  
- `logs/` – (Optional) Training and evaluation logs

---

