import os
# Set environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys

# Redirect stdout and stderr to log file
os.makedirs("/home/jtstudents/rmiguel/files_to_transfer", exist_ok=True)
log_path = "/home/jtstudents/rmiguel/files_to_transfer/evaluate_log.txt"
log_file = open(log_path, "w")
sys.stdout = log_file
sys.stderr = log_file

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Using GPU: {logical_gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found — using CPU.")

from tensorflow.keras.models import load_model
from data_loader import get_generators
from sklearn.metrics import classification_report
import numpy as np
from plot_utils import save_confusion_matrix

IMG_SIZE = 224
BATCH_SIZE = 32

model_path = "models/efficientnetb0_isic16.h5"
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please check training phase completed successfully.")
model = load_model(model_path)
print(f"Loaded trained model from {model_path}")

print("Preparing test generator...")
_, _, test_gen = get_generators(img_size=IMG_SIZE, batch_size=BATCH_SIZE)

print("Evaluating model on test set...")
results = model.evaluate(test_gen)
for name, val in zip(model.metrics_names, results):
    print(f"{name}: {val:.4f}")

print("Generating predictions...")
y_prob = model.predict(test_gen)
y_pred = (y_prob > 0.5).astype(int).flatten()

y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

os.makedirs("logs", exist_ok=True)
print("Saving confusion matrix plot...")
save_confusion_matrix(y_true, y_pred, labels, "logs/confusion_matrix.png")

print("Evaluation complete.")
log_file.close()
