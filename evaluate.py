import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

# Modern GPU memory management
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
    print("⚠️ No GPU found — using CPU.")

from tensorflow.keras.models import load_model
from data_loader import get_generators
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from plot_utils import save_confusion_matrix

# Load trained model
print("Loading trained model...")
model = load_model("models/efficientnetb0_isic16.h5")

# Load test data
print("Preparing test generator...")
_, _, test_gen = get_generators()

# Evaluate model
print("Evaluating model on test set...")
results = model.evaluate(test_gen)
for name, val in zip(model.metrics_names, results):
    print(f"{name}: {val:.4f}")

# Predict
print("Generating predictions...")
y_prob = model.predict(test_gen)
y_pred = (y_prob > 0.5).astype(int).flatten()
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

# Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# Confusion Matrix
print("Saving confusion matrix plot...")
save_confusion_matrix(y_true, y_pred, labels, "logs/confusion_matrix.png")
