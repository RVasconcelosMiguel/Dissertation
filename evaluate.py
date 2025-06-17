import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from tensorflow.keras.models import load_model
from data_loader import get_generators
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from plot_utils import save_confusion_matrix

# Debug GPU usage
print("Available GPU devices:", tf.config.list_physical_devices('GPU'))

# Load model and data
print("📦 Loading trained model...")
model = load_model("models/efficientnetb0_isic16.h5")
_, _, test_gen = get_generators()

# Evaluate
print("📊 Running evaluation...")
results = model.evaluate(test_gen)
for name, val in zip(model.metrics_names, results):
    print(f"{name}: {val:.4f}")

# Predict and report
print("🔍 Generating predictions...")
y_prob = model.predict(test_gen)
y_pred = (y_prob > 0.5).astype(int).flatten()
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

print("\n📑 Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# Confusion Matrix
save_confusion_matrix(y_true, y_pred, labels, "logs/confusion_matrix.png")
