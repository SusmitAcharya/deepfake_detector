"""
evaluate.py
Robust evaluation script for Deepfake Detection models.
Fixes .h5 loading issues by rebuilding model architecture & loading weights only.

Author: Susmit Acharya | 2025
"""

import os
import sys
import glob
import time
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import your model builder functions
from mobilenet_transfer import build_mobilenet_transfer
from mesonet import build_mesonet


# -------------------------
# Helper functions
# -------------------------
def abort(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


def find_config():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(this_dir, "..", "config.yaml"),
        os.path.join(this_dir, "config.yaml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    abort("❌ Could not find config.yaml in root or model folder.")


def find_latest_model(model_dir):
    model_dir = os.path.abspath(model_dir)
    if not os.path.exists(model_dir):
        abort(f"❌ Model directory not found: {model_dir}")
    models = glob.glob(os.path.join(model_dir, "*.h5")) + glob.glob(os.path.join(model_dir, "*.keras"))
    if not models:
        abort(f"❌ No model files (.h5/.keras) found in {model_dir}")
    return max(models, key=os.path.getmtime)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# -------------------------
# Load configuration
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = find_config()

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = (
    os.path.join(BASE_DIR, cfg["data"]["processed_path"])
    if not os.path.isabs(cfg["data"]["processed_path"])
    else cfg["data"]["processed_path"]
)
IMG_SIZE = tuple(cfg["model"]["input_shape"][:2])
BATCH_SIZE = cfg["model"]["batch_size"]
LEARNING_RATE = cfg["model"]["learning_rate"]
ARCH = cfg["model"].get("architecture", "mobilenet_v3").lower()

RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")

ensure_dirs(METRICS_DIR, GRAPHS_DIR, MODEL_DIR)

# -------------------------
# Locate model file
# -------------------------
MODEL_PATH = find_latest_model(MODEL_DIR)
print(f"[INFO] Using model weights: {MODEL_PATH}")

# -------------------------
# Prepare validation data
# -------------------------
val_dir = os.path.join(DATA_DIR, "val")
if not os.path.exists(val_dir):
    abort(f"❌ Validation directory not found: {val_dir}")

datagen = ImageDataGenerator(rescale=1.0 / 255)
print(f"[INFO] Loading validation data from: {val_dir}")

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

if val_data.samples == 0:
    abort(f"❌ No images found in {val_dir}. Please extract frames first.")

# -------------------------
# Rebuild model & load weights
# -------------------------
print(f"[INFO] Rebuilding architecture: {ARCH.upper()}")

try:
    if ARCH == "mesonet":
        model = build_mesonet(input_shape=IMG_SIZE + (3,), learning_rate=LEARNING_RATE)
    elif ARCH in ["mobilenet_v3", "mobilenet"]:
        model = build_mobilenet_transfer(input_shape=IMG_SIZE + (3,), learning_rate=LEARNING_RATE)
    else:
        abort(f"❌ Unknown architecture: {ARCH}")
except Exception as e:
    abort(f"❌ Failed to build model architecture: {e}")

# Load weights only
try:
    model.load_weights(MODEL_PATH)
    print("[INFO] Weights loaded successfully from .h5 file.")
except Exception as e:
    abort(f"❌ Failed to load weights from {MODEL_PATH}: {e}")

# -------------------------
# Run inference
# -------------------------
print("[INFO] Running inference on validation set...")
start_time = time.time()
y_pred_probs = model.predict(val_data, verbose=1)
end_time = time.time()

y_pred_probs = np.asarray(y_pred_probs).reshape(-1)
y_pred = (y_pred_probs > 0.5).astype(int)
y_true = val_data.classes

# -------------------------
# Compute metrics
# -------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
latency = (end_time - start_time) / max(1, len(y_true))
fps = 1.0 / latency if latency > 0 else float("inf")

try:
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
except Exception:
    fpr, tpr, roc_auc = None, None, None

# -------------------------
# Print & save summary
# -------------------------
summary_path = os.path.join(METRICS_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write("Evaluation Summary\n===================\n")
    f.write(f"Model weights: {os.path.basename(MODEL_PATH)}\n")
    f.write(f"Validation samples: {len(y_true)}\n\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")
    if roc_auc:
        f.write(f"ROC-AUC:   {roc_auc:.4f}\n")
    f.write(f"Inference per frame: {latency:.5f}s ({fps:.2f} FPS)\n")

print("\n[RESULTS]")
print(open(summary_path).read())

# -------------------------
# Classification report
# -------------------------
report = classification_report(y_true, y_pred, target_names=list(val_data.class_indices.keys()))
with open(os.path.join(METRICS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# -------------------------
# Confusion matrix
# -------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=list(val_data.class_indices.keys()),
    yticklabels=list(val_data.class_indices.keys()),
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "confusion_matrix_eval.png"))
plt.close()

# -------------------------
# ROC Curve
# -------------------------
if fpr is not None:
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "roc_curve.png"))
    plt.close()

print("\n[INFO] Evaluation complete.")
print(f"[INFO] Results saved in: {METRICS_DIR} and {GRAPHS_DIR}")