"""
train.py
Deepfake Detection Model Training Script
Author: Susmit Acharya | 2025
"""

import os
import sys
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Import model architectures
from mesonet import build_mesonet
from mobilenet_transfer import build_mobilenet_transfer


# ---------------------------------------------------------
# 1. Resolve Base Paths Safely
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

if not os.path.exists(CONFIG_PATH):
    sys.exit(f"[ERROR] Configuration file not found at: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = os.path.join(BASE_DIR, cfg["data"]["processed_path"])
IMG_SIZE = tuple(cfg["model"]["input_shape"][:2])
BATCH_SIZE = cfg["model"]["batch_size"]
EPOCHS = cfg["model"]["epochs"]
LR = cfg["model"].get("learning_rate", 0.001)
MODEL_NAME = cfg["model"].get("architecture", "mobilenet_v3").lower()

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")

# ---------------------------------------------------------
# 2. Validate Dataset Structure
# ---------------------------------------------------------
for path in [train_dir, val_dir]:
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Dataset path not found: {path}\n"
                 "Please ensure your dataset is structured as:\n"
                 "data/processed/train/{authentic,fake}/\n"
                 "data/processed/val/{authentic,fake}/")

print(f"[INFO] Training data directory: {train_dir}")
print(f"[INFO] Validation data directory: {val_dir}")

# ---------------------------------------------------------
# 3. Prepare Data Generators
# ---------------------------------------------------------
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
)

val_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ---------------------------------------------------------
# 4. Build Selected Model
# ---------------------------------------------------------
print(f"\n[INFO] Initializing model: {MODEL_NAME.upper()} ...")

if MODEL_NAME == "mesonet":
    model = build_mesonet(input_shape=(*IMG_SIZE, 3), learning_rate=LR)
elif MODEL_NAME == "mobilenet_v3":
    model = build_mobilenet_transfer(input_shape=(*IMG_SIZE, 3), learning_rate=LR)
else:
    sys.exit(f"[ERROR] Unknown architecture: {MODEL_NAME}")

model.summary()

# ---------------------------------------------------------
# 5. Setup Callbacks & Output Paths
# ---------------------------------------------------------
os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "graphs"), exist_ok=True)

checkpoint_path = os.path.join(RESULTS_DIR, "models", f"{MODEL_NAME}_best.h5")

callbacks = [
    ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1, min_lr=1e-6)
]

# ---------------------------------------------------------
# 6. Train Model
# ---------------------------------------------------------
print(f"\n[INFO] Training started for {MODEL_NAME.upper()}...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---------------------------------------------------------
# 7. Save Final Model
# ---------------------------------------------------------
final_model_path = os.path.join(RESULTS_DIR, "models", f"{MODEL_NAME}_final.h5")
model.save(final_model_path)
print(f"\n✅ [INFO] Training complete. Final model saved to {final_model_path}")

# ---------------------------------------------------------
# 8. Plot & Save Training Curves
# ---------------------------------------------------------
def save_plot(metric_name, title, ylabel):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history[metric_name], label=f'Train {metric_name.capitalize()}')
    plt.plot(history.history[f'val_{metric_name}'], label=f'Validation {metric_name.capitalize()}')
    plt.title(f'{MODEL_NAME.upper()} - {title}')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    graph_path = os.path.join(RESULTS_DIR, "graphs", f"{MODEL_NAME}_{metric_name}_curve.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"[INFO] Saved: {graph_path}")

save_plot('accuracy', 'Accuracy over Epochs', 'Accuracy')
save_plot('loss', 'Loss over Epochs', 'Loss')

print(f"\n📊 [INFO] Training curves saved under {RESULTS_DIR}/graphs/")