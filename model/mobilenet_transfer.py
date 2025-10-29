"""
mobilenet_transfer.py
Lightweight Transfer Learning Model for Deepfake Detection
Based on MobileNetV3-Small (TensorFlow/Keras)
Author: Susmit Acharya | 2025
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_mobilenet_transfer(input_shape=(224, 224, 3),
                             learning_rate=1e-4,
                             freeze_layers_ratio=0.8,
                             dropout_rate=0.3):
    """
    Build and compile a lightweight deepfake detector using MobileNetV3-Small backbone.

    Args:
        input_shape (tuple): Input image dimensions.
        learning_rate (float): Learning rate for the Adam optimizer.
        freeze_layers_ratio (float): Fraction of base layers to freeze.
        dropout_rate (float): Dropout applied before classifier layers.

    Returns:
        model (tf.keras.Model): Compiled MobileNetV3-based model ready for training.
    """

    # ---------------------------------------------------------
    # 1. Load Pretrained Backbone
    # ---------------------------------------------------------
    base_model = MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # ---------------------------------------------------------
    # 2. Freeze Layers
    # ---------------------------------------------------------
    num_layers_to_freeze = int(len(base_model.layers) * freeze_layers_ratio)
    for layer in base_model.layers[:num_layers_to_freeze]:
        layer.trainable = False

    # ---------------------------------------------------------
    # 3. Add Custom Classifier
    # ---------------------------------------------------------
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    # ---------------------------------------------------------
    # 4. Compile Model
    # ---------------------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


if __name__ == "__main__":
    # Quick architecture test
    model = build_mobilenet_transfer()
