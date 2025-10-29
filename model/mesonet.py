"""
mesonet.py
MesoNet: Lightweight CNN for Deepfake Detection
Adapted from Afchar et al. (2018), 'MesoNet: a Compact Facial Video Forgery Detection Network'
Author: Susmit Acharya | 2025
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU, MaxPooling2D,
    Dropout, Flatten, Dense
)
from tensorflow.keras.optimizers import Adam


def build_mesonet(input_shape=(224, 224, 3), learning_rate=0.001):
    """
    Builds the MesoNet architecture.
    A lightweight CNN designed for efficient deepfake detection.

    Args:
        input_shape (tuple): Shape of input images (H, W, C).
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        model (tf.keras.Model): Compiled Keras model ready for training.
    """

    model = Sequential([
        # Block 1
        Conv2D(8, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # Block 2
        Conv2D(8, (5, 5), padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # Block 3
        Conv2D(16, (5, 5), padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # Block 4
        Conv2D(16, (5, 5), padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),

        # Fully connected classifier
        Flatten(),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


if __name__ == "__main__":
    # Quick test
    model = build_mesonet()
