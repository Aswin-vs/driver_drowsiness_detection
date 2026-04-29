"""
cnn_model.py
------------
Custom CNN built entirely from scratch — the baseline model for this project.

This model learns drowsiness-related visual features (eye shape, eyelid position,
facial tension) directly from the binary dataset produced by prepare_dataset.py.
It receives no prior knowledge from any pre-trained source, which makes it a fair
baseline to compare against the MobileNetV2 transfer learning approach.

Dataset it is trained on:
    dataset_split/
        alert/   ← Open + no_yawn images from the original Kaggle 4-class set
        drowsy/  ← Closed + yawn images from the original Kaggle 4-class set

Architecture:
    Block 1 : Conv2D(32)  → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    Block 2 : Conv2D(64)  → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    Block 3 : Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    Head    : Flatten → Dense(256) → Dropout(0.5) → Dense(1, sigmoid)

L2 regularisation (1e-4) is applied to all Conv and Dense layers to reduce
overfitting on the relatively small ~2900-image dataset.
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout, Activation, Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_cnn(input_shape=(224, 224, 3), learning_rate=1e-3):
    """
    Builds, compiles, and returns the custom CNN model.

    Parameters
    ----------
    input_shape    : tuple  — (height, width, channels)
    learning_rate  : float  — Adam learning rate

    Returns
    -------
    model : compiled tf.keras.Sequential
    """
    model = Sequential(name="Custom_CNN")

    # ── Block 1 ───────────────────────────────────────────────────────────────
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Block 2 ───────────────────────────────────────────────────────────────
    model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Block 3 ───────────────────────────────────────────────────────────────
    model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Classification Head ───────────────────────────────────────────────────
    model.add(Flatten())
    model.add(Dense(256, activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))   # binary output

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model
