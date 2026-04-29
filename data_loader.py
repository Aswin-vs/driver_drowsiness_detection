"""
data_loader.py
--------------
Loads and preprocesses the binary alert/drowsy image dataset.

IMPORTANT — This file expects the SPLIT dataset produced by prepare_dataset.py,
NOT the raw Kaggle download. Run prepare_dataset.py first.

Expected folder structure (after running prepare_dataset.py):
    dataset_split/
        train/
            alert/          <- 70% of merged Open + no_yawn  (~1016 images)
            drowsy/         <- 70% of merged Closed + yawn   (~1014 images)
        val/
            alert/          <- 15% of merged Open + no_yawn  (~218 images)
            drowsy/         <- 15% of merged Closed + yawn   (~218 images)
        test/
            alert/          <- 15% of merged Open + no_yawn  (~218 images)
            drowsy/         <- 15% of merged Closed + yawn   (~217 images)

The test/ split is NEVER touched during training or callback decisions.
This ensures the final evaluation metrics in evaluate.py are on truly
unseen data, which is required for a valid comparative study.

Class encoding (Keras alphabetical order):
    alert  -> 0
    drowsy -> 1
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)   # MobileNetV2 expects 224x224; CNN uses same for fair comparison
BATCH_SIZE = 32
SEED       = 42


def get_data_generators(dataset_dir: str):
    """
    Returns (train_gen, val_gen, test_gen) Keras data generators.

    Reads from three separate subfolders: train/, val/, test/
    These are created by prepare_dataset.py with a 70/15/15 split.
    Augmentation is applied only to the training set.
    """
    train_path = os.path.join(dataset_dir, "train")
    val_path   = os.path.join(dataset_dir, "val")
    test_path  = os.path.join(dataset_dir, "test")

    for path in (train_path, val_path, test_path):
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Expected split folder not found: {path}\n\n"
                "Run this first:\n"
                "    python prepare_dataset.py --src dataset --out dataset_split\n\n"
                "Then re-run:  python train.py --dataset dataset_split"
            )

    # ── Augmentation for training only ────────────────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale          = 1.0 / 255,
        rotation_range   = 15,
        zoom_range       = 0.15,
        width_shift_range  = 0.1,
        height_shift_range = 0.1,
        horizontal_flip  = True,
        brightness_range = [0.8, 1.2],
        fill_mode        = "nearest",
    )

    # No augmentation for val / test — only normalise
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # ── Training generator ────────────────────────────────────────────────────
    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size = IMG_SIZE,
        batch_size  = BATCH_SIZE,
        class_mode  = "binary",     # alert=0, drowsy=1 (alphabetical)
        seed        = SEED,
        shuffle     = True,
    )

    # ── Validation generator ──────────────────────────────────────────────────
    val_gen = eval_datagen.flow_from_directory(
        val_path,
        target_size = IMG_SIZE,
        batch_size  = BATCH_SIZE,
        class_mode  = "binary",
        seed        = SEED,
        shuffle     = False,
    )

    # ── Test generator (truly held-out — never touched during training) ───────
    test_gen = eval_datagen.flow_from_directory(
        test_path,
        target_size = IMG_SIZE,
        batch_size  = BATCH_SIZE,
        class_mode  = "binary",
        seed        = SEED,
        shuffle     = False,
    )

    print("\nClass indices:", train_gen.class_indices)
    print(f"Training samples   : {train_gen.samples}")
    print(f"Validation samples : {val_gen.samples}")
    print(f"Test samples       : {test_gen.samples}  [held-out, never seen during training]\n")

    return train_gen, val_gen, test_gen


def load_single_image(image_path: str, model_input_size=(224, 224)):
    """
    Load and preprocess a single image for inference.
    Returns a numpy array of shape (1, H, W, 3).
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    img = load_img(image_path, target_size=model_input_size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)
