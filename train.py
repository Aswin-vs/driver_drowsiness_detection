"""
train.py
--------
Trains both models (CNN from scratch + MobileNetV2 transfer learning)
on the binary alert/drowsy dataset and saves their weights for evaluation.

PREREQUISITE — Run prepare_dataset.py first:
    python prepare_dataset.py --src dataset --out dataset_split

    This produces a proper 3-way train/val/test split:
        dataset_split/
            train/  alert/ + drowsy/   <- 70%
            val/    alert/ + drowsy/   <- 15%  (used by callbacks)
            test/   alert/ + drowsy/   <- 15%  (held-out, never seen here)

Usage:
    python train.py --dataset dataset_split

Output written to saved_models/:
    cnn_best.keras          <- best CNN weights (monitored on val_accuracy)
    mobilenet_best.keras    <- best MobileNetV2 weights (Phase 2, val_accuracy)
    cnn_history.npy         <- training history dict for CNN
    mobilenet_history.npy   <- combined Phase 1 + Phase 2 history for MobileNetV2
"""

import os
import random
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard,
)

from data_loader     import get_data_generators
from cnn_model       import build_cnn
from mobilenet_model import build_mobilenet, unfreeze_for_fine_tuning

# ── Global seeds for reproducibility ──────────────────────────────────────────
# Set before any model or data operations so weight init and shuffling are
# deterministic across runs on the same machine.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS_CNN        = 30
EPOCHS_MN_PHASE1  = 10    # head-only training (base frozen)
EPOCHS_MN_PHASE2  = 20    # fine-tuning (top base layers unfrozen)
LEARNING_RATE_CNN = 1e-3
LEARNING_RATE_MN  = 1e-3
FINE_TUNE_LR      = 1e-5
OUTPUT_DIR        = "saved_models"


def get_checkpoint_callbacks(model_name: str, monitor: str = "val_accuracy"):
    """
    Callbacks that save the best checkpoint, stop early, and reduce LR.
    Used for: CNN training, and MobileNetV2 Phase 2 (the final model).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return [
        ModelCheckpoint(
            filepath      = os.path.join(OUTPUT_DIR, f"{model_name}_best.keras"),
            monitor       = monitor,
            save_best_only= True,
            verbose       = 1,
        ),
        EarlyStopping(
            monitor              = monitor,
            patience             = 7,
            restore_best_weights = True,
            verbose              = 1,
        ),
        ReduceLROnPlateau(
            monitor = "val_loss",
            factor  = 0.5,
            patience= 4,
            min_lr  = 1e-7,
            verbose = 1,
        ),
        TensorBoard(log_dir=os.path.join("logs", model_name)),
    ]


def get_phase1_callbacks():
    """
    Callbacks for MobileNetV2 Phase 1 (head-only warm-up).
    No ModelCheckpoint here — Phase 1 weights are intermediate and are
    never loaded again; saving them wastes ~14 MB and is misleading.
    Phase 2 will save the final mobilenet_best.keras checkpoint.
    """
    return [
        EarlyStopping(
            monitor              = "val_accuracy",
            patience             = 7,
            restore_best_weights = True,
            verbose              = 1,
        ),
        ReduceLROnPlateau(
            monitor = "val_loss",
            factor  = 0.5,
            patience= 4,
            min_lr  = 1e-7,
            verbose = 1,
        ),
        TensorBoard(log_dir=os.path.join("logs", "mobilenet_phase1")),
    ]


# ── CNN ────────────────────────────────────────────────────────────────────────
def train_cnn(train_gen, val_gen):
    print("\n" + "=" * 60)
    print("  Training CNN from Scratch")
    print("=" * 60)

    model   = build_cnn(learning_rate=LEARNING_RATE_CNN)
    history = model.fit(
        train_gen,
        epochs          = EPOCHS_CNN,
        validation_data = val_gen,
        callbacks       = get_checkpoint_callbacks("cnn"),
        verbose         = 1,
    )

    np.save(os.path.join(OUTPUT_DIR, "cnn_history.npy"), history.history)
    print("\nCNN training complete. Best model saved.")
    return model, history


# ── MobileNetV2 ────────────────────────────────────────────────────────────────
def train_mobilenet(train_gen, val_gen):
    # ── Phase 1: train only the classification head ───────────────────────────
    print("\n" + "=" * 60)
    print("  Training MobileNetV2 — Phase 1 (Feature Extraction)")
    print("  Base model frozen. Only the classification head is trained.")
    print("=" * 60)

    model      = build_mobilenet(learning_rate=LEARNING_RATE_MN)
    history_p1 = model.fit(
        train_gen,
        epochs          = EPOCHS_MN_PHASE1,
        validation_data = val_gen,
        callbacks       = get_phase1_callbacks(),   # no checkpoint — intermediate weights
        verbose         = 1,
    )

    # ── Phase 2: fine-tune top base layers at very low LR ────────────────────
    print("\n" + "=" * 60)
    print("  Training MobileNetV2 — Phase 2 (Fine-Tuning)")
    print("  Top 100 base layers unfrozen at LR=1e-5.")
    print("=" * 60)

    model      = unfreeze_for_fine_tuning(model, fine_tune_lr=FINE_TUNE_LR)
    history_p2 = model.fit(
        train_gen,
        epochs          = EPOCHS_MN_PHASE2,
        validation_data = val_gen,
        callbacks       = get_checkpoint_callbacks("mobilenet"),  # saves mobilenet_best.keras
        verbose         = 1,
    )

    # Merge both phase histories so evaluate.py can plot the full training curve
    combined = {}
    for key in history_p1.history:
        combined[key] = history_p1.history[key] + history_p2.history.get(key, [])

    np.save(os.path.join(OUTPUT_DIR, "mobilenet_history.npy"), combined)
    print("\nMobileNetV2 training complete. Best model saved.")
    return model, combined


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train drowsiness detection models")
    parser.add_argument(
        "--dataset", type=str, default="dataset_split",
        help="Root folder produced by prepare_dataset.py "
             "(must contain train/, val/, test/ sub-folders).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        raise FileNotFoundError(
            f"Dataset directory '{args.dataset}' not found.\n\n"
            "Run this first:\n"
            "    python prepare_dataset.py --src dataset --out dataset_split\n\n"
            "Then re-run:  python train.py --dataset dataset_split"
        )

    train_gen, val_gen, _ = get_data_generators(args.dataset)

    train_cnn(train_gen, val_gen)
    train_mobilenet(train_gen, val_gen)

    print("\n✓ Both models trained and saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
