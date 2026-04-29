"""
evaluate.py
-----------
Loads both saved models, runs them on the test set, and produces a full
comparative analysis including metrics, plots, and a summary CSV.

PREREQUISITE — Complete these steps before running evaluate.py:
    1. python prepare_dataset.py --src dataset --out dataset_split
       (merges Kaggle 4-class folders into binary alert/drowsy structure)
    2. python train.py --dataset dataset_split
       (trains both models and saves weights to saved_models/)

Usage:
    python evaluate.py --dataset dataset_split

Output produced:
    plots/cm_custom_cnn.png          ← Confusion matrix for CNN
    plots/cm_mobilenetv2.png         ← Confusion matrix for MobileNetV2
    plots/history_custom_cnn.png     ← Loss & accuracy curves for CNN
    plots/history_mobilenetv2.png    ← Loss & accuracy curves for MobileNetV2
    plots/comparison_metrics.png     ← Side-by-side bar chart of all metrics
    comparison_results.csv           ← Numeric table for your project report

Metrics compared:
    Accuracy, Precision, Recall, F1-Score, Inference time (ms per image)

Note on Recall:
    In a safety-critical system like drowsiness detection, Recall is the most
    important individual metric. Missing a drowsy driver (false negative) is
    far more dangerous than a false alarm (false positive). Both are tracked
    and F1-Score is used as the primary balanced comparison metric.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)

from data_loader import get_data_generators

SAVED_MODELS_DIR = "saved_models"
PLOTS_DIR        = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

CLASS_NAMES = ["Alert", "Drowsy"]


# ── Helper: predict with timing ────────────────────────────────────────────────
def predict_with_timing(model, generator):
    """
    Run inference on all batches in generator.
    Returns (y_true, y_pred_binary, inference_time_per_image_ms).
    """
    generator.reset()
    y_true, y_pred_prob = [], []

    start = time.perf_counter()
    for i in range(len(generator)):
        X_batch, y_batch = generator[i]
        preds = model.predict(X_batch, verbose=0)
        y_pred_prob.extend(preds.flatten().tolist())
        y_true.extend(y_batch.flatten().tolist())
    elapsed_ms = (time.perf_counter() - start) * 1000

    y_true        = np.array(y_true[:generator.samples])
    y_pred_binary = (np.array(y_pred_prob[:generator.samples]) >= 0.5).astype(int)
    ms_per_image  = elapsed_ms / generator.samples

    return y_true, y_pred_binary, ms_per_image


# ── Helper: compute metrics ────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy" : round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "Recall"   : round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "F1-Score" : round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
    }


# ── Plot: confusion matrix ─────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"cm_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {path}")


# ── Plot: training history ─────────────────────────────────────────────────────
def plot_history(history_dict, model_name):
    epochs = range(1, len(history_dict["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(epochs, history_dict["accuracy"],     label="Train Acc",  color="steelblue")
    axes[0].plot(epochs, history_dict["val_accuracy"], label="Val Acc",    color="coral",   linestyle="--")
    axes[0].set_title(f"{model_name} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs, history_dict["loss"],     label="Train Loss", color="steelblue")
    axes[1].plot(epochs, history_dict["val_loss"], label="Val Loss",   color="coral",   linestyle="--")
    axes[1].set_title(f"{model_name} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"history_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved history plot     → {path}")


# ── Plot: side-by-side metric bar chart ───────────────────────────────────────
def plot_comparison(results: dict):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models  = list(results.keys())
    x       = np.arange(len(metrics))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, [results[models[0]][m] for m in metrics], width, label=models[0], color="steelblue")
    bars2 = ax.bar(x + width / 2, [results[models[1]][m] for m in metrics], width, label=models[1], color="coral")

    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — Test Set Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "comparison_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved comparison chart → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare trained models")
    parser.add_argument("--dataset", type=str, default="dataset_split")
    args = parser.parse_args()

    _, _, test_gen = get_data_generators(args.dataset)

    results      = {}
    timing       = {}
    model_paths  = {
        "Custom CNN"       : os.path.join(SAVED_MODELS_DIR, "cnn_best.keras"),
        "MobileNetV2"      : os.path.join(SAVED_MODELS_DIR, "mobilenet_best.keras"),
    }
    history_paths = {
        "Custom CNN"       : os.path.join(SAVED_MODELS_DIR, "cnn_history.npy"),
        "MobileNetV2"      : os.path.join(SAVED_MODELS_DIR, "mobilenet_history.npy"),
    }

    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"⚠  Skipping {model_name}: model file not found at {model_path}")
            continue

        print(f"\n{'─'*55}")
        print(f"  Evaluating: {model_name}")
        print(f"{'─'*55}")

        model = load_model(model_path)
        y_true, y_pred, ms_per_img = predict_with_timing(model, test_gen)

        metrics = compute_metrics(y_true, y_pred)
        results[model_name] = metrics
        timing[model_name]  = round(ms_per_img, 3)

        print(f"\n  Classification Report:\n")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
        print(f"  Inference time per image: {ms_per_img:.3f} ms")

        plot_confusion_matrix(y_true, y_pred, model_name)

        # Plot training history if available
        h_path = history_paths.get(model_name)
        if h_path and os.path.exists(h_path):
            history = np.load(h_path, allow_pickle=True).item()
            plot_history(history, model_name)

    # ── Comparison Table ───────────────────────────────────────────────────────
    if len(results) == 2:
        plot_comparison(results)

        rows = []
        for model_name, metrics in results.items():
            row = {"Model": model_name, **metrics, "Inference (ms/img)": timing.get(model_name, "N/A")}
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Model")
        print("\n" + "=" * 60)
        print("  Final Comparison Table")
        print("=" * 60)
        print(df.to_string())

        csv_path = "comparison_results.csv"
        df.to_csv(csv_path)
        print(f"\n  Saved results table → {csv_path}")

        # Declare winner
        best = df["F1-Score"].idxmax()
        print(f"\n✓ Best model by F1-Score: {best}")


if __name__ == "__main__":
    main()
