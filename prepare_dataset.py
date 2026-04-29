"""
prepare_dataset.py
------------------
Reorganises the original 4-class Kaggle dataset into a proper 3-way
train / val / test split under a binary alert / drowsy structure.

Original structure (what you downloaded):
    dataset/
        train/
            Closed/    <- eyes closed          -> DROWSY
            Open/      <- eyes open             -> ALERT
            no_yawn/   <- no yawning detected   -> ALERT
            yawn/      <- yawning detected      -> DROWSY

Output structure (created by this script):
    dataset_split/
        train/
            alert/     <- 70% of (Open + no_yawn)
            drowsy/    <- 70% of (Closed + yawn)
        val/
            alert/     <- 15% of (Open + no_yawn)
            drowsy/    <- 15% of (Closed + yawn)
        test/
            alert/     <- 15% of (Open + no_yawn)  [NEVER seen during training]
            drowsy/    <- 15% of (Closed + yawn)   [NEVER seen during training]

The 70/15/15 split is done per-class before copying so the class balance
is preserved in every split. Images are shuffled with a fixed seed (42)
for full reproducibility.

Usage:
    python prepare_dataset.py --src dataset --out dataset_split

After running, pass  --dataset dataset_split  to train.py and evaluate.py.
"""

import random
import shutil
import argparse
from pathlib import Path

# ── Label mapping ──────────────────────────────────────────────────────────────
CLASS_MAP = {
    "Closed"  : "drowsy",
    "yawn"    : "drowsy",
    "Open"    : "alert",
    "no_yawn" : "alert",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_RATIOS     = (0.70, 0.15, 0.15)   # train / val / test
SEED             = 42


def _collect_images(src_train: Path) -> dict:
    """Collect all valid image paths per binary label."""
    collected = {"alert": [], "drowsy": []}
    for original_class, binary_label in CLASS_MAP.items():
        src_dir = src_train / original_class
        if not src_dir.exists():
            print(f"  [WARNING] Folder not found, skipping: {src_dir}")
            continue
        imgs = [f for f in src_dir.glob("*") if f.suffix.lower() in VALID_EXTENSIONS]
        print(f"  {original_class:10s} -> {binary_label:6s}  |  {len(imgs)} images")
        # Prefix filename with original class to avoid name collisions
        for img in imgs:
            collected[binary_label].append((img, f"{original_class}_{img.name}"))
    return collected


def _split(items, ratios, seed):
    """Shuffle and split a list into (train, val, test) by ratio."""
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n       = len(shuffled)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def prepare(src_root: str, out_root: str):
    src_train = Path(src_root) / "train"
    if not src_train.exists():
        src_train = Path(src_root)
    if not src_train.exists():
        raise FileNotFoundError(f"Source folder not found: {src_root}")

    out_path = Path(out_root)

    # Create all output directories upfront
    for split in ("train", "val", "test"):
        for label in ("alert", "drowsy"):
            (out_path / split / label).mkdir(parents=True, exist_ok=True)

    collected = _collect_images(src_train)
    total_counts = {s: {"alert": 0, "drowsy": 0} for s in ("train", "val", "test")}

    for label, items in collected.items():
        train_items, val_items, test_items = _split(items, SPLIT_RATIOS, SEED)
        for split_name, bucket in (("train", train_items), ("val", val_items), ("test", test_items)):
            dst_dir = out_path / split_name / label
            for src_img, new_name in bucket:
                shutil.copy2(src_img, dst_dir / new_name)
            total_counts[split_name][label] += len(bucket)

    print("\n✓ Done! Split summary:")
    print(f"  {'Split':<8}  {'alert':>6}  {'drowsy':>7}  {'total':>6}")
    print(f"  {'-'*35}")
    for split in ("train", "val", "test"):
        a = total_counts[split]["alert"]
        d = total_counts[split]["drowsy"]
        print(f"  {split:<8}  {a:>6}  {d:>7}  {a+d:>6}")
    print(f"\n  Output folder: {out_path}/")
    print(f"\n  Now run:  python train.py --dataset {out_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare 3-way train/val/test split from 4-class Kaggle data"
    )
    parser.add_argument(
        "--src", type=str, default="dataset",
        help="Root folder of the downloaded dataset (contains the train/ sub-folder)",
    )
    parser.add_argument(
        "--out", type=str, default="dataset_split",
        help="Output folder for the split dataset",
    )
    args = parser.parse_args()
    prepare(args.src, args.out)


if __name__ == "__main__":
    main()
