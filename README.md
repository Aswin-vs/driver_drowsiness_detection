# Driver Drowsiness Detection
### Deep Learning Microproject — Aswin V S (B23CS2121)
> Comparative Study of CNN from Scratch vs MobileNetV2 Transfer Learning

---

## Changes from Original Version

The following bugs were fixed and improvements were made:

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `prepare_dataset.py` + `data_loader.py` | **Test set = Validation set** (data leakage). Callbacks monitored the same split used for final metrics, so reported accuracy was not on truly unseen data. | `prepare_dataset.py` now writes 3 separate `train/` `val/` `test/` folders before any training begins. `data_loader.py` reads each folder independently. |
| 2 | `train.py` | **Orphan Phase 1 checkpoint.** `get_callbacks("mobilenet_phase1")` saved `mobilenet_phase1_best.keras` — a file nothing ever loads — wasting ~14 MB and causing confusion. | Phase 1 uses a separate callback list with no `ModelCheckpoint`. Only Phase 2 saves `mobilenet_best.keras`. |
| 3 | `mobilenet_model.py` | **Brittle hardcoded layer name.** `unfreeze_for_fine_tuning` searched for `"mobilenetv2_1.00_224"` by string. If TF named it differently the base was silently `None` and wrong layers were unfrozen. | Now searches by `isinstance(layer, tf.keras.Model)` + name contains `"mobilenetv2"`. Raises a clear error if not found. |
| 4 | `train.py` | **No global random seeds.** Weight initialisation varied between runs, making results non-reproducible. | Added `tf.random.set_seed(42)`, `np.random.seed(42)`, `random.seed(42)` at the top of `train.py`. |
| 5 | `realtime_detect.py` | **No-face counter freeze.** When the face detector lost the face, `consecutive_drowsy` stayed frozen, so looking away briefly and back could push it over the alert threshold. | When no face is detected the counter decrements by 1 each frame (gradual cooldown), consistent with the existing alert logic. |
| 6 | `realtime_detect.py` | **Unicode crash.** `cv2.putText` with `"⚠ DROWSINESS ALERT — PLEASE STOP!"` renders as garbage or crashes on some systems because OpenCV's built-in font is ASCII-only. | Replaced with ASCII: `"!! DROWSINESS ALERT - PLEASE STOP !!"` |
| 7 | `requirements.txt` | **Unpinned TensorFlow** installs TF 2.16+ on Python 3.12, which switches to Keras 3 and removes `ImageDataGenerator`, crashing `data_loader.py`. | Pinned to `tensorflow-cpu==2.15.0`. Setup instructions updated to use **Python 3.11**. |

---

## Dataset

This project uses the **Drowsiness Detection Dataset** from Kaggle:
- https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset

The dataset comes with a `train/` root folder containing **4 sub-folders** which `prepare_dataset.py` merges and splits into a proper 3-way train/val/test structure:

```
dataset/
└── train/
    ├── Closed/      726 images  ← eyes closed
    ├── Open/        726 images  ← eyes open
    ├── no_yawn/     725 images  ← no yawning
    └── yawn/        723 images  ← yawning detected
```

Since this project is a **binary classifier** (Alert vs Drowsy), we merge these
4 classes into 2 using `prepare_dataset.py` before training:

| Original folder | Meaning          | → Binary label |
|-----------------|------------------|----------------|
| `Open`          | Eyes open        | **alert**      |
| `no_yawn`       | No yawning       | **alert**      |
| `Closed`        | Eyes closed      | **drowsy**     |
| `yawn`          | Yawning detected | **drowsy**     |

After preparation the final training data is:
- **alert**  : ~1451 images  (Open + no_yawn)
- **drowsy** : ~1449 images  (Closed + yawn)

This gives a near-perfectly balanced dataset, so no class weighting is needed.

---

## Project Structure

```
drowsiness_detection/
│
├── dataset/                    ← Download and place Kaggle dataset here
│   └── train/
│       ├── Closed/
│       ├── Open/
│       ├── no_yawn/
│       └── yawn/
│
├── dataset_split/              ← Created by prepare_dataset.py (do not edit manually)
│   ├── train/
│   │   ├── alert/              ← 70%  ~1016 images
│   │   └── drowsy/             ← 70%  ~1014 images
│   ├── val/
│   │   ├── alert/              ← 15%  ~218 images
│   │   └── drowsy/             ← 15%  ~218 images
│   └── test/                   ← 15% held-out — NEVER seen during training
│       ├── alert/              ← ~218 images
│       └── drowsy/             ← ~217 images
│
├── saved_models/               ← Created automatically after training
│   ├── cnn_best.keras
│   ├── mobilenet_best.keras
│   ├── cnn_history.npy
│   └── mobilenet_history.npy
│
├── plots/                      ← Created automatically during evaluation
│   ├── cm_custom_cnn.png
│   ├── cm_mobilenetv2.png
│   ├── history_custom_cnn.png
│   ├── history_mobilenetv2.png
│   └── comparison_metrics.png
│
├── alarm.py                    ← Escalating audio alarm system (WARNING / ALERT / CRITICAL)
├── prepare_dataset.py          ← STEP 0: Converts 4-class → binary alert/drowsy
├── data_loader.py              ← Dataset loading, splitting & augmentation
├── cnn_model.py                ← Custom CNN architecture (baseline model)
├── mobilenet_model.py          ← MobileNetV2 transfer learning model
├── train.py                    ← Trains both models and saves best weights
├── evaluate.py                 ← Loads models, runs metrics, saves all plots
├── realtime_detect.py          ← Live webcam drowsiness detection with alert
├── requirements.txt            ← All Python dependencies
└── README.md
```

---

## System Requirements

| Component  | Minimum                        | Recommended                    |
|------------|--------------------------------|--------------------------------|
| Python     | 3.9                            | 3.12                           |
| RAM        | 8 GB                           | 16 GB                          |
| Storage    | 2 GB free                      | 5 GB free                      |
| GPU        | Not required                   | NVIDIA (CUDA) for faster train |
| OS         | Windows 10/11, Linux, macOS    | Windows 11 / Ubuntu 22.04      |

---

## GPU Support Notes

### AMD GPU (e.g. AMD Radeon Vega 8)

TensorFlow on Windows does **not** support AMD GPUs natively. You can enable
AMD GPU acceleration through Microsoft's **DirectML** plugin:

```powershell
pip uninstall tensorflow-cpu -y
pip install tensorflow-cpu==2.10.0
pip install tensorflow-directml-plugin
```

**Important caveat for AMD Vega 8 specifically:**
The Vega 8 is an integrated GPU with approximately 2 GB of shared system RAM.
For this project's dataset (~2900 images, 224×224), CPU training is often
equally fast or faster because:
- The Vega 8 shares RAM with the CPU (no dedicated VRAM)
- Memory bandwidth between CPU and integrated GPU is not a bottleneck advantage
- DirectML has overhead that can outweigh the GPU benefit on small datasets

For this microproject, **training on CPU is perfectly fine and recommended.**
The CNN will finish in 10–20 minutes and MobileNetV2 in 15–30 minutes on a
modern Ryzen CPU.

### NVIDIA GPU

```powershell
pip uninstall tensorflow-cpu -y
pip install tensorflow[and-cuda]
```
Requires CUDA 12 and cuDNN installed separately from the NVIDIA website.

### No GPU (default — what requirements.txt installs)

```powershell
pip install -r requirements.txt
```
Uses `tensorflow-cpu`. Works on all hardware with no additional setup.

---

## Setup

```powershell
# 1. Navigate into the project folder
cd "Driver_drowsiness_detection"

# 2. Create a virtual environment with Python 3.11 (NOT 3.12 — see note below)
py -3.11 -m venv venv

# 3. Activate it
venv\Scripts\Activate.ps1

# 4. Upgrade pip (use python -m pip to avoid PowerShell path issues with spaces)
python -m pip install --upgrade pip

# 5. Install all dependencies
python -m pip install -r requirements.txt

# 6. Verify installation
python -c "import tensorflow as tf; import cv2; print('TF:', tf.__version__); print('CV2:', cv2.__version__)"
```

> ⚠ **Important:** Use **Python 3.11**, not 3.12. `tensorflow-cpu==2.15.0` (pinned in
> `requirements.txt`) does not support Python 3.12. TF 2.16+ would be needed for
> Python 3.12, but that version switches to Keras 3 which removes `ImageDataGenerator`,
> breaking `data_loader.py`.

---

## How to Run — Step by Step

### Step 0 — Prepare the dataset  *(run once)*

The Kaggle dataset has 4 class folders inside `train/`. This script merges
them into a binary alert/drowsy structure and creates a proper **3-way
train/val/test split** so the test set is never seen during training.

```powershell
python prepare_dataset.py --src dataset --out dataset_split
```

Expected output:
```
  Closed     -> drowsy  |  726 images
  Open       -> alert   |  726 images
  no_yawn    -> alert   |  725 images
  yawn       -> drowsy  |  723 images

Done! Split summary:
  Split     alert   drowsy   total
  -----------------------------------
  train      1016     1014    2030
  val         218      218     436
  test        218      217     435

  Now run:  python train.py --dataset dataset_split
```

You only need to run this once. The `test/` split is held-out and is
**never touched during training** — callbacks only monitor the `val/` split.
This means `evaluate.py` results are on truly unseen data.

---

### Step 1 — Train both models

```powershell
python train.py --dataset dataset_split
```

This will:
- Load images from `dataset_split/train/`, `val/`, and `test/` sub-folders
- Apply augmentation (rotation, zoom, flip, brightness) to training set only
- Train the **Custom CNN** for up to 30 epochs with early stopping
- Train **MobileNetV2** in two phases:
  - **Phase 1** (10 epochs): head only, base frozen — no checkpoint saved (intermediate weights)
  - **Phase 2** (20 epochs): top 100 MobileNetV2 layers unfrozen at LR = 1e-5 — `mobilenet_best.keras` saved here
- All callbacks (`EarlyStopping`, `ModelCheckpoint`) monitor `val/` only — test set is untouched
- Save training history arrays to `saved_models/` for plotting

**Expected training time on CPU (Ryzen with Vega 8):**
| Model       | Approximate time |
|-------------|------------------|
| Custom CNN  | 10 – 20 minutes  |
| MobileNetV2 | 20 – 35 minutes  |

---

### Step 2 — Evaluate and compare

```powershell
python evaluate.py --dataset dataset_split
```

This will:
- Load `cnn_best.keras` and `mobilenet_best.keras` from `saved_models/`
- Run both models on the held-out test set
- Print full classification reports (accuracy, precision, recall, F1 per class)
- Measure and compare inference time per image in milliseconds
- Save confusion matrix heatmaps to `plots/`
- Save training history curves (loss & accuracy) to `plots/`
- Save a side-by-side metric bar chart to `plots/comparison_metrics.png`
- Print the final comparison table to terminal
- Save `comparison_results.csv` for your project report

---

### Step 3 — Real-time webcam detection

```powershell
# Recommended — MobileNetV2 gives better accuracy
python realtime_detect.py --model saved_models/mobilenet_best.keras

# Alternatively — CNN baseline model
python realtime_detect.py --model saved_models/cnn_best.keras
```

The webcam window shows a live HUD with:
- Face bounding box and confidence score, colour-coded to the current alarm level
- Consecutive drowsy frame counter with threshold tick marks (W / A / C)
- Alarm progress bar that fills and changes colour as the driver becomes drowsier
- Escalating visual alerts (border, banner, screen flash) matching the alarm level

#### Alarm levels

| Level | Consecutive frames | Visual | Audio |
|---|---|---|---|
| **WARNING** | 15–29 (~0.5 s) | Yellow border + yellow banner | Soft single beep every ~1.2 s |
| **ALERT** | 30–44 (~1.0 s) | Orange border + orange banner | Rapid double-beep every ~0.6 s |
| **CRITICAL** | 45+ (~1.5 s) | Red flashing border + screen flash | Continuous rising siren |

Audio requires `pygame` (installed via `requirements.txt`). Falls back to
`winsound` (Windows built-in) or a terminal bell if pygame is unavailable.

| Key | Action                   |
|-----|--------------------------|
| `Q` | Quit                     |
| `M` | Mute / unmute alarm      |
| `S` | Save screenshot          |

---

## Model Architectures

### Model 1 — Custom CNN  (Baseline)

Trained entirely from scratch on `dataset_split/`. Learns features like
eye shape, eyelid position, and facial tension directly from training images
without any prior knowledge.

| Layer Block    | Configuration                                                  |
|----------------|----------------------------------------------------------------|
| Input          | 224 × 224 × 3                                                  |
| Conv2D Block 1 | 32 filters, 3×3, BN + ReLU + MaxPool(2×2) + Dropout(0.25)    |
| Conv2D Block 2 | 64 filters, 3×3, BN + ReLU + MaxPool(2×2) + Dropout(0.25)    |
| Conv2D Block 3 | 128 filters, 3×3, BN + ReLU + MaxPool(2×2) + Dropout(0.25)   |
| Dense Head     | Flatten → FC(256, ReLU) → Dropout(0.5) → FC(1, Sigmoid)      |
| Regularisation | L2 (1e-4) on all Conv and Dense layers                        |
| Loss           | Binary Crossentropy                                            |
| Optimiser      | Adam, LR = 1e-3                                                |
| Total params   | ~25.8 million                                                  |

---

### Model 2 — MobileNetV2  (Transfer Learning)

Uses visual knowledge learned from 1.4 million ImageNet images. Designed
for mobile and real-time inference via depthwise separable convolutions.

| Component    | Configuration                                                    |
|--------------|------------------------------------------------------------------|
| Base model   | MobileNetV2 pre-trained on ImageNet, `include_top=False`         |
| Input        | 224 × 224 × 3  (MobileNetV2 requirement)                         |
| Phase 1      | Base frozen → train head only at LR = 1e-3 for 10 epochs        |
| Head         | GlobalAvgPool2D → BN → Dense(128, ReLU) → Dropout(0.4) → Sigmoid |
| Phase 2      | Top 100 base layers unfrozen, fine-tuned at LR = 1e-5 for 20 epochs |
| BatchNorm    | Kept frozen throughout fine-tuning                               |
| Loss         | Binary Crossentropy                                              |

---

## Data Split and Augmentation

`dataset_split/` is split into 3 separate subfolders by `prepare_dataset.py`
before training begins. The split is done per-class to preserve balance:

| Split      | Percentage | alert  | drowsy | total  |
|------------|------------|--------|--------|--------|
| Training   | 70%        | ~1016  | ~1014  | ~2030  |
| Validation | 15%        | ~218   | ~218   | ~436   |
| Test       | 15%        | ~218   | ~217   | ~435   |

The **test** split is written to a separate folder (`test/`) before any
training begins. It is **never loaded by `train.py`** — callbacks only
monitor the `val/` split. `evaluate.py` loads the `test/` split to produce
final metrics on truly unseen data.

Augmentation applied to training images only:

| Technique           | Setting        |
|---------------------|----------------|
| Rotation            | ±15°           |
| Zoom                | ±15%           |
| Width shift         | ±10%           |
| Height shift        | ±10%           |
| Horizontal flip     | Enabled        |
| Brightness range    | 0.8× to 1.2×  |
| Pixel normalisation | Divide by 255  |

---

## Evaluation Metrics

| Metric         | What it measures                                               |
|----------------|----------------------------------------------------------------|
| Accuracy       | Overall percentage of correctly classified images              |
| Precision      | Of all frames predicted as drowsy, how many truly were drowsy  |
| Recall         | Of all actual drowsy frames, how many were correctly detected  |
| F1-Score       | Harmonic mean of Precision and Recall — the primary metric     |
| Inference time | Milliseconds per image — critical for real-time viability      |

Recall is especially important: a missed drowsy detection (false negative)
is far more dangerous than a false alarm (false positive).

---

## Expected Results

| Metric             | Custom CNN  | MobileNetV2  |
|--------------------|-------------|--------------|
| Accuracy           | ~88 – 91%   | ~93 – 96%    |
| Precision          | ~87 – 91%   | ~92 – 96%    |
| Recall             | ~86 – 90%   | ~91 – 95%    |
| F1-Score           | ~87 – 90%   | ~92 – 95%    |
| Inference (ms/img) | ~5 – 10 ms  | ~8 – 15 ms   |

> MobileNetV2 is expected to outperform CNN on all metrics.
> Actual results depend on hardware and the exact random split.

---

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `No matching distribution found for tensorflow` | Python 3.12+ or wrong pip | Use `py -3.11 -m venv venv` — TF 2.15 requires Python 3.11 |
| `DLL load failed` | Missing Visual C++ Redistributable | Install from https://aka.ms/vs/17/release/vc_redist.x64.exe |
| `TBNotInstalledError` | TensorBoard missing | `python -m pip install tensorboard` |
| `Path not recognised` (PowerShell space error) | Space in folder name | Use `python -m pip` instead of `pip` |
| `GPU support not available on native Windows` | Expected warning | Harmless — training runs on CPU fine |

---

## Output Files Reference

| File / Folder                        | Contents                                           |
|--------------------------------------|----------------------------------------------------|
| `saved_models/cnn_best.keras`        | Best CNN weights saved by val_accuracy             |
| `saved_models/mobilenet_best.keras`  | Best MobileNetV2 weights saved by val_accuracy     |
| `saved_models/cnn_history.npy`       | Training history dict for CNN                      |
| `saved_models/mobilenet_history.npy` | Combined Phase 1 + Phase 2 history for MobileNetV2 |
| `plots/cm_custom_cnn.png`            | Confusion matrix heatmap for CNN                   |
| `plots/cm_mobilenetv2.png`           | Confusion matrix heatmap for MobileNetV2           |
| `plots/history_custom_cnn.png`       | Loss & accuracy training curves for CNN            |
| `plots/history_mobilenetv2.png`      | Loss & accuracy training curves for MobileNetV2    |
| `plots/comparison_metrics.png`       | Side-by-side bar chart of all four metrics         |
| `comparison_results.csv`             | Numeric results table ready for your report        |


