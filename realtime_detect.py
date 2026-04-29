"""
realtime_detect.py
------------------
Real-time driver drowsiness detection using a webcam and a trained model.

PREREQUISITE — Complete these steps before running:
    1. python prepare_dataset.py --src dataset --out dataset_split
    2. python train.py --dataset dataset_split
    3. python evaluate.py --dataset dataset_split  (optional)

Usage:
    python realtime_detect.py --model saved_models/mobilenet_best.keras
    python realtime_detect.py --model saved_models/cnn_best.keras

How it works:
    1. OpenCV Haar Cascade detects the driver's face in each frame
    2. Face crop is resized to 224x224 and normalised to [0, 1]
    3. Model predicts a sigmoid score  (0 = alert, 1 = drowsy)
    4. Score >= DROWSY_THRESHOLD counts as one drowsy frame
    5. Alarm.update() is called every frame with the consecutive-drowsy count
    6. alarm.py escalates through 3 levels based on that count:
         Level 1 WARNING  (>=15 frames): yellow banner + soft beep every ~1.2 s
         Level 2 ALERT    (>=30 frames): orange border + rapid double-beep
         Level 3 CRITICAL (>=45 frames): red flash + continuous siren
    7. All audio runs in a background thread — the video loop is never blocked

Controls:
    Q  →  quit
    M  →  mute / unmute alarm
    S  →  save screenshot
"""

import os
import sys
import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from alarm import (
    AlarmSystem,
    LEVEL_NONE, LEVEL_WARNING, LEVEL_ALERT, LEVEL_CRITICAL,
    LEVEL_NAMES, LEVEL_COLORS, LEVEL_BANNERS,
    FRAMES_WARNING, FRAMES_ALERT, FRAMES_CRITICAL,
)

# ── Configuration ──────────────────────────────────────────────────────────────
IMG_SIZE           = (224, 224)
DROWSY_THRESHOLD   = 0.55
DISPLAY_WIDTH      = 900
HAAR_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ── Fixed colours ──────────────────────────────────────────────────────────────
COLOR_GREEN  = (50,  200, 50)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0,   0,   0)
COLOR_GREY   = (160, 160, 160)


# ── Image helpers ──────────────────────────────────────────────────────────────

def preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """Resize and normalise a BGR face crop for model input."""
    rgb        = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    resized    = cv2.resize(rgb, IMG_SIZE)
    normalised = resized.astype("float32") / 255.0
    return np.expand_dims(normalised, axis=0)


# ── HUD drawing ────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, state: str, confidence: float,
             consecutive: int, alarm: AlarmSystem, flash_on: bool) -> np.ndarray:
    """
    Draw a multi-layer HUD on the frame:
      - Top bar:   state + confidence + frame counter + mute indicator
      - Progress bar: shows consecutive frames vs each threshold
      - Border:    colour and thickness scales with alarm level
      - Bottom banner: alarm level label (flashes on CRITICAL)
    """
    h, w    = frame.shape[:2]
    level   = alarm.level
    l_color = LEVEL_COLORS[level]

    # ── 1. Screen flash on CRITICAL ────────────────────────────────────────────
    if level == LEVEL_CRITICAL and flash_on:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 30, 180), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    # ── 2. Semi-transparent top bar ────────────────────────────────────────────
    bar_h   = 75
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # State text
    state_color = l_color if state == "DROWSY" else COLOR_GREEN
    cv2.putText(frame,
                f"State: {state}  ({confidence * 100:.1f}%)",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.80, state_color, 2)

    # Frame counter
    cv2.putText(frame,
                f"Drowsy frames: {consecutive}  "
                f"[W:{FRAMES_WARNING}  A:{FRAMES_ALERT}  C:{FRAMES_CRITICAL}]",
                (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_GREY, 1)

    # Mute indicator (top-right)
    if alarm.is_muted:
        cv2.putText(frame, "[MUTED]",
                    (w - 105, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_GREY, 2)

    # ── 3. Alarm-level progress bar ────────────────────────────────────────────
    bar_y   = bar_h + 6
    bar_w   = w - 30
    bar_x   = 15
    bar_ht  = 10
    max_frames = FRAMES_CRITICAL + 15   # bar fills up a bit past CRITICAL

    fill_w = int(bar_w * min(consecutive, max_frames) / max_frames)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_ht), (50, 50, 50), -1)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_ht), l_color, -1)

    # Threshold tick marks
    for thresh, label in ((FRAMES_WARNING, "W"), (FRAMES_ALERT, "A"), (FRAMES_CRITICAL, "C")):
        tick_x = bar_x + int(bar_w * thresh / max_frames)
        cv2.line(frame, (tick_x, bar_y - 3), (tick_x, bar_y + bar_ht + 3), COLOR_WHITE, 1)
        cv2.putText(frame, label,
                    (tick_x - 5, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1)

    # ── 4. Border — thickness and colour scale with level ──────────────────────
    border_thickness = {LEVEL_NONE: 0, LEVEL_WARNING: 3, LEVEL_ALERT: 5, LEVEL_CRITICAL: 8}
    bt = border_thickness[level]
    if bt > 0:
        cv2.rectangle(frame, (bt, bt), (w - bt, h - bt), l_color, bt)

    # ── 5. Bottom alarm banner ─────────────────────────────────────────────────
    banner_text = LEVEL_BANNERS[level]
    if banner_text and (level < LEVEL_CRITICAL or flash_on):
        bh = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bh), (w, h), l_color, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, banner_text,
                    (10, h - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, COLOR_WHITE, 2)

    return frame


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time drowsiness detection with alarm")
    parser.add_argument(
        "--model", type=str,
        default=os.path.join("saved_models", "mobilenet_best.keras"),
        help="Path to trained .keras model file",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        sys.exit(f"Model file not found: {args.model}\nRun train.py first.")

    print(f"Loading model: {args.model} ...")
    model = load_model(args.model)
    print("Model loaded OK")

    # ── Face detector ──────────────────────────────────────────────────────────
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        sys.exit("Haar Cascade not found. Check your OpenCV installation.")

    # ── Webcam ─────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Alarm system ───────────────────────────────────────────────────────────
    alarm = AlarmSystem()
    alarm.start()

    # ── State ──────────────────────────────────────────────────────────────────
    consecutive_drowsy = 0
    screenshot_count   = 0
    frame_idx          = 0          # used for flash timing

    print("\nControls:  Q = quit  |  M = mute/unmute  |  S = screenshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        frame_idx += 1
        flash_on   = (frame_idx % 12) < 6     # ~2 Hz flash at 30fps (on for 6 frames, off for 6)

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        state      = "No Face"
        confidence = 0.0

        # Gradual cooldown when no face is visible
        if len(faces) == 0:
            consecutive_drowsy = max(0, consecutive_drowsy - 1)

        for (x, y, w, h) in faces:
            face_crop = frame[y: y + h, x: x + w]
            if face_crop.size == 0:
                continue

            inp        = preprocess_face(face_crop)
            confidence = float(model.predict(inp, verbose=0)[0][0])

            if confidence >= DROWSY_THRESHOLD:
                state = "DROWSY"
                consecutive_drowsy += 1
            else:
                state = "ALERT"
                consecutive_drowsy = max(0, consecutive_drowsy - 1)

            # Face bounding box — colour reflects current alarm level
            box_color = LEVEL_COLORS[alarm.level]
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, f"{state} {confidence * 100:.0f}%",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.60, box_color, 2)
            break   # only process the first (largest) detected face

        # ── Update alarm level ─────────────────────────────────────────────────
        alarm.update(consecutive_drowsy)

        # ── Draw HUD ───────────────────────────────────────────────────────────
        frame = draw_hud(frame, state, confidence, consecutive_drowsy, alarm, flash_on)

        # ── Display ────────────────────────────────────────────────────────────
        display_h = int(frame.shape[0] * DISPLAY_WIDTH / frame.shape[1])
        display   = cv2.resize(frame, (DISPLAY_WIDTH, display_h))
        cv2.imshow("Driver Drowsiness Detection  |  Q=quit  M=mute  S=screenshot", display)

        # ── Key handler ────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            alarm.toggle_mute()
            status = "MUTED" if alarm.is_muted else "UNMUTED"
            print(f"[Alarm] {status}")
        elif key == ord("s"):
            screenshot_count += 1
            fname = f"screenshot_{screenshot_count:03d}.png"
            cv2.imwrite(fname, frame)
            print(f"Screenshot saved: {fname}")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    alarm.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
