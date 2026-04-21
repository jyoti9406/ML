"""
app/webcam_demo.py
==================
Real-time emotion detection using a laptop/desktop webcam.

Runs completely offline (no Streamlit needed).
Displays:
  - Live video with face bounding box + emotion label
  - Confidence bar (ASCII in terminal, coloured box on frame)
  - Emotion trend (rolling 30-frame window)
  - Mental health alert if sustained negative emotion detected

Usage
-----
    python app/webcam_demo.py                      # use default config
    python app/webcam_demo.py --device 0 --fps 30  # specify camera

Dependencies
------------
    opencv-python, torch, numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch

# ── Add project root to path ──────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.multimodal_model import MultimodalEmotionModel
from src.data.face_pipeline import FaceDetector, frame_to_tensor
from src.data.dataset import simulate_data


# ── Constants ─────────────────────────────────────────────────
EMOTION_COLORS_BGR = {
    "anger":   (0,   0,   220),   # red
    "fear":    (0,   165, 220),   # orange
    "sadness": (200, 80,  0  ),   # blue
    "disgust": (130, 50,  180),   # purple
}
CHECKPOINT  = str(ROOT / "checkpoints/multimodal_best.pt")
TREND_LEN   = 30
ALERT_FRAC  = 0.7


def _load_model() -> MultimodalEmotionModel:
    model = MultimodalEmotionModel()
    if Path(CHECKPOINT).exists():
        ckpt = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        print(f"[INFO] Loaded checkpoint: {CHECKPOINT}")
    else:
        print("[WARN] No checkpoint found - using random weights.")
    model.eval()
    return model


def _get_dummy_eeg() -> torch.Tensor:
    """Return a single simulated EEG tensor."""
    X, _ = simulate_data(n_samples=1, seq_len=128, n_features=32, seed=int(time.time() * 1000) % 10000)
    return torch.tensor(X, dtype=torch.float32)


def _draw_prediction(
    frame:      np.ndarray,
    box:        tuple,
    emotion:    str,
    confidence: float,
    probs:      dict,
) -> np.ndarray:
    x, y, w, h = box
    color = EMOTION_COLORS_BGR.get(emotion, (180, 180, 180))

    # ── Bounding box ──────────────────────────────────────────
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # ── Label ─────────────────────────────────────────────────
    label_txt = f"{emotion.upper()}  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)
    cv2.putText(frame, label_txt,
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    # ── Mini probability bars ─────────────────────────────────
    bar_x, bar_y = 10, frame.shape[0] - 10 - 20 * len(probs)
    for i, (emo, prob) in enumerate(sorted(probs.items(), key=lambda v: -v[1])):
        emo_color = EMOTION_COLORS_BGR.get(emo, (150, 150, 150))
        bar_len   = int(prob * 120)
        by        = bar_y + i * 20
        cv2.rectangle(frame, (bar_x, by), (bar_x + 120, by + 14),
                      (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x, by), (bar_x + bar_len, by + 14),
                      emo_color, -1)
        cv2.putText(frame, f"{emo[:3]} {prob:.0%}",
                    (bar_x + 125, by + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    return frame


def _check_alert(
    emotion: str,
    history: deque,
) -> None:
    if len(history) >= 10:
        frac = history.count(emotion) / len(history)
        if frac >= ALERT_FRAC:
            print(
                f"\r[ALERT] Sustained '{emotion}' "
                f"in {frac:.0%} of last {len(history)} frames.           "
            )


def run(
    device_id:  int   = 0,
    target_fps: int   = 25,
    mirror:     bool  = True,
):
    """Main webcam loop."""
    model    = _load_model()
    detector = FaceDetector()

    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {device_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    history = deque(maxlen=TREND_LEN)
    frame_count = 0
    t_start = time.time()

    print("[INFO] Press  Q  to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        frame_count += 1
        crops = detector.detect_and_crop(frame)
        boxes = detector.detect_faces(frame)

        if crops and boxes:
            face_tensor = frame_to_tensor(crops[0], img_size=48)
            eeg_tensor  = _get_dummy_eeg()

            with torch.no_grad():
                result = model.predict(face_tensor, eeg_tensor)

            emotion    = result["label"]
            confidence = result["confidence"]
            probs      = result["probs"]
            history.append(emotion)

            frame = _draw_prediction(
                frame, boxes[0], emotion, confidence, probs
            )
            _check_alert(emotion, history)

        # ── FPS overlay ───────────────────────────────────────
        fps = frame_count / (time.time() - t_start + 1e-9)
        cv2.putText(frame, f"FPS {fps:.1f}",
                    (frame.shape[1] - 90, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ── Trend overlay ─────────────────────────────────────
        if history:
            top_emo, top_cnt = Counter(history).most_common(1)[0]
            trend_txt = f"Trend: {top_emo}  ({top_cnt}/{len(history)})"
            cv2.putText(frame, trend_txt, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

        cv2.imshow("Multimodal Emotion Recognition - press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Demo closed.")


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time webcam emotion demo")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index (default 0)")
    parser.add_argument("--fps",    type=int, default=25,
                        help="Target FPS")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Disable frame mirroring")
    args = parser.parse_args()

    run(device_id=args.device,
        target_fps=args.fps,
        mirror=not args.no_mirror)
