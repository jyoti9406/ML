"""
src/data/face_pipeline.py
=========================
Facial image preprocessing pipeline.

Handles:
  - Loading FER2013 / CK+ directory structure
  - Grayscale → tensor normalisation
  - Data augmentation (train-time only)
  - PyTorch Dataset + DataLoader wrappers
  - OpenCV-based face detection for webcam frames
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── Label mapping: negative subset ────────────────────────────
NEG_LABEL_MAP = {
    "anger":   0,
    "angry":   0,
    "fear":    1,
    "sad":     2,
    "sadness": 2,
    "disgust": 3,
    "disgust": 3,
}

# ─────────────────────────────────────────────────────────────
# 1. Transform Factories
# ─────────────────────────────────────────────────────────────

def build_train_transform(img_size: int = 48) -> transforms.Compose:
    """Augmentation-heavy transform for training."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),                          # [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5]),    # [-1,1]
    ])


def build_eval_transform(img_size: int = 48) -> transforms.Compose:
    """Deterministic transform for validation/test/inference."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


# ─────────────────────────────────────────────────────────────
# 2. PyTorch Dataset
# ─────────────────────────────────────────────────────────────

class FaceEmotionDataset(Dataset):
    """
    Loads facial emotion images from a directory tree:

        root/
          anger/   img1.jpg  img2.png ...
          fear/    ...
          sadness/ ...
          disgust/ ...

    Compatible with FER2013 (train/test split folders) and CK+.

    Parameters
    ----------
    root      : str    Path to class-named subdirectory root.
    transform : Compose  torchvision transform to apply.
    label_map : dict   maps folder names → integer labels.
                       Only folders in label_map are loaded.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        label_map: Optional[dict] = None,
    ):
        self.root = Path(root)
        self.transform = transform or build_eval_transform()
        self.label_map = label_map or NEG_LABEL_MAP

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label_key = class_dir.name.lower()
            if label_key not in self.label_map:
                continue
            label = self.label_map[label_key]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root} matching label_map keys "
                f"{list(self.label_map.keys())}. "
                "Ensure the directory structure is root/<class_name>/<images>."
            )
        logger.info("FaceEmotionDataset: %d images loaded from %s",
                    len(self.samples), self.root)

    # ── Dataset protocol ──────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")     # force grayscale
        if self.transform:
            img = self.transform(img)
        return img, label

    # ── Utility ───────────────────────────────────────────────
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for imbalanced training."""
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=len(set(self.label_map.values())))
        weights = 1.0 / (counts + 1e-6)
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# 3. Webcam / Frame-Level Preprocessing
# ─────────────────────────────────────────────────────────────

class FaceDetector:
    """
    Wraps OpenCV Haar-cascade face detection for real-time use.

    Usage
    -----
    >>> detector = FaceDetector()
    >>> frame = cv2.imread("frame.jpg")
    >>> faces = detector.detect_and_crop(frame)   # list of PIL Images
    """

    _CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ):
        self.cascade = cv2.CascadeClassifier(self._CASCADE_PATH)
        if self.cascade.empty():
            raise RuntimeError(
                "Could not load Haar cascade. Verify opencv-python is installed."
            )
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect_faces(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect face bounding boxes in a BGR frame.

        Returns
        -------
        List of (x, y, w, h) tuples.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        return [(int(x), int(y), int(w), int(h))
                for x, y, w, h in faces] if len(faces) > 0 else []

    def detect_and_crop(self, frame: np.ndarray) -> List[Image.Image]:
        """
        Detect faces and return a list of cropped PIL Images (grayscale).
        """
        crops = []
        for (x, y, w, h) in self.detect_faces(frame):
            roi = frame[y: y + h, x: x + w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            crops.append(Image.fromarray(gray_roi))
        return crops

    def draw_boxes(
        self,
        frame: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Draw detection boxes (and optional labels) on a copy of the frame.
        """
        out = frame.copy()
        boxes = self.detect_faces(frame)
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 80), 2)
            if labels and i < len(labels):
                cv2.putText(
                    out, labels[i],
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 80), 2,
                )
        return out


# ─────────────────────────────────────────────────────────────
# 4. Helper: frame → model-ready tensor
# ─────────────────────────────────────────────────────────────

def frame_to_tensor(
    pil_img: Image.Image,
    img_size: int = 48,
) -> torch.Tensor:
    """
    Convert a PIL Image (grayscale face crop) to a normalised tensor
    ready for the CNN inference.

    Returns
    -------
    tensor : torch.Tensor  shape (1, 1, img_size, img_size)
    """
    tfm = build_eval_transform(img_size)
    t = tfm(pil_img)          # (1, H, W)
    return t.unsqueeze(0)     # (1, 1, H, W)
