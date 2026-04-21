"""
src/data/dataset.py
===================
Paired EEG + Face dataset for multimodal training.

Strategy for combining DEAP (EEG) with FER2013 (faces)
-------------------------------------------------------
Both datasets carry emotion labels but are NOT synchronised
(they come from different participants).

Pairing approach used here:
  1. Stratified label-matching  – for every (eeg_sample, label),
     randomly select a face image with the SAME label.
  2. This creates aligned (eeg, face, label) triplets while
     preserving class distribution.
  3. During real deployment, swap in a truly synchronised dataset
     (e.g. MAHNOB-HCI, or a custom capture session).

This is the standard simulation strategy used in literature
when a single fused dataset is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .face_pipeline import FaceEmotionDataset, build_eval_transform, build_train_transform

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Multimodal Dataset
# ─────────────────────────────────────────────────────────────

class MultimodalEmotionDataset(Dataset):
    """
    Returns aligned (eeg_tensor, face_tensor, label) triplets.

    Parameters
    ----------
    eeg_X       : np.ndarray  shape (N, seq_len, n_feat)  – float32
    eeg_y       : np.ndarray  shape (N,)                  – int64 labels
    face_root   : str         root dir with per-class subdirs
    face_transform : Compose  torchvision transform
    seed        : int         for reproducible pairing
    """

    def __init__(
        self,
        eeg_X: np.ndarray,
        eeg_y: np.ndarray,
        face_root: str,
        face_transform: Optional[transforms.Compose] = None,
        seed: int = 42,
    ):
        assert len(eeg_X) == len(eeg_y), "EEG X and y must have equal length"

        self.eeg_X = torch.tensor(eeg_X, dtype=torch.float32)
        self.eeg_y = torch.tensor(eeg_y, dtype=torch.long)

        self.face_ds = FaceEmotionDataset(
            root=face_root,
            transform=face_transform or build_eval_transform(),
        )

        # Build per-label index lists for face samples
        self._face_by_label: Dict[int, List[int]] = {}
        for idx, (_, lbl) in enumerate(self.face_ds.samples):
            self._face_by_label.setdefault(lbl, []).append(idx)

        # Pre-assign a face index to every EEG sample (deterministic)
        rng = np.random.default_rng(seed)
        self._face_indices: List[int] = []
        for lbl in eeg_y:
            candidates = self._face_by_label.get(int(lbl), [])
            if not candidates:
                # Fallback: any face sample
                self._face_indices.append(rng.integers(len(self.face_ds)))
            else:
                self._face_indices.append(int(rng.choice(candidates)))

        logger.info(
            "MultimodalDataset: %d triplets | EEG shape %s | face classes %s",
            len(self.eeg_X),
            tuple(self.eeg_X.shape),
            list(self._face_by_label.keys()),
        )

    def __len__(self) -> int:
        return len(self.eeg_X)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eeg = self.eeg_X[idx]                           # (seq_len, n_feat)
        label = self.eeg_y[idx]                         # scalar

        face_img, _ = self.face_ds[self._face_indices[idx]]  # (C, H, W)

        return eeg, face_img, label

    # ── Class weights for imbalanced sampling ─────────────────
    def make_sampler(self) -> WeightedRandomSampler:
        labels = self.eeg_y.numpy()
        class_counts = np.bincount(labels)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(self),
            replacement=True,
        )


# ─────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────

def build_dataloaders(
    eeg_X: np.ndarray,
    eeg_y: np.ndarray,
    face_root: str,
    img_size: int = 48,
    batch_size: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split arrays into train/val/test, build MultimodalEmotionDataset
    for each split, return DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    from sklearn.model_selection import train_test_split

    n = len(eeg_X)
    idx = np.arange(n)

    # ── train / (val+test) ────────────────────────────────────
    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        idx, eeg_y,
        test_size=test_size + val_size,
        stratify=eeg_y,
        random_state=seed,
    )
    # ── val / test ────────────────────────────────────────────
    val_frac = val_size / (test_size + val_size)
    idx_val, idx_test, _, _ = train_test_split(
        idx_tmp, y_tmp,
        test_size=1.0 - val_frac,
        stratify=y_tmp,
        random_state=seed,
    )

    train_tfm = build_train_transform(img_size)
    eval_tfm = build_eval_transform(img_size)

    train_ds = MultimodalEmotionDataset(
        eeg_X[idx_train], eeg_y[idx_train], face_root, train_tfm, seed
    )
    val_ds = MultimodalEmotionDataset(
        eeg_X[idx_val], eeg_y[idx_val], face_root, eval_tfm, seed
    )
    test_ds = MultimodalEmotionDataset(
        eeg_X[idx_test], eeg_y[idx_test], face_root, eval_tfm, seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_ds.make_sampler(),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    logger.info("Splits → train: %d | val: %d | test: %d",
                len(train_ds), len(val_ds), len(test_ds))

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────
# Simulation helper (no real dataset required for smoke tests)
# ─────────────────────────────────────────────────────────────

def simulate_data(
    n_samples: int = 500,
    seq_len: int = 128,
    n_features: int = 32,
    num_classes: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random EEG-like data for unit tests / smoke runs.

    Returns
    -------
    X : float32  (n_samples, seq_len, n_features)
    y : int64    (n_samples,)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, seq_len, n_features)).astype(np.float32)
    y = rng.integers(0, num_classes, size=n_samples).astype(np.int64)
    return X, y
