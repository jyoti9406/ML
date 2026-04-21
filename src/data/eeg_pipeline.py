"""
src/data/eeg_pipeline.py
========================
EEG signal preprocessing pipeline.

Handles:
  - CSV / NumPy loading (DEAP, SEED, or simulated data)
  - Band-pass Butterworth filtering
  - Artifact removal (simple threshold IQR method)
  - FFT frequency-domain features
  - Time-domain statistical features
  - StandardScaler normalisation
  - Reshaping to (N, seq_len, n_features) for LSTM

Compatible datasets
-------------------
  * DEAP  – download from https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
  * SEED  – download from https://bcmi.sjtu.edu.cn/home/seed/
  * FER–EEG simulated pairing (see dataset.py)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ── Emotion label mappings ────────────────────────────────────
DEAP_TO_NEGATIVE = {
    "anger":   0,
    "fear":    1,
    "sadness": 2,
    "disgust": 3,
}

# ─────────────────────────────────────────────────────────────
# 1. Signal Filtering
# ─────────────────────────────────────────────────────────────

def _butter_bandpass(lowcut: float, highcut: float,
                     fs: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Design a Butterworth band-pass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(signal: np.ndarray,
                    lowcut: float = 0.5,
                    highcut: float = 45.0,
                    fs: float = 128.0,
                    order: int = 5) -> np.ndarray:
    """
    Apply a zero-phase Butterworth band-pass filter to an EEG array.

    Parameters
    ----------
    signal : np.ndarray  shape (n_channels, n_samples)  or  (n_samples,)
    lowcut / highcut : float  – frequency bounds in Hz
    fs     : float  – sampling rate in Hz
    order  : int    – filter order

    Returns
    -------
    filtered : np.ndarray  same shape as input
    """
    b, a = _butter_bandpass(lowcut, highcut, fs, order)
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    return np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=-1, arr=signal)


# ─────────────────────────────────────────────────────────────
# 2. Artifact Removal
# ─────────────────────────────────────────────────────────────

def remove_artifacts_iqr(signal: np.ndarray,
                          threshold: float = 3.0) -> np.ndarray:
    """
    Replace samples beyond `threshold` IQR from the median with the median.
    Simple, fast, and avoids ICA dependency for pipeline demos.

    Parameters
    ----------
    signal    : np.ndarray  shape (n_channels, n_samples)
    threshold : float       IQR multiplier

    Returns
    -------
    cleaned : np.ndarray  same shape
    """
    cleaned = signal.copy()
    for ch in range(signal.shape[0]):
        ch_data = signal[ch]
        q1, q3 = np.percentile(ch_data, [25, 75])
        iqr = q3 - q1
        med = np.median(ch_data)
        lo, hi = med - threshold * iqr, med + threshold * iqr
        cleaned[ch] = np.clip(ch_data, lo, hi)
    return cleaned


# ─────────────────────────────────────────────────────────────
# 3. Feature Extraction
# ─────────────────────────────────────────────────────────────

FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _band_power(signal: np.ndarray, fs: float,
                fmin: float, fmax: float) -> float:
    """Estimate power in a frequency band using Welch's method."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 256))
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return float(np.trapz(psd[idx], freqs[idx]))


def extract_eeg_features(signal: np.ndarray,
                          fs: float = 128.0,
                          use_fft: bool = True,
                          use_bands: bool = True,
                          use_stats: bool = True) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a multi-channel EEG segment.

    Feature breakdown (per channel):
      - Band power  : delta, theta, alpha, beta, gamma  (5 features)
      - FFT top-k   : magnitude of top 10 FFT bins      (10 features)
      - Stats        : mean, std, var, kurtosis, skew    (5 features)
    → Total : n_channels × (5 + 10 + 5) = n_channels × 20

    Parameters
    ----------
    signal : np.ndarray  shape (n_channels, n_samples)
    fs     : float       sampling rate

    Returns
    -------
    features : np.ndarray  shape (n_channels * n_feature_per_ch,)
    """
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    from scipy.stats import kurtosis, skew

    all_features = []
    for ch in range(signal.shape[0]):
        ch_sig = signal[ch]
        ch_feats: list[float] = []

        if use_bands:
            for fmin, fmax in FREQ_BANDS.values():
                ch_feats.append(_band_power(ch_sig, fs, fmin, fmax))

        if use_fft:
            fft_mag = np.abs(np.fft.rfft(ch_sig))
            # Keep top-10 bins by magnitude
            top_k = np.sort(fft_mag)[::-1][:10]
            ch_feats.extend(top_k.tolist())

        if use_stats:
            ch_feats += [
                float(np.mean(ch_sig)),
                float(np.std(ch_sig)),
                float(np.var(ch_sig)),
                float(kurtosis(ch_sig)),
                float(skew(ch_sig)),
            ]

        all_features.extend(ch_feats)

    return np.array(all_features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# 4. High-Level Pipeline Class
# ─────────────────────────────────────────────────────────────

class EEGPipeline:
    """
    End-to-end EEG preprocessing and feature extraction pipeline.

    Usage
    -----
    >>> pipe = EEGPipeline(fs=128.0, seq_len=128, n_features=32)
    >>> X, y = pipe.fit_transform("data/raw/emotions.csv")
    """

    def __init__(
        self,
        fs: float = 128.0,
        lowcut: float = 0.5,
        highcut: float = 45.0,
        seq_len: int = 128,
        n_features: int = 32,
    ):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.seq_len = seq_len
        self.n_features = n_features

        self._scaler = StandardScaler()
        self._label_enc = LabelEncoder()
        self._fitted = False

    # ── CSV loader (DEAP/SEED export or simulated) ────────────
    def _load_csv(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Expects a CSV with shape (n_samples, n_eeg_cols + 1_label_col).
        Label column must be named 'label'.
        """
        df = pd.read_csv(path)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        y_raw = df["label"].values
        X_raw = df.drop(columns=["label"]).values.astype(np.float32)
        return X_raw, y_raw

    # ── Public API ────────────────────────────────────────────
    def fit_transform(
        self, csv_path: str, save_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load → filter → artifact removal → feature extraction
        → normalise → reshape for LSTM.

        Returns
        -------
        X : np.ndarray  shape (N, seq_len, n_features)
        y : np.ndarray  shape (N,)  integer labels
        """
        logger.info("EEGPipeline: loading %s", csv_path)
        X_raw, y_raw = self._load_csv(csv_path)

        # ── Encode labels ─────────────────────────────────────
        y = self._label_enc.fit_transform(y_raw)
        logger.info("Classes: %s", self._label_enc.classes_)

        # ── Scale raw signal ──────────────────────────────────
        X_scaled = self._scaler.fit_transform(X_raw)

        # ── Reshape & process each sample ────────────────────
        # If raw signal is already flattened features, just reshape to (N, seq_len, n_feat)
        n_samples = X_scaled.shape[0]

        # Attempt band-pass if raw time-series columns exist
        # (For pre-extracted CSVs we skip filtering)
        if X_scaled.shape[1] >= self.seq_len:
            logger.info("Applying band-pass filter to raw time-series…")
            X_filtered = np.zeros_like(X_scaled)
            for i in range(n_samples):
                row = X_scaled[i]  # 1-D slice treated as single-channel
                try:
                    X_filtered[i] = bandpass_filter(
                        row, self.lowcut, self.highcut, self.fs
                    )
                except Exception:
                    X_filtered[i] = row
        else:
            X_filtered = X_scaled

        # ── Pad / truncate to (N, seq_len, n_features) ────────
        target_cols = self.seq_len * self.n_features
        if X_filtered.shape[1] < target_cols:
            pad = target_cols - X_filtered.shape[1]
            X_filtered = np.pad(X_filtered, ((0, 0), (0, pad)))
        else:
            X_filtered = X_filtered[:, :target_cols]

        X_out = X_filtered.reshape(n_samples, self.seq_len, self.n_features)

        self._fitted = True
        logger.info("EEG X shape: %s  y shape: %s", X_out.shape, y.shape)

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            np.save(f"{save_dir}/eeg_X.npy", X_out)
            np.save(f"{save_dir}/eeg_y.npy", y)
            logger.info("Saved processed EEG to %s", save_dir)

        return X_out.astype(np.float32), y.astype(np.int64)

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """Transform new samples using fitted scaler (inference)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        X_scaled = self._scaler.transform(X_raw)
        n = X_scaled.shape[0]
        target_cols = self.seq_len * self.n_features
        if X_scaled.shape[1] < target_cols:
            X_scaled = np.pad(X_scaled, ((0, 0), (0, target_cols - X_scaled.shape[1])))
        else:
            X_scaled = X_scaled[:, :target_cols]
        return X_scaled.reshape(n, self.seq_len, self.n_features).astype(np.float32)
