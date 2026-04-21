"""
src/evaluation/metrics.py
=========================
Evaluation metrics, confusion matrix, and per-class error analysis.

Functions
---------
compute_metrics        → accuracy, precision, recall, F1 (macro + weighted)
plot_confusion_matrix  → seaborn heatmap saved to results/
plot_training_curves   → loss / accuracy curves saved to results/
error_analysis         → per-class breakdown, worst-case examples
model_comparison       → bar chart comparing CNN vs LSTM vs Multimodal
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

EMOTION_LABELS = ["anger", "fear", "sadness", "disgust"]
RESULTS_DIR    = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Returns a dict of standard classification metrics.

    Keys
    ----
    accuracy, precision_macro, recall_macro, f1_macro,
    precision_weighted, recall_weighted, f1_weighted
    """
    return {
        "accuracy":           accuracy_score(y_true, y_pred),
        "precision_macro":    precision_score(y_true, y_pred, average="macro",
                                              zero_division=0),
        "recall_macro":       recall_score(y_true, y_pred, average="macro",
                                           zero_division=0),
        "f1_macro":           f1_score(y_true, y_pred, average="macro",
                                       zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted",
                                              zero_division=0),
        "recall_weighted":    recall_score(y_true, y_pred, average="weighted",
                                           zero_division=0),
        "f1_weighted":        f1_score(y_true, y_pred, average="weighted",
                                       zero_division=0),
    }


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
):
    labels = labels or EMOTION_LABELS
    print(classification_report(y_true, y_pred, target_names=labels,
                                zero_division=0))


# ─────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    labels:   Optional[List[str]] = None,
    title:    str = "Confusion Matrix",
    filename: str = "confusion_matrix.png",
    normalise: bool = True,
) -> np.ndarray:
    """
    Save a seaborn confusion-matrix heatmap.

    Returns
    -------
    cm : np.ndarray  raw confusion matrix
    """
    labels   = labels or EMOTION_LABELS
    cm       = confusion_matrix(y_true, y_pred)
    cm_disp  = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalise else cm
    fmt      = ".2f" if normalise else "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_disp, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()

    out_path = RESULTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved -> {out_path}")
    return cm


# ─────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────

def plot_training_curves(
    history:  Dict[str, List[float]],
    filename: str = "training_curves.png",
):
    """
    Plot loss and accuracy curves from a history dict.

    Expected keys: train_loss, val_loss, train_acc, val_acc
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train", color="#2563EB")
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="#DC2626")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train", color="#2563EB")
    ax2.plot(epochs, history["val_acc"],   label="Val",   color="#DC2626")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved -> {out_path}")


# ─────────────────────────────────────────────────────────────
# Model comparison bar chart
# ─────────────────────────────────────────────────────────────

def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    filename: str = "model_comparison.png",
):
    """
    Bar chart comparing multiple models.

    Parameters
    ----------
    results : dict
        {
          "CNN (Face)":       {"accuracy": 0.82, "f1_macro": 0.80},
          "LSTM (EEG)":       {"accuracy": 0.65, "f1_macro": 0.63},
          "Multimodal":       {"accuracy": 0.89, "f1_macro": 0.88},
        }
    """
    models  = list(results.keys())
    metrics = ["accuracy", "f1_macro"]
    x       = np.arange(len(models))
    width   = 0.35
    colors  = ["#2563EB", "#16A34A"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        vals = [results[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i * width, vals, width,
                      label=metric.replace("_", " ").title(),
                      color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Accuracy & Macro-F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = RESULTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Model comparison saved -> {out_path}")


# ─────────────────────────────────────────────────────────────
# Per-class error analysis
# ─────────────────────────────────────────────────────────────

def error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Per-class precision, recall, F1, and confusion breakdown.

    Returns dict keyed by class name.
    """
    labels    = labels or EMOTION_LABELS
    cm        = confusion_matrix(y_true, y_pred)
    per_class = {}

    for i, lbl in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)

        # Most confused with
        row       = cm[i].copy()
        row[i]    = -1
        confused_with_idx = int(row.argmax())
        confused_with     = labels[confused_with_idx] if confused_with_idx >= 0 else "none"

        per_class[lbl] = {
            "precision":    round(precision, 4),
            "recall":       round(recall, 4),
            "f1":           round(f1, 4),
            "support":      int(cm[i].sum()),
            "confused_with": confused_with,
        }

    # Print table
    print("\n-- Per-class Error Analysis ----------------------")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} "
          f"{'Support':>9} {'Confused With':>14}")
    print("-" * 63)
    for lbl, m in per_class.items():
        print(f"{lbl:<12} {m['precision']:>10.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['support']:>9} {m['confused_with']:>14}")
    print()

    return per_class
