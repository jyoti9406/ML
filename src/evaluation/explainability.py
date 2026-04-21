"""
src/evaluation/explainability.py
=================================
Explainability tools for the multimodal emotion model.

1. Grad-CAM  – heatmap on face images (which pixels drove the CNN decision)
2. EEG Attention Map  – plot temporal attention weights over time-steps
3. SHAP (DeepExplainer) – feature importance on the fused embedding

Usage
-----
    from src.evaluation.explainability import GradCAMVisualizer, plot_eeg_attention

    viz = GradCAMVisualizer(model.face_encoder)
    heatmap = viz.generate(face_tensor, class_idx=0)
    viz.overlay(original_img, heatmap, save_path="results/gradcam.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. Grad-CAM
# ─────────────────────────────────────────────────────────────

class GradCAMVisualizer:
    """
    Grad-CAM for a CNN encoder.

    Registers forward/backward hooks on the target convolutional layer,
    computes class-activation maps, and overlays them on the input image.

    Parameters
    ----------
    cnn_encoder : CNNFaceEncoder  (or any module with .get_gradcam_layer())
    """

    def __init__(self, cnn_encoder):
        self.encoder      = cnn_encoder
        self._activations = None
        self._gradients   = None

        target_layer = cnn_encoder.get_gradcam_layer()
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_backward_hook(self._save_gradients)

    # ── Hooks ─────────────────────────────────────────────────
    def _save_activations(self, module, input, output):
        self._activations = output.detach()          # (B, C, H, W)

    def _save_gradients(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()       # (B, C, H, W)

    # ── Generate CAM ──────────────────────────────────────────
    def generate(
        self,
        face_tensor: torch.Tensor,
        class_idx:   int,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Parameters
        ----------
        face_tensor : Tensor  (1, C, H, W)
        class_idx   : int     target class for back-propagation

        Returns
        -------
        cam : np.ndarray  shape (H, W)  normalised to [0, 1]
        """
        face_tensor = face_tensor.requires_grad_(True)
        emb         = self.encoder(face_tensor)      # (1, emb_dim)

        # Create one-hot score for target class
        score = emb[0, class_idx % emb.shape[1]]
        self.encoder.zero_grad()
        score.backward()

        # Global average pooling of gradients
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)  # (1,1,H,W)
        cam     = F.relu(cam)
        cam     = F.interpolate(
            cam,
            size=face_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0].cpu().numpy()

        # Normalise
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    # ── Overlay on image ──────────────────────────────────────
    def overlay(
        self,
        original: np.ndarray,   # (H, W) or (H, W, C)  uint8 or float [0,1]
        cam:      np.ndarray,   # (H, W) float [0,1]
        save_path: Optional[str] = None,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original face image.

        Returns the composite as uint8 numpy array (H, W, 3).
        """
        import cv2

        if original.dtype != np.uint8:
            original = (original * 255).clip(0, 255).astype(np.uint8)

        if original.ndim == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

        heatmap   = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        composite = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)

        if save_path:
            cv2.imwrite(str(save_path), composite)
            print(f"Grad-CAM saved → {save_path}")

        return composite


# ─────────────────────────────────────────────────────────────
# 2. EEG Temporal Attention Map
# ─────────────────────────────────────────────────────────────

def plot_eeg_attention(
    attn_weights: np.ndarray,
    fs:           float = 128.0,
    emotion_label: str  = "",
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """
    Plot the temporal attention weights as a bar chart over time.

    Parameters
    ----------
    attn_weights : np.ndarray  shape (T,)  summing to ~1
    fs           : float       sampling rate (for x-axis in seconds)
    emotion_label: str         shown in title

    Returns
    -------
    fig : plt.Figure
    """
    T       = len(attn_weights)
    times   = np.arange(T) / fs

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(times, attn_weights, width=1.0 / fs,
           color="#2563EB", alpha=0.7, edgecolor="none")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Attention Weight")
    ax.set_title(f"EEG Temporal Attention  [{emotion_label}]")
    ax.set_xlim(0, times[-1])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = save_path or str(RESULTS_DIR / "eeg_attention.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"EEG attention map saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────
# 3. SHAP Feature Importance
# ─────────────────────────────────────────────────────────────

def shap_explain_fusion(
    model,
    background_eeg:  torch.Tensor,
    background_face: torch.Tensor,
    test_eeg:        torch.Tensor,
    test_face:       torch.Tensor,
    n_background:    int = 50,
    save_path:       Optional[str] = None,
):
    """
    Use SHAP DeepExplainer to explain predictions on the fused embedding.

    Parameters
    ----------
    model            : MultimodalEmotionModel
    background_*     : tensors used as SHAP background (reference distribution)
    test_*           : tensors to explain
    n_background     : number of background samples to use

    Note
    ----
    SHAP DeepExplainer works on the logit outputs.
    We wrap the model's EEG branch for separate SHAP analysis.
    """
    try:
        import shap
    except ImportError:
        print("Install shap: pip install shap")
        return

    # ── Wrapper: EEG-only forward ──────────────────────────────
    class EEGWrapper(torch.nn.Module):
        def __init__(self, full_model, fixed_face):
            super().__init__()
            self.model      = full_model
            self.fixed_face = fixed_face.mean(0, keepdim=True)

        def forward(self, eeg):
            face = self.fixed_face.expand(eeg.shape[0], -1, -1, -1)
            return self.model(face, eeg)["logits"]

    wrapper = EEGWrapper(model, background_face[:n_background])
    wrapper.eval()

    bg_eeg   = background_eeg[:n_background]
    expl     = shap.DeepExplainer(wrapper, bg_eeg)
    shap_vals = expl.shap_values(test_eeg)  # list[num_classes] of (N, T, F)

    # ── Summary plot of first class ────────────────────────────
    shap_arr = np.abs(shap_vals[0]).mean(axis=0)  # (T, F)
    fig, ax  = plt.subplots(figsize=(8, 4))
    im = ax.imshow(shap_arr.T, aspect="auto", cmap="YlOrRd",
                   origin="lower")
    plt.colorbar(im, ax=ax, label="|SHAP|")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("EEG Feature")
    ax.set_title("SHAP Feature Importance (EEG → Anger)")
    plt.tight_layout()

    path = save_path or str(RESULTS_DIR / "shap_eeg.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"SHAP plot saved → {path}")
    return shap_vals
