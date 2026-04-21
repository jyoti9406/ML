"""
src/models/multimodal_model.py
==============================
Full multimodal emotion recognition model.

Architecture Diagram
--------------------

  Face Image (B,1,48,48)          EEG Signal (B,128,32)
         │                                 │
  ┌──────▼──────┐                  ┌───────▼──────┐
  │  CNN Encoder │                  │  BiLSTM Enc  │
  │  (4 conv    │                  │  + Temporal  │
  │   blocks)   │                  │  Attention   │
  └──────┬──────┘                  └───────┬──────┘
         │ face_emb (B,256)                │ eeg_emb (B,256)
         └─────────────┬──────────────────┘
                       │
            ┌──────────▼──────────┐
            │  Cross-Modal        │
            │  Attention Fusion   │
            └──────────┬──────────┘
                       │ fused (B,256)
                       │
            ┌──────────▼──────────┐
            │  FC(256→128)→ReLU   │
            │  Dropout            │
            │  FC(128→num_classes)│
            └──────────┬──────────┘
                       │
                   logits (B,4)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .cnn_model import CNNFaceEncoder
from .lstm_model import LSTMEEGEncoder
from .attention import build_fusion

EMOTION_LABELS = ["anger", "fear", "sadness", "disgust"]


class MultimodalEmotionModel(nn.Module):
    """
    Multimodal negative emotion recognition model.

    Parameters
    ----------
    num_classes   : int   number of emotion categories
    img_size      : int   face image spatial size
    in_channels   : int   face image channels (1 = grayscale)
    cnn_filters   : list  filter counts per CNN block
    cnn_emb_dim   : int   CNN output embedding dim
    cnn_dropout   : float CNN dropout
    eeg_n_feat    : int   EEG input feature width
    eeg_hidden    : int   LSTM hidden units (per direction)
    eeg_layers    : int   LSTM stacked layers
    eeg_dropout   : float LSTM dropout
    eeg_emb_dim   : int   LSTM output embedding dim
    fusion_method : str   "attention" | "concat"
    fusion_hidden : int   fusion MLP hidden size
    fusion_dropout: float fusion MLP dropout
    """

    def __init__(
        self,
        num_classes:    int   = 4,
        img_size:       int   = 48,
        in_channels:    int   = 1,
        cnn_filters:    list  | None = None,
        cnn_emb_dim:    int   = 256,
        cnn_dropout:    float = 0.4,
        eeg_n_feat:     int   = 32,
        eeg_hidden:     int   = 128,
        eeg_layers:     int   = 2,
        eeg_dropout:    float = 0.3,
        eeg_emb_dim:    int   = 256,
        fusion_method:  str   = "attention",
        fusion_hidden:  int   = 256,
        fusion_dropout: float = 0.4,
    ):
        super().__init__()

        # ── Encoders ──────────────────────────────────────────
        self.face_encoder = CNNFaceEncoder(
            in_channels=in_channels,
            img_size=img_size,
            filters=cnn_filters,
            embedding_dim=cnn_emb_dim,
            dropout=cnn_dropout,
        )

        self.eeg_encoder = LSTMEEGEncoder(
            n_features=eeg_n_feat,
            hidden_size=eeg_hidden,
            num_layers=eeg_layers,
            dropout=eeg_dropout,
            embedding_dim=eeg_emb_dim,
        )

        # ── Fusion layer ──────────────────────────────────────
        assert cnn_emb_dim == eeg_emb_dim, \
            "CNN and LSTM embedding dims must match for attention fusion."

        self.fusion = build_fusion(
            method=fusion_method,
            dim=cnn_emb_dim,              # for CrossModalAttention
            face_dim=cnn_emb_dim,         # for ConcatFusion
            eeg_dim=eeg_emb_dim,
            hidden_dim=fusion_hidden,
            out_dim=fusion_hidden,
            dropout=fusion_dropout,
        )

        # ── Classifier head ───────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(128, num_classes),
        )

        self.num_classes = num_classes
        self.labels = EMOTION_LABELS[:num_classes]

    # ── Forward ───────────────────────────────────────────────
    def forward(
        self,
        face: torch.Tensor,
        eeg:  torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        face : Tensor  (B, C, H, W)
        eeg  : Tensor  (B, seq_len, n_features)

        Returns
        -------
        dict with:
          "logits"        : (B, num_classes)
          "face_emb"      : (B, cnn_emb_dim)       – for XAI
          "eeg_emb"       : (B, eeg_emb_dim)        – for XAI
          "eeg_attn"      : (B, seq_len)             – temporal attention
          "fused_emb"     : (B, fusion_hidden)       – for SHAP
        """
        face_emb             = self.face_encoder(face)       # (B, D)
        eeg_emb, eeg_attn    = self.eeg_encoder(eeg)         # (B, D), (B, T)
        fused                = self.fusion(face_emb, eeg_emb)  # (B, D_fused)
        logits               = self.classifier(fused)        # (B, C)

        return {
            "logits":    logits,
            "face_emb":  face_emb,
            "eeg_emb":   eeg_emb,
            "eeg_attn":  eeg_attn,
            "fused_emb": fused,
        }

    # ── Convenience: predict single sample ────────────────────
    @torch.no_grad()
    def predict(
        self,
        face: torch.Tensor,
        eeg:  torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> dict:
        """
        Returns top-1 prediction with probabilities.

        Parameters
        ----------
        face : Tensor  (1, C, H, W)
        eeg  : Tensor  (1, T, F)

        Returns
        -------
        {
          "label"       : str,
          "class_id"    : int,
          "confidence"  : float,
          "probs"       : dict[str, float],
          "eeg_attn"    : np.ndarray  shape (T,)
        }
        """
        import torch.nn.functional as F
        import numpy as np

        if device:
            face, eeg = face.to(device), eeg.to(device)
        self.eval()
        out   = self(face, eeg)
        probs = F.softmax(out["logits"], dim=-1)[0]
        idx   = int(probs.argmax())

        return {
            "label":      self.labels[idx],
            "class_id":   idx,
            "confidence": float(probs[idx]),
            "probs":      {lbl: float(probs[i])
                           for i, lbl in enumerate(self.labels)},
            "eeg_attn":   out["eeg_attn"][0].cpu().numpy(),
        }

    # ── Checkpointing ─────────────────────────────────────────
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "MultimodalEmotionModel":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**kwargs)
        model.load_state_dict(ckpt["state_dict"])
        return model

    # ── Parameter count ───────────────────────────────────────
    def parameter_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
