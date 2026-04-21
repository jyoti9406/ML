"""
src/models/lstm_model.py
========================
Bidirectional LSTM encoder for EEG temporal feature extraction.

Architecture
------------
Input: (B, seq_len, n_features)  e.g. (B, 128, 32)

BiLSTM stack (2 layers, hidden=128, bidirectional)
  → (B, seq_len, 256)   [128 × 2 directions]

Temporal Attention  (soft-weight the most emotion-relevant timesteps)
  → context vector (B, 256)

FC(256 → 256) → ReLU → Dropout → embedding (B, 256)

Reasoning
---------
- Bidirectional LSTM captures both past and future context in EEG.
- Temporal attention lets the model focus on emotionally salient
  bursts (e.g. alpha-wave suppression during fear).
- Output dim 256 matches CNN embedding for symmetric fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Temporal Attention
# ─────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Soft attention over the time dimension of LSTM output.

    Computes a weighted sum of hidden states:
        e_t = tanh(W · h_t + b)
        α_t = softmax(v · e_t)
        context = Σ α_t · h_t

    Parameters
    ----------
    hidden_dim : int  dimension of each h_t
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden : Tensor  shape (B, T, hidden_dim)

        Returns
        -------
        context : Tensor  shape (B, hidden_dim)
        attn_weights : Tensor  shape (B, T)   for visualisation
        """
        energy = torch.tanh(self.W(hidden))          # (B, T, H)
        scores = self.v(energy).squeeze(-1)           # (B, T)
        weights = F.softmax(scores, dim=-1)           # (B, T)
        context = torch.bmm(
            weights.unsqueeze(1), hidden              # (B, 1, T) @ (B, T, H)
        ).squeeze(1)                                  # (B, H)
        return context, weights


# ─────────────────────────────────────────────────────────────
# BiLSTM Encoder
# ─────────────────────────────────────────────────────────────

class LSTMEEGEncoder(nn.Module):
    """
    Bidirectional LSTM encoder with temporal attention.

    Parameters
    ----------
    n_features    : int   input feature width per timestep
    hidden_size   : int   LSTM hidden units (per direction)
    num_layers    : int   stacked LSTM layers
    dropout       : float dropout between LSTM layers
    embedding_dim : int   final output embedding size
    """

    def __init__(
        self,
        n_features: int = 32,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        embedding_dim: int = 256,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2   # bidirectional

        self.attention = TemporalAttention(lstm_out_dim)

        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.embedding_dim = embedding_dim
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor  shape (B, seq_len, n_features)

        Returns
        -------
        embedding    : Tensor  shape (B, embedding_dim)
        attn_weights : Tensor  shape (B, seq_len)   – for XAI
        """
        lstm_out, _ = self.lstm(x)          # (B, T, 2*H)
        context, attn_w = self.attention(lstm_out)  # (B, 2*H)
        emb = self.head(context)            # (B, embedding_dim)
        return emb, attn_w


# ─────────────────────────────────────────────────────────────
# Standalone LSTM classifier (single-modal EEG baseline)
# ─────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    Full LSTM-only emotion classifier for single-modal baseline.
    """

    def __init__(
        self,
        num_classes: int = 4,
        n_features: int = 32,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        embedding_dim: int = 256,
    ):
        super().__init__()
        self.encoder = LSTMEEGEncoder(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            embedding_dim=embedding_dim,
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits       : Tensor  shape (B, num_classes)
        attn_weights : Tensor  shape (B, seq_len)
        """
        emb, attn_w = self.encoder(x)
        return self.classifier(emb), attn_w
