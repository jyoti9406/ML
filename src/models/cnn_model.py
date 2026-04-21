"""
src/models/cnn_model.py
=======================
CNN backbone for facial expression feature extraction.

Architecture
------------
Input: (B, 1, 48, 48)

Block 1: Conv2d(1→32)   + BN + ReLU + MaxPool  → (B, 32, 24, 24)
Block 2: Conv2d(32→64)  + BN + ReLU + MaxPool  → (B, 64, 12, 12)
Block 3: Conv2d(64→128) + BN + ReLU + MaxPool  → (B, 128, 6, 6)
Block 4: Conv2d(128→256)+ BN + ReLU + MaxPool  → (B, 256, 3, 3)

Flatten → FC(2304→256) → ReLU → Dropout → embedding (B, 256)

Reasoning
---------
- Small architecture fits the 48×48 FER2013/CK+ images.
- BatchNorm after every conv stabilises training.
- Embedding dim 256 matches the LSTM output so fusion is symmetric.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU + MaxPool."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout_2d: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool_size)
        self.drop = nn.Dropout2d(dropout_2d) if dropout_2d > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(F.relu(self.bn(self.conv(x)))))


class CNNFaceEncoder(nn.Module):
    """
    Convolutional encoder that maps a grayscale face image to
    a fixed-size embedding vector.

    Parameters
    ----------
    in_channels   : int   1 (grayscale) or 3 (RGB)
    img_size      : int   spatial resolution (default 48)
    filters       : list  number of filters per conv block
    embedding_dim : int   output embedding dimensionality
    dropout       : float dropout rate before the embedding FC
    """

    def __init__(
        self,
        in_channels: int = 1,
        img_size: int = 48,
        filters: list[int] | None = None,
        embedding_dim: int = 256,
        dropout: float = 0.4,
    ):
        super().__init__()
        filters = filters or [32, 64, 128, 256]

        # ── Convolutional blocks ──────────────────────────────
        blocks = []
        ch = in_channels
        for out_ch in filters:
            blocks.append(ConvBlock(ch, out_ch, kernel_size=3, pool_size=2))
            ch = out_ch
        self.conv_blocks = nn.Sequential(*blocks)

        # ── Compute flattened size after all pooling ──────────
        n_pools = len(filters)
        spatial = img_size // (2 ** n_pools)      # e.g. 48 // 16 = 3
        flat_dim = filters[-1] * spatial * spatial  # 256 * 3 * 3 = 2304

        # ── Embedding head ────────────────────────────────────
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.embedding_dim = embedding_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  shape (B, C, H, W)

        Returns
        -------
        embedding : Tensor  shape (B, embedding_dim)
        """
        feat_map = self.conv_blocks(x)   # (B, 256, 3, 3)
        return self.head(feat_map)       # (B, embedding_dim)

    # ── Grad-CAM hook support ─────────────────────────────────
    def get_gradcam_layer(self) -> nn.Module:
        """Return the last conv layer for Grad-CAM visualisation."""
        return self.conv_blocks[-1].conv


# ─────────────────────────────────────────────────────────────
# Standalone classifier (single-modal CNN baseline)
# ─────────────────────────────────────────────────────────────

class CNNClassifier(nn.Module):
    """
    Full CNN-only emotion classifier for single-modal baseline.

    Parameters
    ----------
    num_classes   : int  number of output emotion classes
    All others forwarded to CNNFaceEncoder.
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        img_size: int = 48,
        filters: list[int] | None = None,
        embedding_dim: int = 256,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.encoder = CNNFaceEncoder(
            in_channels=in_channels,
            img_size=img_size,
            filters=filters,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits : Tensor  shape (B, num_classes)
        """
        emb = self.encoder(x)
        return self.classifier(emb)
