"""
src/models/attention.py
=======================
Cross-modal attention fusion for CNN (face) + LSTM (EEG) embeddings.

Two fusion strategies are provided:

1. ConcatFusion  – simple concatenation + MLP.
   Fast, no extra parameters for the attention mechanism itself.

2. CrossModalAttention  – each modality queries the other.
   The face embedding queries the EEG context and vice versa.
   The gated outputs are concatenated and projected.

   Mathematically:
       Q_face = W_Q · e_face              (B, d)
       K_eeg, V_eeg = W_K · e_eeg, W_V · e_eeg
       attended_eeg  = softmax(Q_face · K_eeg^T / √d) · V_eeg

   … and symmetrically for eeg→face.

   This is the "attention" fusion mode selected in config.yaml.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 1. Concat Fusion (baseline)
# ─────────────────────────────────────────────────────────────

class ConcatFusion(nn.Module):
    """
    Concatenate both embeddings and project through an MLP.

    Parameters
    ----------
    face_dim   : int  dimension of face (CNN) embedding
    eeg_dim    : int  dimension of EEG  (LSTM) embedding
    hidden_dim : int  MLP hidden layer size
    out_dim    : int  output fused embedding dim
    dropout    : float
    """

    def __init__(
        self,
        face_dim: int = 256,
        eeg_dim: int = 256,
        hidden_dim: int = 256,
        out_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(face_dim + eeg_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        face_emb: torch.Tensor,
        eeg_emb:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        face_emb : (B, face_dim)
        eeg_emb  : (B, eeg_dim)

        Returns
        -------
        fused : (B, out_dim)
        """
        cat = torch.cat([face_emb, eeg_emb], dim=-1)
        return self.mlp(cat)


# ─────────────────────────────────────────────────────────────
# 2. Cross-Modal Attention Fusion (advanced)
# ─────────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention between face and EEG.

    Each modality attends to the other, producing complementary
    context vectors that are then gated and concatenated.

    Parameters
    ----------
    dim        : int  shared embedding dimension (face_dim == eeg_dim)
    num_heads  : int  multi-head attention heads
    hidden_dim : int  post-fusion MLP hidden size
    out_dim    : int  output embedding size
    dropout    : float
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        hidden_dim: int = 256,
        out_dim: int = 256,
        dropout: float = 0.4,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # ── Projection matrices ───────────────────────────────
        # face → queries eeg
        self.face_q = nn.Linear(dim, dim)
        self.eeg_k  = nn.Linear(dim, dim)
        self.eeg_v  = nn.Linear(dim, dim)

        # eeg → queries face
        self.eeg_q  = nn.Linear(dim, dim)
        self.face_k = nn.Linear(dim, dim)
        self.face_v = nn.Linear(dim, dim)

        # ── Gating ────────────────────────────────────────────
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.Sigmoid(),
        )

        # ── Output projection ─────────────────────────────────
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single cross-attention step.
        q/k/v are (B, 1, dim) – treating each sample as a single token.
        Returns attended context (B, dim).
        """
        B = q.size(0)
        q = q.view(B, self.num_heads, self.head_dim)   # (B, H, D_h)
        k = k.view(B, self.num_heads, self.head_dim)
        v = v.view(B, self.num_heads, self.head_dim)

        # scaled dot-product attention
        attn = torch.einsum("bhd,bhd->bh", q, k) * self.scale  # (B, H)
        attn = F.softmax(attn, dim=-1)

        out = attn.unsqueeze(-1) * v                   # (B, H, D_h)
        return out.reshape(B, self.dim)                # (B, dim)

    def forward(
        self,
        face_emb: torch.Tensor,
        eeg_emb:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        face_emb : (B, dim)
        eeg_emb  : (B, dim)

        Returns
        -------
        fused : (B, out_dim)
        """
        # face queries EEG context
        face_attended = self._attend(
            self.face_q(face_emb),
            self.eeg_k(eeg_emb),
            self.eeg_v(eeg_emb),
        )  # (B, dim)

        # EEG queries face context
        eeg_attended = self._attend(
            self.eeg_q(eeg_emb),
            self.face_k(face_emb),
            self.face_v(face_emb),
        )  # (B, dim)

        # Gated fusion
        concat = torch.cat([face_attended, eeg_attended], dim=-1)  # (B, 2*dim)
        gate   = self.gate(concat)
        gated  = concat * gate                          # element-wise gate

        return self.out_proj(gated)                     # (B, out_dim)


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def build_fusion(method: str = "attention", **kwargs) -> nn.Module:
    """
    Factory function.

    Parameters
    ----------
    method : "concat" | "attention"
    **kwargs forwarded to the fusion class constructor.
    """
    if method == "concat":
        concat_keys = {"face_dim", "eeg_dim", "hidden_dim", "out_dim", "dropout"}
        return ConcatFusion(
            **{key: value for key, value in kwargs.items() if key in concat_keys}
        )
    elif method == "attention":
        attention_keys = {"dim", "num_heads", "hidden_dim", "out_dim", "dropout"}
        return CrossModalAttention(
            **{key: value for key, value in kwargs.items() if key in attention_keys}
        )
    else:
        raise ValueError(f"Unknown fusion method: {method!r}. "
                         "Choose 'concat' or 'attention'.")
