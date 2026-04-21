"""
src/training/trainer.py
=======================
Modular training engine with:
  - Mixed-precision (AMP) training
  - Learning rate scheduling (cosine / step / plateau)
  - Early stopping
  - Gradient clipping
  - TensorBoard + loguru logging
  - Best-checkpoint saving
  - Cross-validation support
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..evaluation.metrics import compute_metrics


# ─────────────────────────────────────────────────────────────
# Helper: choose device
# ─────────────────────────────────────────────────────────────

def resolve_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class Trainer:
    """
    Generic trainer that works with any model exposing:
        out = model(face, eeg)          # for multimodal
        out = model(x)                  # for single-modal (via wrapper)

    The caller supplies a `forward_fn` to abstract the call signature.

    Parameters
    ----------
    model          : nn.Module
    optimizer      : torch.optim.Optimizer
    criterion      : loss function (CrossEntropyLoss recommended)
    device         : torch.device
    scheduler      : optional LR scheduler
    clip_grad_norm : float   0 = disabled
    log_dir        : str     TensorBoard log directory
    ckpt_dir       : str     checkpoint directory
    patience       : int     early stopping patience (epochs)
    """

    def __init__(
        self,
        model:          nn.Module,
        optimizer:      torch.optim.Optimizer,
        criterion:      nn.Module,
        device:         torch.device,
        scheduler       = None,
        clip_grad_norm: float = 1.0,
        log_dir:        str   = "logs/",
        ckpt_dir:       str   = "checkpoints/",
        patience:       int   = 10,
        use_amp:        bool  = True,
    ):
        self.model          = model.to(device)
        self.optimizer      = optimizer
        self.criterion      = criterion
        self.device         = device
        self.scheduler      = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.patience       = patience
        self.use_amp        = use_amp and device.type == "cuda"

        self.scaler  = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.writer  = SummaryWriter(log_dir=log_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._best_val_f1  = -1.0
        self._no_improve   = 0
        self.history: dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "val_f1":     [],
        }

    # ── One epoch ─────────────────────────────────────────────
    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        forward_fn: Callable,
    ) -> dict:
        self.model.train() if training else self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                # ── Move to device ────────────────────────────
                batch = [b.to(self.device) if isinstance(b, torch.Tensor)
                         else b for b in batch]
                labels = batch[-1]

                # ── Forward ───────────────────────────────────
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = forward_fn(batch)
                    loss   = self.criterion(logits, labels)

                # ── Backward ──────────────────────────────────
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.clip_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        n = len(all_labels)
        avg_loss = total_loss / n
        metrics  = compute_metrics(
            np.array(all_labels), np.array(all_preds)
        )
        metrics["loss"] = avg_loss
        return metrics

    # ── Full training loop ────────────────────────────────────
    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int,
        forward_fn:   Callable,
        model_name:   str = "model",
    ) -> dict[str, list]:
        """
        Train for `epochs` with early stopping.

        Parameters
        ----------
        forward_fn : Callable
            Receives the raw batch (list of tensors) and returns logits.
            Example for multimodal:
                def fwd(batch):
                    eeg, face, labels = batch
                    return model(face, eeg)["logits"]

        Returns
        -------
        self.history   training metrics over epochs
        """
        logger.info("Training on {} | AMP={}", self.device, self.use_amp)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_m = self._run_epoch(train_loader, training=True,
                                      forward_fn=forward_fn)
            val_m   = self._run_epoch(val_loader,   training=False,
                                      forward_fn=forward_fn)

            # ── LR scheduling ─────────────────────────────────
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    if isinstance(self.scheduler,
                                  torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_m["loss"])
                    else:
                        self.scheduler.step()

            # ── History ───────────────────────────────────────
            self.history["train_loss"].append(train_m["loss"])
            self.history["val_loss"].append(val_m["loss"])
            self.history["train_acc"].append(train_m["accuracy"])
            self.history["val_acc"].append(val_m["accuracy"])
            self.history["val_f1"].append(val_m["f1_macro"])

            # ── TensorBoard ───────────────────────────────────
            self.writer.add_scalars(
                "Loss", {"train": train_m["loss"], "val": val_m["loss"]}, epoch
            )
            self.writer.add_scalars(
                "Accuracy", {"train": train_m["accuracy"],
                             "val":   val_m["accuracy"]}, epoch
            )
            self.writer.add_scalar("LR",
                self.optimizer.param_groups[0]["lr"], epoch)

            elapsed = time.time() - t0
            logger.info(
                "Epoch {:03d}/{:03d}  "
                "loss {:.4f}/{:.4f}  acc {:.3f}/{:.3f}  "
                "f1 {:.3f}  lr {:.2e}  [{:.1f}s]",
                epoch, epochs,
                train_m["loss"], val_m["loss"],
                train_m["accuracy"], val_m["accuracy"],
                val_m["f1_macro"],
                self.optimizer.param_groups[0]["lr"],
                elapsed,
            )

            # ── Checkpoint ────────────────────────────────────
            if val_m["f1_macro"] > self._best_val_f1:
                self._best_val_f1 = val_m["f1_macro"]
                self._no_improve  = 0
                ckpt_path = self.ckpt_dir / f"{model_name}_best.pt"
                torch.save(
                    {
                        "epoch":      epoch,
                        "state_dict": self.model.state_dict(),
                        "optimizer":  self.optimizer.state_dict(),
                        "val_f1":     val_m["f1_macro"],
                        "val_acc":    val_m["accuracy"],
                    },
                    ckpt_path,
                )
                logger.info("New best -> {}", ckpt_path)
            else:
                self._no_improve += 1
                if self._no_improve >= self.patience:
                    logger.info("Early stopping at epoch {}", epoch)
                    break

        self.writer.close()
        return self.history

    # ── Evaluate on test set ──────────────────────────────────
    def evaluate(
        self, loader: DataLoader, forward_fn: Callable
    ) -> dict:
        return self._run_epoch(loader, training=False,
                                forward_fn=forward_fn)


# ─────────────────────────────────────────────────────────────
# Scheduler Factory
# ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, method: str, epochs: int, **kwargs):
    if method == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=kwargs.get("eta_min", 1e-6)
        )
    if method == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.5),
        )
    if method == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=kwargs.get("patience", 5),
            factor=kwargs.get("factor", 0.5), verbose=True,
        )
    raise ValueError(f"Unknown scheduler: {method!r}")
