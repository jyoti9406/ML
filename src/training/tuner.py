"""
src/training/tuner.py
=====================
Optuna-based hyperparameter search for the multimodal model.

Searches over:
  - CNN dropout, filter multiplier
  - LSTM hidden size, num layers, dropout
  - Fusion: concat vs attention
  - Learning rate, weight decay
  - Batch size

Usage
-----
    from src.training.tuner import run_tuning

    best = run_tuning(
        eeg_X=X_train, eeg_y=y_train,
        face_root="data/raw/facial/train",
        n_trials=30,
        timeout=3600,
    )
    print(best)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

logger = logging.getLogger(__name__)


def _build_and_train(
    trial: optuna.Trial,
    eeg_X:     np.ndarray,
    eeg_y:     np.ndarray,
    face_root: str,
    device:    torch.device,
    epochs:    int = 10,
    seed:      int = 42,
) -> float:
    """
    Objective function called by Optuna for each trial.
    Returns validation macro-F1 (maximise).
    """
    # ── Lazy imports (avoid circular at module level) ─────────
    from ..data.dataset import MultimodalEmotionDataset, build_dataloaders
    from ..data.face_pipeline import build_train_transform, build_eval_transform
    from ..models.multimodal_model import MultimodalEmotionModel
    from ..evaluation.metrics import compute_metrics

    # ── Sample hyperparameters ────────────────────────────────
    lr           = trial.suggest_float("lr",           1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])

    cnn_dropout  = trial.suggest_float("cnn_dropout",  0.2, 0.6, step=0.1)
    eeg_hidden   = trial.suggest_categorical("eeg_hidden",  [64, 128, 256])
    eeg_layers   = trial.suggest_int("eeg_layers",     1, 3)
    eeg_dropout  = trial.suggest_float("eeg_dropout",  0.1, 0.5, step=0.1)
    fusion       = trial.suggest_categorical("fusion",  ["concat", "attention"])

    # ── Build dataloaders ─────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(
        eeg_X=eeg_X,
        eeg_y=eeg_y,
        face_root=face_root,
        batch_size=batch_size,
        seed=seed,
    )

    # ── Build model ───────────────────────────────────────────
    model = MultimodalEmotionModel(
        cnn_dropout=cnn_dropout,
        eeg_hidden=eeg_hidden,
        eeg_layers=eeg_layers,
        eeg_dropout=eeg_dropout,
        fusion_method=fusion,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # ── Mini training loop ────────────────────────────────────
    def forward_fn(batch):
        eeg, face, labels = batch
        return model(face.to(device), eeg.to(device))["logits"]

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b
                     for b in batch]
            optimizer.zero_grad(set_to_none=True)
            logits = forward_fn(batch)
            loss   = criterion(logits, batch[-1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ── Validate ──────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) if isinstance(b, torch.Tensor) else b
                         for b in batch]
                logits = forward_fn(batch)
                preds  = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(batch[-1].cpu().numpy().tolist())

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        f1      = metrics["f1_macro"]

        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        best_f1 = max(best_f1, f1)

    return best_f1


def run_tuning(
    eeg_X:     np.ndarray,
    eeg_y:     np.ndarray,
    face_root: str,
    n_trials:  int = 30,
    timeout:   int = 3600,
    epochs_per_trial: int = 10,
    study_name: str = "emotion_recognition",
    storage:   Optional[str] = None,
    device_str: str = "auto",
    seed:      int = 42,
) -> dict:
    """
    Run an Optuna hyperparameter search.

    Parameters
    ----------
    eeg_X / eeg_y : training data (full array; split is done inside)
    face_root     : directory with per-class image subdirs
    n_trials      : number of Optuna trials
    timeout       : max search time in seconds
    storage       : Optuna DB URL (e.g. "sqlite:///optuna.db")
                    None = in-memory (results lost after run)

    Returns
    -------
    best_params : dict  best hyperparameters found
    """
    from ..training.trainer import resolve_device
    device = resolve_device(device_str)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=3
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    logger.info("Starting Optuna search: %d trials, timeout=%ds", n_trials, timeout)

    study.optimize(
        lambda trial: _build_and_train(
            trial, eeg_X, eeg_y, face_root,
            device=device,
            epochs=epochs_per_trial,
            seed=seed,
        ),
        n_trials=n_trials,
        timeout=timeout,
        catch=(RuntimeError,),
        show_progress_bar=True,
    )

    best = study.best_params
    logger.info("Best trial  val_f1=%.4f  params=%s",
                study.best_value, best)

    # ── Importance plot (saved to results/) ───────────────────
    try:
        import plotly
        from pathlib import Path
        fig = optuna.visualization.plot_param_importances(study)
        Path("results").mkdir(exist_ok=True)
        fig.write_html("results/optuna_importances.html")
        fig2 = optuna.visualization.plot_optimization_history(study)
        fig2.write_html("results/optuna_history.html")
        logger.info("Optuna plots saved to results/")
    except Exception as e:
        logger.warning("Could not save Optuna plots: %s", e)

    return best
