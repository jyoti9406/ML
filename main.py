"""
main.py
=======
Entry point for the Multimodal Emotion Recognition project.

Commands
--------
  train     – train the full multimodal model
  tune      – run Optuna hyperparameter search
  evaluate  – load checkpoint and evaluate on test set
  compare   – train & compare CNN / LSTM / Multimodal
  explain   – generate Grad-CAM + EEG attention + SHAP plots
  simulate  – quick smoke test with random data (no dataset required)

Usage
-----
    python main.py simulate
    python main.py train --eeg data/raw/emotions.csv --face data/raw/facial/train
    python main.py evaluate --checkpoint checkpoints/multimodal_best.pt
    python main.py compare
    python main.py tune --trials 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from src.data.dataset import build_dataloaders, simulate_data
from src.data.eeg_pipeline import EEGPipeline
from src.evaluation.metrics import (
    error_analysis,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_training_curves,
    print_classification_report,
)
from src.models.cnn_model import CNNClassifier
from src.models.lstm_model import LSTMClassifier
from src.models.multimodal_model import MultimodalEmotionModel
from src.training.trainer import Trainer, build_scheduler, resolve_device

# ── Suppress noisy loggers ────────────────────────────────────
logging.getLogger("PIL").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────
# Forward function adapters
# ─────────────────────────────────────────────────────────────

def multimodal_forward(model):
    def fn(batch):
        eeg, face, labels = batch
        return model(face, eeg)["logits"]
    return fn


def cnn_forward(model):
    def fn(batch):
        eeg, face, labels = batch
        return model(face)
    return fn


def lstm_forward(model):
    def fn(batch):
        eeg, face, labels = batch
        logits, _ = model(eeg)
        return logits
    return fn


# ─────────────────────────────────────────────────────────────
# Shared: build loaders
# ─────────────────────────────────────────────────────────────

def prepare_loaders(args, simulate=False):
    if simulate:
        logger.info("Using SIMULATED data")
        X, y = simulate_data(n_samples=600)
        face_root = None
    else:
        logger.info("Loading EEG from: {}", args.eeg)
        pipeline = EEGPipeline()
        X, y     = pipeline.fit_transform(args.eeg,
                                           save_dir="data/processed")
        face_root = args.face

    if face_root is None:
        # ── Simulation: build synthetic image dataset ──────────
        from torch.utils.data import DataLoader, TensorDataset, random_split

        # Fake face images: (N, 1, 48, 48)
        face_X = torch.randn(len(X), 1, 48, 48)
        eeg_T  = torch.tensor(X, dtype=torch.float32)
        y_T    = torch.tensor(y, dtype=torch.long)

        full_ds = TensorDataset(eeg_T, face_X, y_T)
        n_train = int(0.7 * len(full_ds))
        n_val   = int(0.15 * len(full_ds))
        n_test  = len(full_ds) - n_train - n_val
        train_ds, val_ds, test_ds = random_split(
            full_ds, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )
        kw = dict(batch_size=32, num_workers=0)
        return (DataLoader(train_ds, shuffle=True,  **kw),
                DataLoader(val_ds,   shuffle=False, **kw),
                DataLoader(test_ds,  shuffle=False, **kw))

    return build_dataloaders(X, y, face_root,
                              batch_size=args.batch_size,
                              num_workers=0)


# ─────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────

def cmd_simulate(args):
    """Quick smoke test with random data."""
    logger.info("=== Smoke Test (simulated data, 3 epochs) ===")
    args.eeg   = None
    args.face  = None
    args.epochs = 3
    args.batch_size = 32
    args.lr = 1e-3
    args.checkpoint = "checkpoints/smoke_test.pt"
    cmd_train(args, simulate=True)


def cmd_train(args, simulate=False):
    device = resolve_device(getattr(args, "device", "auto"))
    train_loader, val_loader, test_loader = prepare_loaders(args, simulate)

    model     = MultimodalEmotionModel()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = build_scheduler(optimizer, "cosine", args.epochs)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        log_dir="logs/multimodal",
        ckpt_dir="checkpoints",
        patience=10,
        use_amp=False,
    )

    history = trainer.fit(
        train_loader, val_loader,
        epochs=args.epochs,
        forward_fn=multimodal_forward(model),
        model_name="multimodal",
    )

    # ── Test evaluation ───────────────────────────────────────
    logger.info("=== Test Evaluation ===")
    # Reload best checkpoint
    best_ckpt = Path("checkpoints/multimodal_best.pt")
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        logger.info("Loaded best ckpt (val_f1={:.4f})", ckpt.get("val_f1", 0))

    test_m = trainer.evaluate(test_loader, multimodal_forward(model))
    logger.info("Test  acc={:.4f}  f1={:.4f}", test_m["accuracy"], test_m["f1_macro"])

    # ── Collect predictions for plots ─────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b
                     for b in batch]
            logits = multimodal_forward(model)(batch)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(batch[-1].cpu().tolist())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    print_classification_report(all_labels, all_preds)
    plot_confusion_matrix(all_labels, all_preds, title="Multimodal Model")
    plot_training_curves(history)
    error_analysis(all_labels, all_preds)


def cmd_compare(args):
    """Train CNN, LSTM, and Multimodal; compare on test set."""
    device = resolve_device(getattr(args, "device", "auto"))
    args.eeg  = getattr(args, "eeg",  None)
    args.face = getattr(args, "face", None)
    args.batch_size = getattr(args, "batch_size", 32)
    train_loader, val_loader, test_loader = prepare_loaders(
        args, simulate=(args.eeg is None)
    )
    results = {}
    criterion = nn.CrossEntropyLoss()

    for model_name, model, fwd in [
        ("CNN (Face only)",     CNNClassifier(),       cnn_forward),
        ("BiLSTM (EEG only)",   LSTMClassifier(),      lstm_forward),
        ("Multimodal CNN+LSTM", MultimodalEmotionModel(), multimodal_forward),
    ]:
        logger.info("=== Training {} ===", model_name)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = build_scheduler(opt, "cosine", 15)
        trainer = Trainer(model, opt, criterion, device, sch,
                           log_dir=f"logs/{model_name.replace(' ','_')}",
                           ckpt_dir="checkpoints", patience=8,
                           use_amp=False)
        trainer.fit(train_loader, val_loader, epochs=15,
                    forward_fn=fwd(model), model_name=model_name)

        test_m = trainer.evaluate(test_loader, fwd(model))
        results[model_name] = {
            "accuracy": test_m["accuracy"],
            "f1_macro": test_m["f1_macro"],
        }
        logger.info("{}: acc={:.4f}  f1={:.4f}",
                    model_name, test_m["accuracy"], test_m["f1_macro"])

    plot_model_comparison(results)
    logger.info("Comparison plot saved to results/")


def cmd_evaluate(args):
    device = resolve_device(getattr(args, "device", "auto"))
    args.batch_size = getattr(args, "batch_size", 32)
    _, _, test_loader = prepare_loaders(
        args, simulate=(getattr(args, "eeg", None) is None)
    )

    model = MultimodalEmotionModel()
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch  = [b.to(device) if isinstance(b, torch.Tensor) else b
                      for b in batch]
            logits = multimodal_forward(model)(batch)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(batch[-1].cpu().tolist())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    error_analysis(y_true, y_pred)


def cmd_tune(args):
    from src.training.tuner import run_tuning
    args.batch_size = getattr(args, "batch_size", 32)
    X, y = simulate_data(n_samples=600)
    best = run_tuning(
        eeg_X=X, eeg_y=y,
        face_root=getattr(args, "face", "data/raw/facial/train"),
        n_trials=args.trials,
        epochs_per_trial=5,
    )
    logger.info("Best params: {}", best)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multimodal Negative Emotion Recognition"
    )
    sub = p.add_subparsers(dest="command")

    # simulate
    sub.add_parser("simulate", help="Smoke test with random data")

    # train
    tr = sub.add_parser("train", help="Train multimodal model")
    tr.add_argument("--eeg",        type=str, required=True)
    tr.add_argument("--face",       type=str, required=True)
    tr.add_argument("--epochs",     type=int, default=50)
    tr.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    tr.add_argument("--lr",         type=float, default=1e-3)
    tr.add_argument("--device",     type=str, default="auto")

    # compare
    cmp = sub.add_parser("compare", help="Train all 3 models and compare")
    cmp.add_argument("--eeg",       type=str, default=None)
    cmp.add_argument("--face",      type=str, default=None)
    cmp.add_argument("--device",    type=str, default="auto")

    # evaluate
    ev = sub.add_parser("evaluate", help="Evaluate a saved checkpoint")
    ev.add_argument("--checkpoint", type=str, required=True)
    ev.add_argument("--eeg",        type=str, default=None)
    ev.add_argument("--face",       type=str, default=None)
    ev.add_argument("--device",     type=str, default="auto")

    # tune
    tn = sub.add_parser("tune", help="Optuna hyperparameter search")
    tn.add_argument("--trials",     type=int, default=30)
    tn.add_argument("--face",       type=str, default=None)

    return p.parse_args()


COMMANDS = {
    "simulate": cmd_simulate,
    "train":    cmd_train,
    "compare":  cmd_compare,
    "evaluate": cmd_evaluate,
    "tune":     cmd_tune,
}

if __name__ == "__main__":
    args = parse_args()
    if args.command is None:
        print("Use: python main.py <command>  (simulate|train|compare|evaluate|tune)")
        sys.exit(0)
    COMMANDS[args.command](args)
