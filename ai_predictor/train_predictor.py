# ai_predictor/train_predictor.py

from __future__ import annotations
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW

from ai_predictor.dataset import build_dataloaders, T_IN, T_OUT
from ai_predictor.model_conv_lstm import OilSpillPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-3


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X = X.to(DEVICE)  # (B, T_IN, C, H, W)
        y = y.to(DEVICE)  # (B, T_OUT, 1, H, W)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npz_path = os.path.join(base_dir, "data", "processed", "train_sequences.npz")

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Training data not found at {npz_path}. "
            f"Run data/make_synthetic_data.py first."
        )

    train_loader, val_loader, test_loader = build_dataloaders(
        npz_path, batch_size=BATCH_SIZE, t_in=T_IN, t_out=T_OUT
    )

    model = OilSpillPredictor(in_channels=3, t_out=T_OUT).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    ckpt_dir = os.path.join(base_dir, "ai_predictor", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "predictor_best.pt")

    best_val_loss = float("inf")

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Training for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = eval_epoch(model, val_loader, criterion)
        dt = time.time() - t0

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f} "
            f"({dt:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": {
                        "T_IN": T_IN,
                        "T_OUT": T_OUT,
                    },
                },
                best_ckpt_path,
            )
            print(f"    [INFO] New best model saved to {best_ckpt_path}")

    test_loss = eval_epoch(model, test_loader, criterion)
    print(f"[INFO] Final test_loss={test_loss:.6f}")


if __name__ == "__main__":
    main()
