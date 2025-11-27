# ai_predictor/train_predictor.py

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ai_predictor.dataset import get_cmems_dataset
from ai_predictor.model_conv_lstm import ConvLSTMForecaster

# ===== 하이퍼파라미터 / 경로 설정 =====
NPZ_PATH = "data/processed/cmems_sequences.npz"

BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-3

INPUT_LEN = 14      # x에서 사용할 시계열 길이
TARGET_LEN = 1      # y에서 예측할 프레임 수 (현재 모델은 1프레임용)
VAL_RATIO = 0.2


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ----- 데이터 로더 준비 -----
    train_loader, val_loader = get_cmems_dataset(
        npz_path=NPZ_PATH,
        batch_size=BATCH_SIZE,
        input_len=INPUT_LEN,
        target_len=TARGET_LEN,
        val_ratio=VAL_RATIO,
    )
    print("[DEBUG] train/val DataLoader 생성 완료")

    # ----- 모델 생성 -----
    model = ConvLSTMForecaster(
        in_channels=3,
        hidden_channels=[32, 32],
        kernel_size=3,
        input_len=INPUT_LEN,
        target_len=TARGET_LEN,
        out_channels=1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    # ----- 학습 루프 -----
    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        n_train_batch = 0

        for x, y in train_loader:  # x: (B, T_in, C, H, W), y: (B, 1, 1, H, W)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)  # (B, 1, 1, H, W)

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_train_batch += 1

        train_loss = train_loss_sum / max(1, n_train_batch)

        # ----- 검증 -----
        model.eval()
        val_loss_sum = 0.0
        n_val_batch = 0
        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss_sum += loss.item()
                n_val_batch += 1

        val_loss = val_loss_sum / max(1, n_val_batch)
        elapsed = time.time() - t0

        print(
            f"[Epoch {epoch:02d}/{EPOCHS}] "
            f"train_loss={train_loss:.4e}, "
            f"val_loss={val_loss:.4e}, "
            f"time={elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    print("[INFO] Training finished")
    print(f"[INFO] Best val_loss = {best_val_loss:.4e}")


if __name__ == "__main__":
    main()
