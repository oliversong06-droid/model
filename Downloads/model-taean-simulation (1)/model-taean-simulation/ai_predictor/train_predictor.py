# ai_predictor/train_predictor.py

from __future__ import annotations

import os
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai_predictor.dataset import get_cmems_dataset
from ai_predictor.model_conv_lstm import ConvLSTMForecaster


# ==========================
# 설정 값 (태안 데이터 맞춤 수정)
# ==========================

# [수정 1] 데이터 경로를 태안 데이터로 변경
NPZ_PATH = "data/processed/taean_prepared.npz"

INPUT_LEN = 14
TARGET_LEN = 1

BATCH_SIZE = 4

# [수정 2] 데이터가 적으므로 검증 비율을 줄여서(10%) 학습에 더 많이 투자
VAL_RATIO = 0.1

# [수정 3] 학습 횟수 20 -> 100으로 증가 (데이터가 적을 땐 많이 반복해야 함)
N_EPOCHS = 100
LR = 1e-3

CHECKPOINT_DIR = "checkpoints"
# 예측 코드(predict_taean.py)가 이 이름을 찾으므로 그대로 둡니다 (덮어쓰기)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "convlstm_cmems.pth")


# ==========================
# 학습/검증 루프
# ==========================

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    if optimizer is None:
        model.eval()
    else:
        model.train()

    running_loss = 0.0
    n_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(optimizer is not None):
            preds = model(xb)           # (B, 1, 1, H, W)
            loss = criterion(preds, yb)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

        bs = xb.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

    return running_loss / max(1, n_samples)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 1) 데이터셋 / DataLoader
    train_loader, val_loader = get_cmems_dataset(
        npz_path=NPZ_PATH,
        batch_size=BATCH_SIZE,
        input_len=INPUT_LEN,
        target_len=TARGET_LEN,
        val_ratio=VAL_RATIO,
    )

    print(f"[INFO] 데이터 로드 완료: Train {len(train_loader.dataset)}개, Val {len(val_loader.dataset)}개")

    # 2) 모델 생성
    model = ConvLSTMForecaster(
        in_channels=3,
        hidden_channels=[32, 32],
        kernel_size=3,
        input_len=INPUT_LEN,
        target_len=TARGET_LEN,
        out_channels=1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_val_loss = float("inf")
    print(f"[INFO] Start training for {N_EPOCHS} epochs...")

    for epoch in range(1, N_EPOCHS + 1):
        start = time.time()

        train_loss = run_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # 데이터가 너무 적어서 val_loader가 비어있을 경우 대비
        if len(val_loader) > 0:
            val_loss = run_one_epoch(
                model, val_loader, criterion, optimizer=None, device=device
            )
        else:
            val_loss = 0.0

        elapsed = time.time() - start

        if epoch % 10 == 0 or epoch == 1: # 10번마다 로그 출력 (너무 빠를 수 있어서)
            print(
                f"[Epoch {epoch:03d}/{N_EPOCHS}] "
                f"train_loss={train_loss:.4e}, "
                f"val_loss={val_loss:.4e}, "
                f"time={elapsed:.1f}s"
            )

        # 검증 Loss가 줄어들면 저장 (Val 데이터가 없으면 무조건 저장)
        if val_loss < best_val_loss or len(val_loader) == 0:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "input_len": INPUT_LEN,
                        "target_len": TARGET_LEN,
                        "in_channels": 3,
                        "hidden_channels": [32, 32],
                    },
                },
                CHECKPOINT_PATH,
            )

    print("[INFO] Training finished")
    print(f"[INFO] Best val_loss = {best_val_loss:.4e}")


if __name__ == "__main__":
    main()