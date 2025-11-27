# ai_predictor/dataset.py

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    """
    이미 (N, T, C, H, W) 형태로 잘린 시퀀스를 받아
    x: (T_in, C, H, W), y: (T_out=1, 1, H, W) 를 반환.
    """

    def __init__(
        self,
        x: np.ndarray,  # (N, T_in, C, H, W)
        y: np.ndarray,  # (N, 1, 1, H, W)
    ) -> None:
        assert x.shape[0] == y.shape[0]
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def get_cmems_dataset(
    npz_path: str,
    batch_size: int,
    input_len: int,
    target_len: int,
    val_ratio: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """
    cmems_sequences.npz를 읽어서
    - NaN/Inf 처리
    - 전체 mean/std로 정규화
    - (x, y) 분리
    - train/val DataLoader 생성
    을 수행.
    """
    raw = np.load(npz_path)
    features = raw["features"].astype(np.float32)  # (N, T, C, H, W)
    print(f"[DEBUG] raw features shape : {features.shape}")

    # NaN / Inf 통계
    nan_count = int(np.isnan(features).sum())
    inf_count = int(np.isinf(features).sum())
    print(f"[DEBUG] NaN 개수 : {nan_count}, Inf 개수 : {inf_count}")

    # NaN을 제외하고 mean, std 계산
    mean = float(np.nanmean(features))
    std = float(np.nanstd(features))
    print(f"[DEBUG] 정규화 이전 mean={mean:.4f}, std={std:.4f}")

    # NaN 위치는 mean으로 채우기
    features = np.where(np.isnan(features), mean, features)
    # 혹시 모를 inf도 클리핑
    features = np.clip(features, mean - 10 * std, mean + 10 * std)

    # 표준화
    features = (features - mean) / (std + 1e-6)

    # x, y 분리
    #   x : 처음 input_len 프레임
    #   y : 그 다음 target_len 프레임 중 첫 채널만 (surface field 가정)
    x = features[:, :input_len, :, :, :]                         # (N, T_in, C, H, W)
    y = features[:, input_len : input_len + target_len, 0:1]     # (N, T_out, 1, H, W)

    N = x.shape[0]
    print(
        f"[DEBUG] Dataset ready: x={x.shape}, y={y.shape} (N, T, C, H, W)"
    )

    # train / val split
    n_val = max(1, int(N * val_ratio))
    n_train = N - n_val
    train_x, val_x = x[:n_train], x[n_train:]
    train_y, val_y = y[:n_train], y[n_train:]

    print(f"[DEBUG] 전체 샘플 : {N}, train: {n_train}, val: {n_val}")

    train_ds = SequenceDataset(train_x, train_y)
    val_ds = SequenceDataset(val_x, val_y)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # 첫 배치 shape 확인
    bx, by = next(iter(train_loader))
    print(
        f"[DEBUG] 첫 train 배치 x: {bx.shape}, y: {by.shape} "
        "(B, T_in, C, H, W), (B, 1, 1, H, W)"
    )

    return train_loader, val_loader
