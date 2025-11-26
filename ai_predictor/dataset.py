# ai_predictor/dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset


class OilSpillSequenceDataset(Dataset):
    """
    시계열 grid 데이터를 사용하는 Dataset.

    data_dict 예시 구조 (npz / pkl 로 저장해둘 것):
        {
            "features": (N, T_total, C, H, W),  # oil + forcing fields
        }

    우리가 가져가는 것은:
        입력:  past T_in 프레임
        정답:  다음 1프레임 (oil 채널만)
    """

    def __init__(self, npz_path, T_in=4, oil_channel_index=0, transform=None):
        super().__init__()
        self.npz = np.load(npz_path)
        self.features = self.npz["features"]  # (N, T_total, C, H, W)
        self.T_in = T_in
        self.oil_channel_index = oil_channel_index
        self.transform = transform

        _, T_total, _, _, _ = self.features.shape
        if T_total <= T_in:
            raise ValueError("T_total must be > T_in")

        # 가능한 시점 개수
        self.max_t = T_total - T_in

    def __len__(self):
        # 샘플 수 = N * (T_total - T_in)
        return self.features.shape[0] * self.max_t

    def __getitem__(self, idx):
        N, T_total, C, H, W = self.features.shape
        seq_idx = idx // self.max_t   # 몇 번째 시계열인지
        t0 = idx % self.max_t         # 이 시계열에서의 시작 시간

        x_seq = self.features[seq_idx, t0:t0 + self.T_in]   # (T_in, C, H, W)
        y_next = self.features[seq_idx, t0 + self.T_in, self.oil_channel_index]  # (H, W)

        x_seq = torch.from_numpy(x_seq).float()   # (T_in, C, H, W)
        y_next = torch.from_numpy(y_next).unsqueeze(0).float()  # (1, H, W)

        if self.transform:
            x_seq, y_next = self.transform(x_seq, y_next)

        return x_seq, y_next

