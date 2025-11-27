# ai_predictor/model_conv_lstm.py

from __future__ import annotations
from typing import List, Tuple, Optional

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    표준 ConvLSTM 셀 하나.
    입력:  x_t : (B, C_in, H, W)
           (h, c) : (B, C_hidden, H, W)
    출력:  (h_new, c_new)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x : (B, C_in, H, W)
        B, _, H, W = x.shape

        if state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)  # (B, C_in + C_h, H, W)
        gates = self.conv(combined)
        # 4 * hidden_dim 채널 → i, f, o, g 로 분리
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTMForecaster(nn.Module):
    """
    CMEMS feature sequence를 받아서 다음 1시간(1 frame)의 surface field를 예측하는 모델.

    입력  x : (B, T_in, C_in, H, W)
    출력  y : (B, 1, 1, H, W)   # target_len = 1, out_channels = 1
    """

    def __init__(
        self,
        in_channels: int = 3,          # CMEMS에서 고른 변수 개수
        hidden_channels: List[int] = [32, 32],
        kernel_size: int = 3,
        input_len: int = 14,
        target_len: int = 1,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.input_len = input_len
        self.target_len = target_len
        self.out_channels = out_channels

        cells = []
        for i, h in enumerate(hidden_channels):
            if i == 0:
                c_in = in_channels
            else:
                c_in = hidden_channels[i - 1]
            cells.append(ConvLSTMCell(c_in, h, kernel_size=kernel_size))
        self.cells = nn.ModuleList(cells)

        # 마지막 hidden state → 출력 1채널
        self.output_conv = nn.Conv2d(hidden_channels[-1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_in, C_in, H, W)
        return : (B, 1, 1, H, W)
        """
        B, T_in, C_in, H, W = x.shape
        assert (
            T_in == self.input_len
        ), f"expected input_len={self.input_len}, but got {T_in}"

        # 각 층마다 (h, c) 초기화
        states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for h_dim in self.hidden_channels:
            h0 = torch.zeros(B, h_dim, H, W, device=x.device, dtype=x.dtype)
            c0 = torch.zeros(B, h_dim, H, W, device=x.device, dtype=x.dtype)
            states.append((h0, c0))

        # 입력 시퀀스를 차례로 흘려보내며 최종 hidden state 얻기
        for t in range(T_in):
            x_t = x[:, t]  # (B, C_in, H, W)
            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = states[layer_idx]
                h_new, c_new = cell(x_t, (h_prev, c_prev))
                states[layer_idx] = (h_new, c_new)
                x_t = h_new  # 다음 layer의 입력으로 사용

        # 마지막 레이어의 hidden state로 다음 한 프레임을 예측
        h_top, _ = states[-1]              # (B, C_hidden, H, W)
        y_frame = self.output_conv(h_top)  # (B, 1, H, W)

        # Dataset의 y와 맞추기 위해 (B, 1, 1, H, W) 로 reshape
        y = y_frame.unsqueeze(1)
        return y
