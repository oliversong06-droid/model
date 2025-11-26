# ai_predictor/model_conv_lstm.py

from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell.

    Input:
        x_t: (B, C_in, H, W)
        h_prev, c_prev: (B, C_hidden, H, W)
    Output:
        h_t, c_t
    """

    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x_t, h_prev], dim=1)
        conv_out = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)

        return h_t, c_t

    def init_hidden(self, batch_size: int, H: int, W: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        return h, c


class ConvLSTMEncoder(nn.Module):
    """
    Multi-layer ConvLSTM encoder over T_in frames.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels_list: List[int],
        kernel_size: int = 3,
    ):
        super().__init__()

        layers = []
        prev_channels = input_channels
        for hidden_channels in hidden_channels_list:
            layers.append(
                ConvLSTMCell(
                    input_channels=prev_channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                )
            )
            prev_channels = hidden_channels

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        x: (B, T_in, C_in, H, W)
        Returns:
            h_list: [h_T^layer0, h_T^layer1, ...]
            c_list: [c_T^layer0, c_T^layer1, ...]
        """
        B, T_in, C, H, W = x.shape
        device = x.device

        # initialize states
        hs = []
        cs = []
        for layer in self.layers:
            h0, c0 = layer.init_hidden(B, H, W, device)
            hs.append(h0)
            cs.append(c0)

        # iterate over time
        for t in range(T_in):
            x_t = x[:, t]  # (B, C_in, H, W)
            for li, layer in enumerate(self.layers):
                h_prev, c_prev = hs[li], cs[li]
                h_t, c_t = layer(x_t, h_prev, c_prev)
                hs[li], cs[li] = h_t, c_t
                x_t = h_t  # input for next layer

        h_list = [h for h in hs]
        c_list = [c for c in cs]

        return h_list, c_list


class OilSpillPredictor(nn.Module):
    """
    ConvLSTM-based multi-step predictor.

    - Encode past T_in frames with multi-layer ConvLSTM
    - Decode T_out future frames using a single ConvLSTMCell
      on top of the last-layer hidden state.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels_list: List[int] = [32, 32],
        kernel_size: int = 3,
        t_out: int = 5,
    ):
        super().__init__()
        self.t_out = t_out

        self.encoder = ConvLSTMEncoder(
            input_channels=in_channels,
            hidden_channels_list=hidden_channels_list,
            kernel_size=kernel_size,
        )

        last_hidden = hidden_channels_list[-1]
        # Decoder cell that evolves hidden state in time (no external forcing for now)
        self.decoder_cell = ConvLSTMCell(
            input_channels=last_hidden,  # we feed previous h as x
            hidden_channels=last_hidden,
            kernel_size=kernel_size,
        )

        self.out_conv = nn.Conv2d(
            in_channels=last_hidden,
            out_channels=1,  # predict oil channel
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_in, C_in, H, W)

        Returns:
            y_pred: (B, T_out, 1, H, W)
        """
        B, T_in, C, H, W = x.shape
        device = x.device

        h_list, c_list = self.encoder(x)
        # Use only the last layer hidden state as starting point for decoder
        h_t = h_list[-1]
        c_t = c_list[-1]

        preds = []
        for _ in range(self.t_out):
            # Use previous hidden as input to decoder cell
            h_t, c_t = self.decoder_cell(h_t, h_t, c_t)
            y_t = self.out_conv(h_t)
            preds.append(y_t)

        y_pred = torch.stack(preds, dim=1)  # (B, T_out, 1, H, W)
        return y_pred
