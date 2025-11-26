
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    기본 ConvLSTM 셀.
    입력:  (B, C_in, H, W)
    hidden: (h, c) 각각 (B, C_hidden, H, W)
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, state):
        h, c = state  # (B, C_h, H, W)
        combined = torch.cat([x, h], dim=1)  # (B, C_in + C_h, H, W)
        gates = self.conv(combined)
        (i, f, o, g) = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch_size, spatial_size, device=None):
        H, W = spatial_size
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        return h, c


class ConvLSTMPredictor(nn.Module):
    """
    과거 T_in 프레임의 (oil + forcing field)를 입력으로 받아
    다음 한 프레임의 oil 분포를 예측하는 기본 ConvLSTM 모델.

    - input_channels: grid 당 채널 수
        예: [oil, u_curr, v_curr, u_wind, v_wind, Hs, Tp, SST] 등
    - hidden_channels: ConvLSTM hidden 크기
    """

    def __init__(self, input_channels, hidden_channels=32, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        cells = []
        for layer_idx in range(num_layers):
            in_ch = input_channels if layer_idx == 0 else hidden_channels
            cells.append(ConvLSTMCell(in_ch, hidden_channels))
        self.cells = nn.ModuleList(cells)

        # 마지막 hidden → oil 예측 채널 (1채널 가정: oil thickness/conc)
        self.out_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x_seq):
        """
        x_seq: (B, T_in, C_in, H, W)
        출력:  (B, 1, H, W)  (다음 시간의 oil 분포)
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        # layer별 hidden state 초기화
        states = []
        for cell in self.cells:
            states.append(cell.init_state(B, (H, W), device=device))

        # 시간 순회
        for t in range(T):
            xt = x_seq[:, t]  # (B, C, H, W)
            for layer_idx, cell in enumerate(self.cells):
                h, c = states[layer_idx]
                h, c = cell(xt, (h, c))
                states[layer_idx] = (h, c)
                xt = h  # 다음 layer 입력

        # 마지막 layer hidden → output
        h_last, _ = states[-1]
        out = self.out_conv(h_last)
        # 음수 유막은 없으니 ReLU or clamp
        out = torch.clamp(out, min=0.0)
        return out
