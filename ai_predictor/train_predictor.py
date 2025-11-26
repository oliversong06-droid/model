# ai_predictor/train_predictor.py

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from ai_predictor.dataset import OilSpillSequenceDataset
from ai_predictor.model_conv_lstm import ConvLSTMPredictor


def train_predictor(
    npz_path,
    T_in=4,
    batch_size=4,
    num_epochs=10,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # 1) Dataset & DataLoader
    dataset = OilSpillSequenceDataset(npz_path=npz_path, T_in=T_in)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2) 모델 생성 (채널 수는 npz features 기준으로 자동 결정)
    sample_x, _ = dataset[0]  # (T_in, C, H, W)
    _, C, H, W = sample_x.shape
    model = ConvLSTMPredictor(input_channels=C, hidden_channels=32, num_layers=2)
    model.to(device)

    # 3) Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4) 학습 루프
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x_seq, y_true in loader:
            # x_seq: (B, T_in, C, H, W)
            # y_true: (B, 1, H, W)
            x_seq = x_seq.to(device)
            y_true = y_true.to(device)

            optimizer.zero_grad()
            y_pred = model(x_seq)  # (B, 1, H, W)

            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_seq.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] loss = {avg_loss:.6f}")

    # 5) 모델 저장
    torch.save(model.state_dict(), "conv_lstm_predictor.pth")
    print("Model saved to conv_lstm_predictor.pth")


if __name__ == "__main__":
    # TODO: npz_path를 실제 데이터 경로로 바꿔야 함
    train_predictor(npz_path="data/processed/train_sequences.npz")

