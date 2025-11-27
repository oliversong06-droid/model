# ai_predictor/predict_taean.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 기존 모듈 불러오기
from ai_predictor.dataset import get_cmems_dataset
from ai_predictor.model_conv_lstm import ConvLSTMForecaster

# ==========================
# [설정] 태안 데이터 경로로 변경됨!
# ==========================
NPZ_PATH = "data/processed/taean_prepared.npz"
CHECKPOINT_PATH = "checkpoints/convlstm_cmems.pth"

INPUT_LEN = 14
TARGET_LEN = 1
BATCH_SIZE = 1 

def load_model(device):
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"모델 파일이 없습니다: {CHECKPOINT_PATH}")
        
    print(f"[INFO] 모델 로딩 중... {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config = checkpoint["config"]
    
    model = ConvLSTMForecaster(
        in_channels=config["in_channels"],
        hidden_channels=config["hidden_channels"],
        kernel_size=3,
        input_len=config["input_len"],
        target_len=config["target_len"],
        out_channels=1
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

def visualize_prediction(inputs, target, prediction, save_path="taean_result.png"):
    # 첫 번째 배치, 마지막 시점, 첫 번째 채널(유속 U)만 시각화
    input_last = inputs[0, -1, 0, :, :].cpu().numpy()
    target_frame = target[0, 0, 0, :, :].cpu().numpy()
    pred_frame = prediction[0, 0, 0, :, :].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 입력 (과거 14일차 흐름)
    im1 = axes[0].imshow(input_last, cmap='jet') # 잘 보이게 jet 컬러맵 사용
    axes[0].set_title("Input (Last Day)")
    plt.colorbar(im1, ax=axes[0])

    # 2. 실제 정답 (다음 날 실제 흐름)
    im2 = axes[1].imshow(target_frame, cmap='jet')
    axes[1].set_title("Ground Truth (Actual)")
    plt.colorbar(im2, ax=axes[1])

    # 3. AI 예측 (AI가 본 다음 날 흐름)
    im3 = axes[2].imshow(pred_frame, cmap='jet')
    axes[2].set_title("AI Prediction")
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle("Taean Oil Spill Movement Prediction", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] 결과 이미지 저장 완료: {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    
    # 태안 데이터셋 로드
    # (데이터가 적으므로 val_ratio를 0으로 해서 전체를 다 가져옵니다)
    train_loader, _ = get_cmems_dataset(
        npz_path=NPZ_PATH,
        batch_size=BATCH_SIZE,
        input_len=INPUT_LEN,
        target_len=TARGET_LEN,
        val_ratio=0.0 
    )
    
    print("[INFO] 태안 데이터 예측 시작...")
    
    # 첫 번째 데이터(사고 초기 시점)를 가져와서 예측
    # iter()로 데이터를 하나 뽑습니다.
    inputs, targets = next(iter(train_loader))
    
    inputs = inputs.to(device)
    
    with torch.no_grad():
        preds = model(inputs)
    
    # 결과 저장
    visualize_prediction(inputs, targets, preds)

if __name__ == "__main__":
    main()
