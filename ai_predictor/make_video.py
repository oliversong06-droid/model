# ai_predictor/make_continuous_video.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ai_predictor.model_conv_lstm import ConvLSTMForecaster

# ==========================
# 설정 값
# ==========================
NPZ_PATH = "data/processed/taean_prepared.npz"
CHECKPOINT_PATH = "checkpoints/convlstm_cmems.pth"
SAVE_PATH = "taean_continuous_flow.gif"

INPUT_LEN = 14
TARGET_LEN = 1

def load_model(device):
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError("모델 파일이 없습니다.")
    
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] 연속 예측 영상을 생성합니다...")
    
    # 1. 모델 로드
    model = load_model(device)
    
    # 2. 전체 원본 데이터 로드 (Loader 안 쓰고 통으로 가져옴)
    raw_data = np.load(NPZ_PATH)
    # features shape: (N_samples, Time_chunk, C, H, W) -> 우리는 이걸 다 합쳐서 긴 시간축으로 복원해야 함
    # 하지만 여기서는 간단하게 첫 번째 샘플부터 순서대로 예측해봅니다.
    # taean_prepared.npz 저장 방식에 따라 다르지만, features 키에 (N, 15, 3, 64, 64)로 저장되어 있음.
    # 연속성을 위해 가장 첫 샘플(0번) ~ 마지막 샘플까지 '정답(Target)' 부분만 이어 붙여서 전체 타임라인을 만듭니다.
    
    data_chunk = raw_data["features"] # (61-14, 15, 3, 64, 64)
    num_samples = data_chunk.shape[0]
    
    print(f"[INFO] 총 프레임 수: {num_samples}시간")

    # 색상 범위 고정 (전체 데이터 기준)
    vmin = data_chunk[:, :, 0, :, :].min()
    vmax = data_chunk[:, :, 0, :, :].max()

    # 3. 그림판 준비
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # 초기 화면
    # 왼쪽: 실제 데이터(Ground Truth) / 오른쪽: AI 예측(Prediction)
    im1 = ax1.imshow(np.zeros((64, 64)), cmap='jet', vmin=vmin, vmax=vmax)
    ax1.set_title("Ground Truth (Actual)")
    
    im2 = ax2.imshow(np.zeros((64, 64)), cmap='jet', vmin=vmin, vmax=vmax)
    ax2.set_title("AI Prediction (Model)")
    
    plt.suptitle("Continuous Ocean Flow Prediction (Taean)", fontsize=15)
    
    # 4. 프레임 업데이트 함수
    def update(i):
        # i번째 샘플 가져오기
        # input: i번째 샘플의 앞쪽 14개
        # target: i번째 샘플의 마지막 1개 (우리가 맞혀야 할 미래)
        
        sample = data_chunk[i] # (15, 3, 64, 64)
        
        input_tensor = torch.tensor(sample[:INPUT_LEN]).unsqueeze(0).float().to(device) # (1, 14, 3, 64, 64)
        target_img = sample[-1, 0, :, :] # (64, 64) - 유속 U 채널만 봄
        
        # AI 예측 수행
        with torch.no_grad():
            pred = model(input_tensor) # (1, 1, 1, 64, 64)
        
        pred_img = pred[0, 0, 0, :, :].cpu().numpy()
        
        # 화면 업데이트
        im1.set_data(target_img) # 실제
        im2.set_data(pred_img)   # 예측
        
        ax1.set_title(f"Actual Flow (Time: {i}h)")
        ax2.set_title(f"AI Prediction (Time: {i}h)")
        
        # 진행 상황 출력
        if i % 10 == 0:
            print(f"[Processing] {i}/{num_samples} frame...")
            
        return [im1, im2]

    # 5. 애니메이션 생성
    ani = animation.FuncAnimation(
        fig, update, frames=range(num_samples), interval=100, blit=False
    )
    
    ani.save(SAVE_PATH, writer='pillow', fps=10)
    print(f"[Success] 영상 저장 완료: {SAVE_PATH}")

if __name__ == "__main__":
    main()