# convert_data.py

import xarray as xr
import numpy as np
import torch
import torch.nn.functional as F
import os

# ==========================================
# [설정] 파일 이름 확인
# ==========================================
NC_FILE_PATH = "taean.nc" 
OUTPUT_PATH = "data/processed/taean_prepared.npz"

TARGET_SIZE = (64, 64)
# 모델 학습때 14(입력)+1(정답) = 15개를 한 세트로 썼으므로, 여기서도 맞춰줍니다.
SEQUENCE_LENGTH = 15 

def preprocess_nc_to_npz():
    if not os.path.exists(NC_FILE_PATH):
        print(f"[Error] 파일을 찾을 수 없습니다: {NC_FILE_PATH}")
        return

    print(f"[INFO] NetCDF 파일 로딩 중...: {NC_FILE_PATH}")
    ds = xr.open_dataset(NC_FILE_PATH)
    
    try:
        u = ds['uo'].values
        v = ds['vo'].values
        if 'thetao' in ds:
            feat3 = ds['thetao'].values
        elif 'zos' in ds:
            feat3 = ds['zos'].values
        else:
            feat3 = np.zeros_like(u)
    except KeyError as e:
        print(f"[Error] 변수 추출 실패: {e}")
        return

    if u.ndim == 4:
        u = u.squeeze()
        v = v.squeeze()
        feat3 = feat3.squeeze()

    # 결측치 처리
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)
    feat3 = np.nan_to_num(feat3, nan=0.0)

    # (Time, 3, H, W) 형태로 합치기
    raw_data = np.stack([u, v, feat3], axis=1) # Axis 1에 채널 배치
    
    # 리사이징
    tensor_data = torch.from_numpy(raw_data).float()
    resized_data = F.interpolate(
        tensor_data, 
        size=TARGET_SIZE, 
        mode='bilinear', 
        align_corners=False
    )
    
    # 정규화
    mean = resized_data.mean()
    std = resized_data.std()
    normalized_data = (resized_data - mean) / (std + 1e-6)
    
    # ==========================================================
    # [핵심 수정] 긴 데이터를 조각조각 자르기 (Sliding Window)
    # (61, 3, 64, 64) -> (N, 15, 3, 64, 64) 로 변경
    # ==========================================================
    data_numpy = normalized_data.numpy()
    total_time = data_numpy.shape[0]
    
    sequences = []
    # 0일부터 시작해서 15일치씩 묶음
    for i in range(total_time - SEQUENCE_LENGTH + 1):
        # i부터 i+15까지 자름
        seq = data_numpy[i : i + SEQUENCE_LENGTH]
        sequences.append(seq)
    
    final_data = np.stack(sequences)
    
    print(f"[INFO] 데이터 변환 완료!")
    print(f"       원본: {data_numpy.shape} (Time, C, H, W)")
    print(f"       변환: {final_data.shape} (Sample, Time, C, H, W) <- 5차원 완성!")

    # 저장 (이름표 features로 통일)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(OUTPUT_PATH, features=final_data)
    print(f"[Success] 저장 완료: {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_nc_to_npz()