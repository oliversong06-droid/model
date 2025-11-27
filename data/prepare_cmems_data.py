# data/prepare_cmems_data.py

import os
import sys
import numpy as np
import xarray as xr

# utils/ 안에 있는 물리 연산 함수 불러오기
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.physics_ops import apply_diffusion, apply_current_advection


def build_cmems_sequences(
    ds,
    num_sequences: int = 32,
    T_total: int = 15,
    H: int = 64,
    W: int = 64,
) -> np.ndarray:
    """
    CMEMS 실측 해류(u, v)를 이용해서 oil + U + V 시퀀스를 만드는 함수.
    출력 shape: (N, T, C, H, W),  C=3 (0: oil, 1: U, 2: V)
    """

    # ---- 1) 변수 이름 맞추기 (필요하면 여기만 수정하면 됨) ----
    # 보통 GLOBAL_ANALYSISFORECAST_PHY_001_024 제품은 uo / vo 를 씀.
    # depth, latitude, longitude 축 이름은 파일마다 조금씩 다르니,
    # 안 맞으면 ds 출력 보고 isel 부분만 고치면 됨.
    u3d = ds["uo"]  # (time, depth, lat, lon) 혹은 (time, depth, y, x)
    v3d = ds["vo"]

    # 표층만 사용 (depth=0)
    u2d = u3d.isel(depth=0)
    v2d = v3d.isel(depth=0)

    # numpy 배열로 변환: (time, Y, X)
    u_all = u2d.values
    v_all = v2d.values

    # 사용할 time 범위 확인
    n_time, full_H, full_W = u_all.shape
    if n_time < T_total + num_sequences:
        raise ValueError(f"시간 축이 부족함: {n_time} < {T_total + num_sequences}")

    # 공간 해상도에서 중앙 HxW 패치 추출 (너무 크면 잘라냄)
    start_y = (full_H - H) // 2 if full_H > H else 0
    start_x = (full_W - W) // 2 if full_W > W else 0

    u_patch = u_all[:, start_y : start_y + H, start_x : start_x + W]
    v_patch = v_all[:, start_y : start_y + H, start_x : start_x + W]

    # 속도 스케일링 (대략 synthetic 데이터 스케일과 비슷하게)
    max_speed = np.nanmax(np.sqrt(u_patch**2 + v_patch**2)) + 1e-6
    u_patch = 0.5 * u_patch / max_speed
    v_patch = 0.5 * v_patch / max_speed

    # 출력 배열
    data = np.zeros((num_sequences, T_total, 3, H, W), dtype=np.float32)

    # 격자 좌표 (oil 초기 분포용)
    xs = np.arange(W)
    ys = np.arange(H)
    xx, yy = np.meshgrid(xs, ys)

    for n in range(num_sequences):
        # time 시작 인덱스: 겹치지 않게 순차로
        t0 = n
        t1 = t0 + T_total

        # 이 시퀀스에 쓸 U, V (T, H, W)
        u_seq = u_patch[t0:t1]
        v_seq = v_patch[t0:t1]

        # ---- oil 초기 분포 (Gaussian blob, 실제 유출 대신) ----
        cx = np.random.randint(H // 4, 3 * H // 4)
        cy = np.random.randint(W // 4, 3 * W // 4)
        sigma = np.random.uniform(3.0, 8.0)

        oil0 = np.exp(-(((xx - cy) ** 2 + (yy - cx) ** 2) / (2 * sigma**2))).astype(
            np.float32
        )
        oil = oil0.copy()

        # 물리 파라미터 (논문 기반으로 나중에 보정 가능)
        dt = 1.0
        D = 0.1  # diffusion coeff
        beta = 1.0  # advection scale

        for t in range(T_total):
            # 저장: oil, u, v
            data[n, t, 0] = oil
            data[n, t, 1] = u_seq[t]
            data[n, t, 2] = v_seq[t]

            # 마지막 프레임은 업데이트 안 함
            if t == T_total - 1:
                break

            # 다음 스텝으로 전파: 확산 + 해류에 의한 advection
            oil = apply_diffusion(oil, D=D, dt=dt)
            oil = apply_current_advection(oil, u_seq[t], v_seq[t], beta=beta, dt=dt)

            # 수치적 이상 방지
            oil = np.clip(oil, 0.0, 1.0)

    return data


def main():
    # 경로 설정
    here = os.path.dirname(__file__)

    raw_nc = os.path.join(here, "raw", "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m.nc")
    save_dir = os.path.join(here, "processed")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cmems_sequences.npz")

    print(f"[INFO] Loading CMEMS file: {raw_nc}")
    ds = xr.open_dataset(raw_nc)

    print("[INFO] Building sequences from CMEMS currents...")
    features = build_cmems_sequences(
        ds,
        num_sequences=32,  # 필요에 따라 늘리기
        T_total=15,
        H=64,
        W=64,
    )

    print(f"[INFO] Saving to {save_path}")
    np.savez_compressed(save_path, features=features)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

