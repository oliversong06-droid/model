# cleanup_system/run_cleanup.py

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from simulator import CleanupSimulator

def find_data_file(filename):
    search_paths = [
        os.path.join(current_dir, filename),
        os.path.join(current_dir, "..", filename),
        os.path.join(current_dir, "..", "..", filename),
        os.path.join(current_dir, "..", "..", "model", filename)
    ]
    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None

def main():
    print(">>> [Comparison] AI 적용 vs 미적용 방제 효율 비교 시작...")

    # 1. 데이터 찾기
    target_file = "taean_prediction_output.npy"
    data_path = find_data_file(target_file)

    if data_path is None:
        print(f"[Error] '{target_file}' 파일이 없습니다.")
        return

    # 2. 데이터 로드
    raw_data = np.load(data_path)
    oil_map = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min() + 1e-8)
    print(f"[OK] 데이터 로드 완료. (Total Mass: {np.sum(oil_map):.2f})")

    # 3. 시뮬레이터 설정 (동일한 조건)
    # 배 50척 투입 (차이를 극명하게 보기 위해 자원을 좀 늘림)
    config = {'num_ships': 50, 'ship_capacity': 0.2, 'dispersant_rate': 0.2}
    sim = CleanupSimulator(config)
    
    # 4. 비교 시뮬레이션 (Combined 전략 사용)
    strategy = "combined"
    
    # (A) AI 미적용 (Random / Manual)
    res_no_ai, eff_no_ai = sim.run(oil_map, strategy=strategy, mode="random")
    
    # (B) AI 적용 (Targeted)
    res_ai, eff_ai = sim.run(oil_map, strategy=strategy, mode="targeted")

    # 5. 결과 출력
    print("\n========= [최종 결과 비교] =========")
    print(f"1. AI 미적용 (랜덤 방제):  {eff_no_ai:.2f}% 제거됨")
    print(f"2. AI 적용 (타겟 방제):    {eff_ai:.2f}% 제거됨")
    print(f"------------------------------------")
    print(f"📈 AI 도입 효과: 효율 {eff_ai - eff_no_ai:.2f}%p 증가!")
    print("====================================")

    # 6. 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin, vmax = 0, 1.0
    
    # 원본
    axes[0].imshow(oil_map, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title("Original Spill (Start)")
    axes[0].axis('off')
    
    # AI 미적용
    axes[1].imshow(res_no_ai, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Without AI (Random)\nEfficiency: {eff_no_ai:.1f}%")
    axes[1].axis('off')
    
    # AI 적용
    axes[2].imshow(res_ai, cmap='jet', vmin=vmin, vmax=vmax)
    axes[2].set_title(f"With AI (Targeted)\nEfficiency: {eff_ai:.1f}%")
    axes[2].axis('off')
    
    plt.suptitle(f"Cleanup Efficiency Comparison (Strategy: {strategy})", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(current_dir, "ai_vs_no_ai_result.png")
    plt.savefig(save_path)
    print(f"[Success] 비교 결과 이미지 저장 완료: {save_path}")

if __name__ == "__main__":
    main()