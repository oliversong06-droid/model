# cleanup_system/simulator.py

import numpy as np

class CleanupSimulator:
    def __init__(self, config=None):
        self.config = config if config else {
            'num_ships': 10,
            'ship_capacity': 0.1,
            'dispersant_rate': 0.3
        }

    def run(self, oil_map, strategy="combined", mode="targeted"):
        """
        mode: 
          - 'targeted' (AI 적용): 농도가 높은 곳(Hotspot)을 찾아가서 제거
          - 'random' (AI 미적용): 위치를 몰라서 무작위 구역을 탐색하며 제거
        """
        current_oil = oil_map.copy()
        H, W = current_oil.shape
        
        # 1. 화학적 방제 (Dispersant)
        if strategy in ["chemical", "combined"]:
            rate = self.config.get('dispersant_rate', 0.3)
            # AI가 없으면 효율이 떨어짐 (오염되지 않은 곳에도 뿌리게 됨)
            effective_rate = rate if mode == "targeted" else rate * 0.5
            current_oil = current_oil * (1.0 - effective_rate)
            
        # 2. 물리적 방제 (Ships)
        if strategy in ["ships", "combined"]:
            ships = self.config.get('num_ships', 10)
            capacity = self.config.get('ship_capacity', 0.1)
            
            # [핵심 비교 로직]
            if mode == "targeted":
                # AI 적용: 농도가 높은 순서대로 정렬 (내림차순) -> 핫스팟 타격
                target_indices = np.argsort(current_oil.ravel())[::-1][:ships]
            else:
                # AI 미적용: 무작위 순서로 섞음 -> 랜덤 타격 (운 좋으면 닦고, 아니면 허탕)
                # 오염이 있는 곳(>0) 중에서라도 랜덤하게 찾으려면 현실성 부여
                # 여기서는 완전 랜덤 탐색 가정
                all_indices = np.arange(current_oil.size)
                target_indices = np.random.choice(all_indices, ships, replace=False)

            rows, cols = np.unravel_index(target_indices, (H, W))
            
            for r, c in zip(rows, cols):
                amount = current_oil[r, c]
                # 제거량은 현재 있는 양을 넘을 수 없음
                removal = min(amount, capacity)
                current_oil[r, c] -= removal
                
        # 효율 계산
        initial = np.sum(oil_map)
        final = np.sum(current_oil)
        efficiency = (initial - final) / initial * 100 if initial > 0 else 0
        
        return current_oil, efficiency