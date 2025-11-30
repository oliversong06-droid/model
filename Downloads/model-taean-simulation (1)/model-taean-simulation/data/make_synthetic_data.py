# data/make_synthetic_data.py
"""
Generate synthetic training sequences using the physics module.

Output:
    data/processed/train_sequences.npz
    - features: (N, T, C, H, W)
      C = 3 channels: [0]=oil, [1]=U, [2]=V
"""

from __future__ import annotations
import os
import numpy as np

from utils.physics_ops import (
    generate_initial_oil,
    generate_current_field,
    step_physics,
)

# ---------------------------
# CONFIG
# ---------------------------
NUM_SEQUENCES = 80      # increase if you want more training data
T_TOTAL = 30            # timesteps per sequence
H, W = 40, 40           # grid size
DX = 1.0                # spatial resolution (arbitrary units)
DT = 1.0                # time step (arbitrary units)
D_BASE = 0.3            # baseline diffusion coefficient (tunable)

RANDOM_SEED = 42


def generate_synthetic_dataset(
    num_sequences: int = NUM_SEQUENCES,
    t_total: int = T_TOTAL,
    H: int = H,
    W: int = W,
) -> np.ndarray:
    """
    Generate synthetic dataset with shape:
        (N, T, C, H, W), C=3
    """
    rng = np.random.default_rng(RANDOM_SEED)

    all_features = np.zeros(
        (num_sequences, t_total, 3, H, W), dtype=np.float32
    )

    for n in range(num_sequences):
        # Initial oil field
        oil = generate_initial_oil(H, W)

        # Time-varying currents
        U, V = generate_current_field(
            T=t_total,
            H=H,
            W=W,
            base_speed_min=0.01,
            base_speed_max=0.05,
            noise_level=0.01,
        )

        # Slightly random diffusion for each sequence
        D = float(D_BASE * rng.uniform(0.5, 1.5))

        for t in range(t_total):
            all_features[n, t, 0] = oil
            all_features[n, t, 1] = U[t]
            all_features[n, t, 2] = V[t]

            # Evolve oil field
            oil = step_physics(
                oil=oil,
                u=U[t],
                v=V[t],
                D=D,
                dt=DT,
                dx=DX,
            )

        print(f"[INFO] Sequence {n + 1}/{num_sequences} generated")

    return all_features


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    save_path = os.path.join(processed_dir, "train_sequences.npz")

    print("[INFO] Generating synthetic dataset...")
    features = generate_synthetic_dataset()
    print("[INFO] Dataset shape:", features.shape)

    np.savez_compressed(save_path, features=features)
    print(f"[INFO] Saved to: {save_path}")


if __name__ == "__main__":
    main()
