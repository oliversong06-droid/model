import numpy as np
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.physics_ops import apply_diffusion, apply_current_advection

def generate_synthetic_data(save_path, num_sequences=10, T_total=20, H=64, W=64):
    """
    Generates synthetic oil spill sequences using physics simulations.
    Returns the path to the saved .npz file.
    """
    print(f"Generating {num_sequences} sequences of length {T_total} with size {H}x{W}...")
    
    # (N, T, C, H, W)
    # C=3: Channel 0=Oil, Channel 1=U (Current X), Channel 2=V (Current Y)
    data = np.zeros((num_sequences, T_total, 3, H, W), dtype=np.float32)
    
    for n in range(num_sequences):
        # 1. Initialize random oil spill (Gaussian blob)
        # Random center
        cx, cy = np.random.randint(H//4, 3*H//4), np.random.randint(W//4, 3*W//4)
        # Create grid
        x = np.arange(W)
        y = np.arange(H)
        xx, yy = np.meshgrid(x, y)
        # Gaussian
        sigma = np.random.uniform(3.0, 8.0)
        oil = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        
        # 2. Initialize random current (U, V)
        # Constant current for the whole sequence (simplification)
        u_vel = np.random.uniform(-0.5, 0.5)
        v_vel = np.random.uniform(-0.5, 0.5)
        
        U_field = np.full((H, W), u_vel)
        V_field = np.full((H, W), v_vel)
        
        # 3. Simulation Loop
        dt = 1.0
        D = 0.1 # Diffusion coefficient
        beta = 1.0 # Advection coefficient
        
        current_oil = oil
        
        for t in range(T_total):
            # Save state
            data[n, t, 0] = current_oil
            data[n, t, 1] = U_field
            data[n, t, 2] = V_field
            
            # Evolve oil
            # Diffusion
            current_oil = apply_diffusion(current_oil, D, dt)
            # Advection (Current)
            current_oil = apply_current_advection(current_oil, u_vel, v_vel, beta, dt)
            
            # Clip to avoid negative values or instability
            current_oil = np.clip(current_oil, 0, 1)
            
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save
    np.savez(save_path, features=data)
    print(f"Data saved to {save_path}")
    return save_path

if __name__ == "__main__":
    generate_synthetic_data("processed/demo_sequences.npz")
