import os
import sys
import torch
import numpy as np
import time
import matplotlib.pyplot as plt



# Ensure we can import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import scipy
except ImportError:
    print("Error: 'scipy' is missing. Please run: pip install scipy")
    sys.exit(1)

from data.make_synthetic_data import generate_synthetic_data
from ai_predictor.train_predictor import train_predictor
from ai_predictor.model_conv_lstm import ConvLSTMPredictor
from utils.biology_ops import update_DO, plankton_response, ecological_recovery_index

def main():
    print("========================================================")
    print("   OIL SPILL AI PREDICTOR - FULL DEMO PIPELINE")
    print("========================================================")
    print("This script will:")
    print("1. Generate synthetic oil spill data (Physics Simulation)")
    print("2. Train an AI model to predict oil movement")
    print("3. Predict a future oil spill scenario")
    print("4. Analyze the ecological impact (Biology Model)")
    print("========================================================\n")

    # --- STEP 1: DATA GENERATION ---
    print("[STEP 1] Generating Synthetic Data...")
    data_path = os.path.join("data", "processed", "demo_sequences.npz")
    # Generate 20 sequences of 15 frames, 64x64
    generate_synthetic_data(data_path, num_sequences=20, T_total=15, H=64, W=64)
    print("Done.\n")

    # --- STEP 2: AI TRAINING ---
    print("[STEP 2] Training AI Model...")
    print("Training for 4 epochs (fast demo mode)...")
    # Train
    train_predictor(
        npz_path=data_path,
        T_in=4,
        batch_size=4,
        num_epochs=4, # Short training for demo
        lr=1e-3
    )
    print("Done.\n")

    # --- STEP 3: PREDICTION ---
    print("[STEP 3] Running Prediction on Test Case...")
    
    # Load Data to get a sample
    npz = np.load(data_path)
    features = npz["features"] # (N, T, C, H, W)
    
    # Take the last sequence as test
    test_seq = features[-1] # (T, C, H, W)
    
    # Input: First 4 frames
    T_in = 4
    input_seq = test_seq[:T_in] # (4, C, H, W)
    true_next = test_seq[T_in, 0] # (H, W) - Oil channel only
    
    # Prepare for model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvLSTMPredictor(input_channels=3, hidden_channels=32, num_layers=2)
    
    # Load trained weights
    model.load_state_dict(torch.load("conv_lstm_predictor.pth", map_location=device))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        x_tensor = torch.from_numpy(input_seq).unsqueeze(0).float().to(device) # (1, 4, C, H, W)
        pred_tensor = model(x_tensor) # (1, 1, H, W)
        pred_oil = pred_tensor.squeeze().cpu().numpy() # (H, W)
        
    print("Prediction complete.")
    print(f"Max Predicted Oil Concentration: {pred_oil.max():.4f}")
    print(f"Max True Oil Concentration: {true_next.max():.4f}")
    print("Done.\n")

    # --- STEP 4: ECOLOGICAL IMPACT ANALYSIS ---
    print("[STEP 4] Calculating Ecological Impact...")
    
    # Initialize Biology Environment
    H, W = pred_oil.shape
    DO_sat = 8.0 # mg/L (Saturation)
    DO_current = np.full((H, W), DO_sat) # Start with healthy ocean
    
    Plankton_ref = 100.0
    Plankton_current = np.full((H, W), Plankton_ref)
    
    # Simulate 1 hour of impact
    dt = 3600 # 1 hour in seconds
    
    # 1. Oxygen Depletion
    # k_consume: consumption rate, k_reaer: reaeration rate
    DO_next = update_DO(DO_current, DO_sat, pred_oil * 10, k_consume=0.05, k_reaer=0.01, dt=dt)
    
    # 2. Plankton Mortality
    Plankton_next = plankton_response(Plankton_current, pred_oil * 10, sens_coeff=0.0001, dt=dt)
    
    # 3. Calculate Recovery Index
    # Assume Benthos is unaffected for this short time
    Benthos_ref = 1.0
    Benthos_curr = 1.0
    
    recovery_index = ecological_recovery_index(
        DO_next, DO_sat,
        Plankton_next, Plankton_ref,
        Benthos_curr, Benthos_ref
    )
    
    avg_index = np.mean(recovery_index)
    min_index = np.min(recovery_index)
    
    print("-" * 30)
    print("ECOLOGICAL IMPACT REPORT")
    print("-" * 30)
    print(f"Average Ecological Recovery Index: {avg_index:.4f} (1.0 = Healthy, 0.0 = Dead)")
    print(f"Worst Impacted Zone Index:       {min_index:.4f}")
    print(f"Average Dissolved Oxygen:        {np.mean(DO_next):.2f} mg/L (Ref: {DO_sat})")
    print(f"Plankton Survival Rate:          {np.mean(Plankton_next)/Plankton_ref*100:.1f}%")
    print("-" * 30)
    
    if min_index < 0.8:
        print("WARNING: SIGNIFICANT ECOLOGICAL DAMAGE DETECTED.")
    else:
        print("STATUS: MINIMAL IMPACT.")

    # --- STEP 5: VISUALIZATION ---
    print("\n[STEP 5] Generating Visual Report...")
    
    # Create a figure to show the movement
    # We will show: 4 Input Frames + 1 Ground Truth + 1 Prediction
    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    
    # Plot Input Frames (History)
    for i in range(4):
        ax = axes[i]
        # input_seq is (4, C, H, W), channel 0 is oil
        img = input_seq[i, 0] 
        ax.imshow(img, cmap='inferno', vmin=0, vmax=1)
        ax.set_title(f"Time t-{3-i}")
        ax.axis('off')
        
    # Plot Ground Truth (Future)
    ax = axes[4]
    ax.imshow(true_next, cmap='inferno', vmin=0, vmax=1)
    ax.set_title("True Future (t+1)")
    ax.axis('off')
    
    # Plot Prediction (Future)
    ax = axes[5]
    ax.imshow(pred_oil, cmap='inferno', vmin=0, vmax=1)
    ax.set_title("AI Prediction (t+1)")
    ax.axis('off')
    
    plt.suptitle("Oil Spill Movement: History vs AI Prediction", fontsize=16)
    plt.tight_layout()
    
    save_path = "images/oil_spill_visual.png"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    abs_path = os.path.abspath(save_path)
    print(f"Visualization saved to: {abs_path}")
    
    os.startfile(abs_path)

    print("Open this image to see the oil spill movement!")
        
    print("\n[DEMO COMPLETED SUCCESSFULLY]")

if __name__ == "__main__":
    main()
