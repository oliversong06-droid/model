import streamlit as st
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import plotly.express as px
import plotly.graph_objects as go
import time
import pandas as pd
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your modules
from data.make_synthetic_data import generate_synthetic_dataset
from ai_predictor.model_conv_lstm import ConvLSTMPredictor
from utils.biology_ops import update_DO, plankton_response, ecological_recovery_index
from utils.chemistry_ops import check_toxicity_thresholds
from utils.metrics import calculate_metrics

# Page Config
st.set_page_config(
    page_title="AI Oil Spill Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for History
if 'history' not in st.session_state:
    st.session_state.history = []

# Custom CSS for "Modern & Classy" look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #0e1117;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #262730;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
# --- LAYOUT: Title & History (Top Right) ---
col_header, col_hist = st.columns([3, 1])

with col_header:
    st.title(" AI Oil Spill Modeling & Ecological Impact")
    st.markdown("---")
    st.markdown("## Advanced Simulation Dashboard")
    st.markdown("### This is a demo of the AI Oil Spill Modeling & Ecological Impact you can run it and choose your own parameters on the sidebar")
    st.markdown("---")

with col_hist:
    # History Section moved here
    if len(st.session_state.history) > 0:
        st.markdown("#### 📜 History")
        # Create a list of labels for the selectbox
        history_labels = [f"{entry['Timestamp']} - Rec: {entry['Recovery Idx']:.3f}" for entry in st.session_state.history]
        history_labels.reverse()
        
        selected_run_label = st.selectbox("Select Past Run", ["Current Run"] + history_labels, key="hist_select")
        
        if selected_run_label != "Current Run":
            selected_idx = history_labels.index(selected_run_label)
            original_idx = len(st.session_state.history) - 1 - selected_idx
            st.session_state.selected_run = st.session_state.history[original_idx]
            if st.button("Load Analysis"):
                st.rerun()
    else:
        st.caption("No history yet.")

# Sidebar - Configuration
st.sidebar.header("⚙️ Simulation Parameters")
st.sidebar.markdown("---")

with st.sidebar.expander("1. Physics & Data Generation", expanded=True):
    num_sequences = st.slider("Number of Sequences", 10, 100, 20)
    t_total = st.slider("Time Steps per Sequence", 10, 30, 15)

with st.sidebar.expander("2. AI Model Training", expanded=True):
    num_epochs = st.slider("Training Epochs", 1, 50, 10)
    learning_rate = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)

with st.sidebar.expander("3. Biology & Chemistry", expanded=True):
    k_consume = st.slider("DO Consumption Rate (k_consume)", 0.01, 0.5, 0.07, help="Scientific default: ~0.07")
    k_reaer = st.slider("Reaeration Rate (k_reaer)", 0.001, 1.0, 0.4, help="Scientific default: ~0.4")
    oil_dampening = st.slider("Oil Dampening Factor", 0.0, 1.0, 0.2, help="Reduction in reaeration due to slick")
    lc50_zoo = st.slider("Zooplankton LC50 (mg/L)", 1.0, 100.0, 30.0, help="Lethal Concentration 50%")

# Helper function for data loading
def prepare_data(features, T_in=4, T_out=1):
    # features: (N, T, C, H, W)
    X_list, y_list = [], []
    N, T, C, H, W = features.shape
    
    for n in range(N):
        for t in range(T - T_in - T_out + 1):
            x = features[n, t : t + T_in] # (T_in, C, H, W)
            y = features[n, t + T_in : t + T_in + T_out, 0] # (T_out, H, W) - predict oil only
            X_list.append(x)
            y_list.append(y)
            
    X = np.stack(X_list) # (B_total, T_in, C, H, W)
    y = np.stack(y_list) # (B_total, T_out, H, W)
    
    # Add channel dim to y
    y = np.expand_dims(y, axis=2) # (B_total, T_out, 1, H, W)
    
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()

def plot_interactive_heatmap(data, title, colorscale, zmin=None, zmax=None):
    """
    Creates an interactive Plotly heatmap.
    """
    fig = px.imshow(
        data, 
        color_continuous_scale=colorscale,
        title=title,
        zmin=zmin,
        zmax=zmax,
        labels=dict(x="X Coordinate", y="Y Coordinate", color="Value")
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=350,
        xaxis_showticklabels=False,
        yaxis_showticklabels=False
    )
    return fig

# Main Execution
if st.sidebar.button("🚀 RUN SIMULATION"):
    
    # --- STEP 1: PHYSICS ---
    st.markdown("#### 1. Physics Simulation and AI Training")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Generating synthetic physics data...")
    
    # Generate Data
    features = generate_synthetic_dataset(
        num_sequences=num_sequences, 
        t_total=t_total, 
        H=64, 
        W=64
    )
    
    # Save locally (optional, but good for caching)
    data_dir = os.path.join("data", "processed")
    os.makedirs(data_dir, exist_ok=True)
    np.savez_compressed(os.path.join(data_dir, "demo_sequences.npz"), features=features)
    
    progress_bar.progress(20)
    st.success(f"✅ Generated {num_sequences} sequences of {t_total} time steps.")

    # --- STEP 2: AI TRAINING ---
    
    
    # Prepare Data
    status_text.text("Preparing tensors for training...")
    X_train, y_train = prepare_data(features, T_in=4, T_out=1)
    
    # Split (Simple 80/20)
    split_idx = int(0.8 * len(X_train))
    X_tr, y_tr = X_train[:split_idx], y_train[:split_idx]
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]
    
    # Dataset & Loader
    batch_size = 8
    train_ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Model Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvLSTMPredictor(input_channels=3, hidden_channels=32, num_layers=2).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    status_text.text(f"Training on {device} for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            
        avg_loss = epoch_loss / len(train_ds)
        
        # Update Progress
        current_progress = 20 + int(60 * (epoch + 1) / num_epochs)
        progress_bar.progress(current_progress)
        status_text.text(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")
        
    st.success("✅ AI Model Trained Successfully")
    
    # Save Model
    torch.save(model.state_dict(), "conv_lstm_predictor.pth")

    # --- STEP 3: PREDICTION & BIO ---
    st.markdown("#### 3. Prediction & Ecological Impact")
    status_text.text("Running final prediction and biological analysis...")
    
    # Take a test sample (last one from validation set)
    test_idx = -1
    input_seq = X_val[test_idx].unsqueeze(0).to(device) # (1, 4, 3, 64, 64)
    true_future = y_val[test_idx].squeeze().numpy() # (64, 64)
    
    model.eval()
    with torch.no_grad():
        pred_tensor = model(input_seq)
        pred_oil = pred_tensor.squeeze().cpu().numpy() # (64, 64)
        
    # Bio Analysis
    H, W = pred_oil.shape
    DO_sat = 8.0
    DO_current = np.full((H, W), DO_sat)
    Plankton_ref = 100.0
    Plankton_current = np.full((H, W), Plankton_ref)
    dt = 3600 # 1 hour
    
    # 1. Update DO with Oil Dampening
    DO_next = update_DO(
        DO_current, 
        DO_sat, 
        pred_oil * 10, # Scaling factor for demo
        k_consume=k_consume, 
        k_reaer=k_reaer, 
        dt=dt,
        oil_dampening=oil_dampening
    )
    
    # 2. Plankton Response with LC50
    Plankton_next = plankton_response(
        Plankton_current, 
        pred_oil * 10, 
        dt=dt,
        lc50=lc50_zoo
    )
    
    # 3. Toxicity Check
    toxicity_flags = check_toxicity_thresholds(pred_oil * 10, thresholds={'zoo_lc50': lc50_zoo, 'phyto_inhibit': 100.0})
    toxic_zone_mask = toxicity_flags['zoo_lc50'].astype(int)
    
    # 4. Recovery Index
    # Assume Benthos is unaffected for this short time
    Benthos_ref = 1.0
    Benthos_curr = 1.0
    recovery_index = ecological_recovery_index(
        DO_next, DO_sat,
        Plankton_next, Plankton_ref,
        Benthos_curr, Benthos_ref
    )
    
    # 5. Accuracy Metrics
    # Compare pred_oil (AI) with true_future (Physics)
    acc_metrics = calculate_metrics(true_future, pred_oil)
    
    # --- REPORTING METRICS ---
    max_oil_idx = np.unravel_index(np.argmax(pred_oil, axis=None), pred_oil.shape)
    most_impacted_zone = f"({max_oil_idx[1]}, {max_oil_idx[0]})" # (x, y)
    
    avg_do = np.mean(DO_next)
    min_do = np.min(DO_next)
    plankton_surv = np.mean(Plankton_next)/Plankton_ref*100
    avg_recovery = np.mean(recovery_index)
    toxic_area_percent = (np.sum(toxic_zone_mask) / toxic_zone_mask.size) * 100
    
    # --- GENERATE REPORT CONTENT (Text) ---
    report_text_md = f"""
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

#### 1. Simulation Parameters
- **DO Consumption Rate:** {k_consume}
- **Reaeration Rate:** {k_reaer}
- **Oil Dampening:** {oil_dampening}
- **Zooplankton LC50:** {lc50_zoo} mg/L

#### 2. Key Ecological Metrics
- **Avg Recovery Index:** {avg_recovery:.3f} ({'✅ Good' if avg_recovery > 0.8 else '⚠️ Critical'})
- **Plankton Survival:** {plankton_surv:.1f}%
- **Toxic Area (LC50):** {toxic_area_percent:.1f}%
- **Min Dissolved Oxygen:** {min_do:.2f} mg/L
- **Most Impacted Zone:** {most_impacted_zone}

#### 3. Prediction Accuracy (AI vs Physics)
- **SSIM (Structural Similarity):** {acc_metrics['SSIM']:.4f} (1.0 = Perfect Match)
- **MSE (Mean Squared Error):** {acc_metrics['MSE']:.6f}
- **PSNR (Peak Signal-to-Noise):** {acc_metrics['PSNR']:.2f} dB

#### 4. Analysis Summary
The simulation predicts a maximum oil concentration of **{pred_oil.max():.4f}**. 
The ecological recovery index indicates the overall health of the ecosystem is at **{avg_recovery:.2f}**.
    """

    temp_dir = "temp_reports"
    os.makedirs(temp_dir, exist_ok=True)
    
    def save_heatmap_img(data, name, cmap, vmin=None, vmax=None):
        plt.figure(figsize=(6, 5))
        plt.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(name)
        plt.axis('off')
        path = os.path.join(temp_dir, f"{name.lower().replace(' ', '_')}.png")
        plt.savefig(path, bbox_inches='tight', dpi=100)
        plt.close()
        return path

    # Save key visualizations
    img_pred = save_heatmap_img(pred_oil, "AI Prediction", "inferno", 0, 1)
    img_do = save_heatmap_img(DO_next, "Dissolved Oxygen", "Blues_r", 0, 8)
    img_rec = save_heatmap_img(recovery_index, "Recovery Index", "RdYlGn", 0, 1)
    img_tox = save_heatmap_img(toxic_zone_mask, "Toxicity Zone", "Reds", 0, 1)

    # --- GENERATE PDF REPORT ---
    class PDFReport(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Ecological Impact Report', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Text Content
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Simulation Parameters", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"- DO Consumption Rate: {k_consume}", 0, 1)
    pdf.cell(0, 10, f"- Reaeration Rate: {k_reaer}", 0, 1)
    pdf.cell(0, 10, f"- Oil Dampening: {oil_dampening}", 0, 1)
    pdf.cell(0, 10, f"- Zooplankton LC50: {lc50_zoo} mg/L", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Key Ecological Metrics", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Avg Recovery Index: {avg_recovery:.3f} ({'Good' if avg_recovery > 0.8 else 'Critical'})", 0, 1)
    pdf.cell(0, 10, f"Plankton Survival: {plankton_surv:.1f}%", 0, 1)
    pdf.cell(0, 10, f"Toxic Area (LC50): {toxic_area_percent:.1f}%", 0, 1)
    pdf.cell(0, 10, f"Min Dissolved Oxygen: {min_do:.2f} mg/L", 0, 1)
    pdf.cell(0, 10, f"Most Impacted Zone: {most_impacted_zone}", 0, 1)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "3. Prediction Accuracy", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"SSIM: {acc_metrics['SSIM']:.4f}", 0, 1)
    pdf.cell(0, 10, f"MSE: {acc_metrics['MSE']:.6f}", 0, 1)
    pdf.cell(0, 10, f"PSNR: {acc_metrics['PSNR']:.2f} dB", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "4. Analysis Summary", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"The simulation predicts a maximum oil concentration of {pred_oil.max():.4f}. The ecological recovery index indicates the overall health of the ecosystem is at {avg_recovery:.2f}.")
    pdf.ln(10)

    # Images Section
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "5. Visual Maps", 0, 1)
    
    # Layout: 2 images per page
    y_start = pdf.get_y()
    pdf.image(img_pred, x=10, y=y_start, w=90)
    pdf.image(img_do, x=110, y=y_start, w=90)
    pdf.ln(80) # Space for images
    
    y_start = pdf.get_y()
    pdf.image(img_rec, x=10, y=y_start, w=90)
    pdf.image(img_tox, x=110, y=y_start, w=90)
    
    # Save PDF
    report_path = os.path.join("reports", "demo_report.pdf")
    pdf.output(report_path)
    
    # Read PDF for download
    with open(report_path, "rb") as f:
        pdf_data = f.read()
    
    progress_bar.progress(100)
    status_text.text("Simulation Complete!")
    
    # --- VISUALIZATION ---
    st.markdown("---")
    st.subheader(" Final Results")
    
    # Metrics in Columns
    col1, col2, col3, col4 = st.columns(4)
    
    max_oil = pred_oil.max()
    avg_do = np.mean(DO_next)
    plankton_surv = np.mean(Plankton_next)/Plankton_ref*100
    avg_recovery = np.mean(recovery_index)

    with col1:
        st.metric("Max Oil Conc.", f"{pred_oil.max():.4f}")
        st.caption(f"Impact Zone: {most_impacted_zone}")
    with col2:
        st.metric("Avg Dissolved Oxygen", f"{avg_do:.2f} mg/L", delta=f"{avg_do-DO_sat:.2f}")
        st.caption(f"Min DO: {min_do:.2f} mg/L")
    with col3:
        st.metric("Plankton Survival", f"{plankton_surv:.1f}%")
        st.caption(f"Toxic Area: {toxic_area_percent:.1f}%")
    with col4:
        st.metric("Recovery Index", f"{avg_recovery:.3f}", delta_color="normal" if avg_recovery > 0.8 else "inverse")

    # Accuracy Metrics Row
    st.markdown("##### 🎯 Model Accuracy")
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.metric("SSIM (Similarity)", f"{acc_metrics['SSIM']:.4f}", help="Structural Similarity Index (1.0 is perfect)")
    with col_a2:
        st.metric("MSE (Error)", f"{acc_metrics['MSE']:.6f}", help="Mean Squared Error (Lower is better)")
    with col_a3:
        st.metric("PSNR", f"{acc_metrics['PSNR']:.2f} dB", help="Peak Signal-to-Noise Ratio (Higher is better)")
        
    # Download Report Button
    st.download_button(
        label="📥 Download Ecological Report (PDF)",
        data=pdf_data,
        file_name="demo_report.pdf",
        mime="application/pdf"
    )

    # Visual Report with Plotly
    st.subheader("Interactive Visual Analysis")
    st.info(" Tip: Hover over the maps to see exact values. Zoom and pan to inspect details.")
    
    # Row 1: Input and Truth
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        # Input (Last Frame)
        input_img = input_seq[0, -1, 0].cpu().numpy()
        fig_input = plot_interactive_heatmap(input_img, "Input (Last Frame)", "inferno", zmin=0, zmax=1)
        st.plotly_chart(fig_input, use_container_width=True)
        st.caption("**Input:** The last observed state of the oil spill before prediction.")
        
    with col_v2:
        # True Future
        fig_true = plot_interactive_heatmap(true_future, "True Future (Physics)", "inferno", zmin=0, zmax=1)
        st.plotly_chart(fig_true, use_container_width=True)
        st.caption("**Ground Truth:** The actual physics simulation result for validation.")
        
    # Row 2: Prediction and Impact
    col_v3, col_v4, col_v5 = st.columns(3)
    
    with col_v3:
        # AI Prediction
        fig_pred = plot_interactive_heatmap(pred_oil, "AI Prediction", "inferno", zmin=0, zmax=1)
        st.plotly_chart(fig_pred, use_container_width=True)
        st.caption("**AI Prediction:** The forecasted oil distribution generated by the ConvLSTM model.")
        
    with col_v4:
        # DO Levels
        fig_do = plot_interactive_heatmap(DO_next, "Dissolved Oxygen (mg/L)", "Blues_r", zmin=0, zmax=8)
        st.plotly_chart(fig_do, use_container_width=True)
        st.caption("**Dissolved Oxygen:** Oxygen levels available for marine life. Darker blue is better.")
        
    with col_v5:
        # Recovery Index
        fig_rec = plot_interactive_heatmap(recovery_index, "Recovery Index", "RdYlGn", zmin=0, zmax=1)
        st.plotly_chart(fig_rec, use_container_width=True)
        st.caption("**Recovery Index:** Overall ecosystem health score (0=Dead, 1=Healthy).")
        
    # Row 3: Plankton & Toxicity
    col_v6, col_v7 = st.columns(2)
    
    with col_v6:
        # Plankton Biomass
        fig_plank = plot_interactive_heatmap(Plankton_next, "Plankton Biomass", "Viridis", zmin=0, zmax=Plankton_ref)
        st.plotly_chart(fig_plank, use_container_width=True)
        st.caption("**Plankton Biomass:** Remaining plankton population density.")
        
    with col_v7:
        # Toxicity Map
        fig_tox = plot_interactive_heatmap(toxic_zone_mask, "Lethal Toxicity Zone (LC50 Exceeded)", "Reds", zmin=0, zmax=1)
        st.plotly_chart(fig_tox, use_container_width=True)
        st.caption("**Toxicity Zone:** Red areas indicate where oil concentration exceeds the lethal threshold (LC50).")
    
    # --- REPORT TEXT ON DASHBOARD ---
    st.markdown("---")
    st.subheader("📄 Analysis Report")
    st.markdown(report_text_md)
    
    st.success("Analysis Completed. See details above.")
    
    # --- SAVE HISTORY ---
    # Store all necessary data to reproduce the view
    run_data = {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Epochs": num_epochs,
        "LR": learning_rate,
        "k_consume": k_consume,
        "Max Oil": max_oil,
        "Avg DO": avg_do,
        "Avg DO": avg_do,
        "Recovery Idx": avg_recovery,
        "SSIM": acc_metrics['SSIM'],
        # Store figure data objects or the data itself to re-plot
        "plot_data": {
            "input_img": input_img,
            "true_future": true_future,
            "pred_oil": pred_oil,
            "DO_next": DO_next,
            "recovery_index": recovery_index
        }
    }
    st.session_state.history.append(run_data)

# --- HISTORY SIDEBAR REMOVED (Moved to Top) ---

# --- DISPLAY SELECTED RUN (IF ANY) ---
if 'selected_run' in st.session_state and st.session_state.selected_run:
    entry = st.session_state.selected_run
    st.markdown("---")
    st.markdown(f"### 🔙 Analysis Replay: {entry['Timestamp']}")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max Oil Conc.", f"{entry['Max Oil']:.4f}")
    with col2:
        st.metric("Avg Dissolved Oxygen", f"{entry['Avg DO']:.2f} mg/L")
    with col3:
        st.metric("Recovery Index", f"{entry['Recovery Idx']:.3f}")
    with col4:
        st.caption(f"Params: Epochs={entry['Epochs']}, LR={entry['LR']}, k={entry['k_consume']}")
        if 'SSIM' in entry:
            st.caption(f"SSIM: {entry['SSIM']:.4f}")

    # Plots
    pdata = entry['plot_data']
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_input = plot_interactive_heatmap(pdata['input_img'], "Input (Last Frame)", "inferno", zmin=0, zmax=1)
        st.plotly_chart(fig_input, use_container_width=True)
    with col_v2:
        fig_true = plot_interactive_heatmap(pdata['true_future'], "True Future", "inferno", zmin=0, zmax=1)
        st.plotly_chart(fig_true, use_container_width=True)
        
    col_v3, col_v4, col_v5 = st.columns(3)
    with col_v3:
        fig_pred = plot_interactive_heatmap(pdata['pred_oil'], "AI Prediction", "inferno", zmin=0, zmax=1)
        st.plotly_chart(fig_pred, use_container_width=True)
    with col_v4:
        fig_do = plot_interactive_heatmap(pdata['DO_next'], "Dissolved Oxygen", "Blues_r", zmin=0, zmax=8)
        st.plotly_chart(fig_do, use_container_width=True)
    with col_v5:
        fig_rec = plot_interactive_heatmap(pdata['recovery_index'], "Recovery Index", "RdYlGn", zmin=0, zmax=1)
        st.plotly_chart(fig_rec, use_container_width=True)
        
    if st.button("Close Replay"):
        del st.session_state.selected_run
        st.rerun()
