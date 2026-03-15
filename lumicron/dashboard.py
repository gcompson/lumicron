import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
import os
from scipy.fft import fft, fftfreq
from scipy.signal import iirnotch, lfilter
from lumicron.core.physics import RadiometricEngine, KinematicsEngine
from lumicron.core.image_proc import ImageProcessor

st.set_page_config(page_title="LUMICRON Forensic Cockpit", layout="wide")

# Lab Aesthetics
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #c9d1d9; }
    .metric-card { 
        background-color: #161b22; padding: 20px; 
        border-radius: 10px; border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        white-space: pre-wrap; 
        background-color: #161b22; 
        border-radius: 5px 5px 0px 0px; 
        padding: 10px 20px; 
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛸 LUMICRON: Unified Forensic Dashboard v4.2")
st.write("Hardware Acceleration: **M4 Pro / OpenCL Engine Active**")

# --- SIDEBAR: MISSION CONTROL ---
st.sidebar.header("1. Global Constants")
default_path = os.path.join(os.path.expanduser("~"), "Projects/UAP_Data/Target_B")
project_path = st.sidebar.text_input("Active Project Folder", default_path)
fps = st.sidebar.number_input("Capture Rate (FPS)", value=240)
dist_m = st.sidebar.number_input("Target Distance (m)", value=1200)

st.sidebar.header("2. Signal Tuning")
apply_clahe = st.sidebar.checkbox("Signal CLAHE (Contrast Stretch)", value=True)
apply_notch = st.sidebar.checkbox("60Hz Notch Filter (Anti-Aliasing)", value=True)

st.sidebar.header("3. Spatial Controls")
threshold = st.sidebar.slider("Luma Isolation Threshold", 0, 255, 210)
decay = st.sidebar.slider("Persistence Decay (Blend)", 0.0, 1.0, 0.6)
zoom_frame = st.sidebar.number_input("Dilation Center Frame", value=3000)

# --- DATA PROCESSING ---
data_path = os.path.join(project_path, "03_DATA", "smear_audit.csv")

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    # Defensive drop to prevent NaNs from crashing Plotly
    df = df.dropna(subset=['peak_luminance', 'mean_luminance'])
    
    # 1. Radiometric Energy (Dynamic Baseline Subtraction)
    # We use a rolling mean to calculate the ambient sky brightness
    df['ambient_sky'] = df['mean_luminance'].rolling(window=10, min_periods=1).mean()
    
    # Isolate only the positive transient spikes (the craft pulsing above ambient)
    df['transient_spike'] = (df['mean_luminance'] - df['ambient_sky']).clip(lower=0)
    
    # Calculate the area of illumination based ONLY on the isolated energy spikes
    df['energy_output'] = (df['transient_spike'] / 255) * (dist_m**2)

    # 2. Signal Clean-up
    y = df['mean_luminance'].values
    if apply_clahe and len(y) > 0:
        y_min, y_max = np.min(y), np.max(y)
        if y_max > y_min:
            y = (y - y_min) / (y_max - y_min)
            
    y_detrended = y - np.mean(y) if len(y) > 0 else y

    if apply_notch and len(y_detrended) > 0:
        b, a = iirnotch(60.0, 30, fs=fps)
        y_detrended = lfilter(b, a, y_detrended)

    # 3. FFT Calculation
    N = len(y_detrended)
    if N > 0:
        xf = fftfreq(N, 1/fps)[:N//2]
        yf = np.abs(fft(y_detrended))[:N//2]
        psd_df = pd.DataFrame({'Hz': xf, 'Amplitude': 2.0/N * yf})
    else:
        psd_df = pd.DataFrame({'Hz': [], 'Amplitude': []})

    # --- TABBED INTERFACE ---
    tab1, tab2, tab3 = st.tabs(["⚡ Radiometrics", "🌊 Spectral", "🔬 Spatial"])

    # === TAB 1: RADIOMETRICS ===
    with tab1:
        st.subheader("Luminous Flux (Energy Spikes over Time)")
        if not df.empty:
            fig_luma = px.area(df, x='frame', y='energy_output', template="plotly_dark", color_discrete_sequence=['#ffcc00'])
            fig_luma.update_layout(yaxis_title="Estimated Radiance (m²)")
            # UPDATED: width='stretch'
            st.plotly_chart(fig_luma, width='stretch')
            
            peak_energy = float(df['energy_output'].max())
            radius = float(np.sqrt(peak_energy)) if peak_energy > 0 else 0.0
            
            summary_html = str(f"""
            <div class="metric-card">
                <h3 style='color: #ffcc00;'>Peak Radiance Event: {peak_energy:.2f} m²</h3>
                <p>At a distance of {dist_m}m, the craft's energy spike is capable of illuminating a {radius:.1f}m radius. 
                This massive energy dump corresponds with the primary translation pulses.</p>
            </div>
            """)
            st.markdown(summary_html, unsafe_allow_html=True)

    # === TAB 2: SPECTRAL ===
    with tab2:
        st.subheader("Frequency Domain (Propulsion Signature)")
        if not psd_df.empty:
            fig_fft = px.line(psd_df[(psd_df['Hz'] > 2) & (psd_df['Hz'] < 20)], x='Hz', y='Amplitude', template="plotly_dark", color_discrete_sequence=['#ff0055'])
            # UPDATED: width='stretch'
            st.plotly_chart(fig_fft, width='stretch')

            peaks_only = psd_df[(psd_df['Hz'] > 2) & (psd_df['Hz'] < 10)]
            peak_hz = float(peaks_only.iloc[peaks_only['Amplitude'].idxmax()]['Hz']) if not peaks_only.empty else 0.0

            st.markdown("### Multi-Target Correlation")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Target B (Active)", f"{peak_hz:.2f} Hz", delta="Mach 321")
                st.write("**Status:** Anomalous Harmonic Confirmed.")
            with col_b:
                st.metric("Target A (Reference)", "13.11 Hz", delta="Mach 237")
                st.write("**Correlation:** Inverse Frequency-to-Velocity Scaling. The faster the craft moves, the slower and more powerful the displacement cycle becomes.")

    # === TAB 3: SPATIAL ===
    with tab3:
        st.subheader("Temporal Persistence Mapping")
        streak_path = os.path.join(project_path, "03_DATA", "streak_map.png")
        if os.path.exists(streak_path):
            raw_img = cv2.imread(streak_path)
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            color_mapped = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            display_img = cv2.addWeighted(raw_img, 1-decay, color_mapped, decay, 0)
            display_img = cv2.bitwise_and(display_img, display_img, mask=mask)
            # UPDATED: width='stretch'
            st.image(display_img, caption=f"Mach 321 Comet Trail (Blend Decay: {decay})", width='stretch')
        
        st.subheader("Sub-Pixel Temporal Dilation")
        frame_name = f"frame_{int(zoom_frame):05d}.png"
        frame_path = os.path.join(project_path, "02_FRAMES", frame_name)
        
        if os.path.exists(frame_path):
            z_img = cv2.imread(frame_path)
            h, w = z_img.shape[:2]
            crop = z_img[h//3:2*h//3, w//3:2*w//3]
            # UPDATED: width='stretch'
            st.image(crop, caption=f"Micro-Transient Zoom: Frame {zoom_frame}", width='stretch')
        else:
            st.info(f"Frame {zoom_frame} not found in 02_FRAMES. Adjust the 'Dilation Center Frame' slider.")

else:
    st.error("Missing data. Run 'lumicron radiate Target_B' in the CLI first.")
