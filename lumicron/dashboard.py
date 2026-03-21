import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import json
import plotly.express as px

# --- 1. CORE CONFIGURATION ---
st.set_page_config(page_title="LUMICRON v4.4 | Forensic Master", layout="wide", initial_sidebar_state="expanded")

# --- 2. SIDEBAR & GLOBAL CONTROLS ---
with st.sidebar:
    st.title("🛸 LUMICRON")
    st.caption("Forensic Cockpit v4.4")
    
    home = os.path.expanduser("~")
    base_path = os.path.join(home, "Projects/UAP_Data")
    
    if os.path.exists(base_path):
        projects = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        current_p = st.session_state.get('active_project', projects[0])
        selected_project = st.selectbox("Active Target Dossier", projects, index=projects.index(current_p) if current_p in projects else 0)
        st.session_state.active_project = selected_project
    else:
        st.error("Data root not found.")
        st.stop()
        
    st.divider()
    if st.button("🔄 PURGE CACHE"):
        st.cache_data.clear()
        st.rerun()

# --- 3. DATA INGESTION ---
project_path = os.path.join(base_path, selected_project)
data_dir = os.path.join(project_path, "03_DATA")
# Logic: Use Stabilized frames if they exist, otherwise raw
stab_dir = os.path.join(project_path, "02_STABILIZED")
raw_dir = os.path.join(project_path, "02_FRAMES")
frames_dir = stab_dir if os.path.exists(stab_dir) and len(os.listdir(stab_dir)) > 0 else raw_dir

smear_data = pd.read_csv(os.path.join(data_dir, "smear_audit.csv")) if os.path.exists(os.path.join(data_dir, "smear_audit.csv")) else None
morph_data = json.load(open(os.path.join(data_dir, "morphology.json"))) if os.path.exists(os.path.join(data_dir, "morphology.json")) else {}
track_data = json.load(open(os.path.join(data_dir, "tracking.json"))) if os.path.exists(os.path.join(data_dir, "tracking.json")) else {}

files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
frames_count = len(files)

# --- 4. HUD METRICS ---
st.markdown(f"### 🛸 TARGET: `{selected_project}`")
if frames_dir == stab_dir:
    st.caption("🟢 ECC REGISTRATION ACTIVE (Stabilized Ground)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    v = smear_data['delta_flux'].max() if smear_data is not None and 'delta_flux' in smear_data.columns else "N/A"
    st.metric("Peak Delta Flux", f"{v:.2f}" if isinstance(v, (float, int)) else v)
with col2:
    s = morph_data.get('ssi', 'N/A')
    st.metric("SSI (Rigidity)", f"{s:.2f}" if isinstance(s, (float, int)) else s)
with col3:
    g = track_data.get('max_g', 'N/A')
    st.metric("Kinematics", f"{g} G" if g != "N/A" else "Pending")
with col4:
    st.metric("Timeline Depth", f"{frames_count} Frames")

st.divider()

# --- 5. MASTER TIMELINE SYNC ---
if frames_count > 0:
    # Auto-find the peak event
    auto_f = 1
    if smear_data is not None and 'delta_flux' in smear_data.columns:
        auto_f = int(smear_data.iloc[smear_data['delta_flux'].idxmax()]['frame'])

    c_s1, c_s2 = st.columns([4, 1])
    with c_s2: use_peak = st.toggle("Sync to Peak Event", value=True if selected_project == "Target_C" else False)
    with c_s1:
        f_display = st.slider("Master Temporal Sync", 1, frames_count, value=auto_f if use_peak else 1)
        f_idx = f_display - 1 # 0-based index for cv2
else:
    st.stop()

# --- 6. WORKSPACE TABS ---
tab_rad, tab_spatial, tab_persist = st.tabs(["📈 Telemetry", "🔍 Spatial Matrix", "☄️ Persistence"])

with tab_rad:
    if smear_data is not None:
        # Safe column check to prevent Target_B crashes
        cols = [c for c in ['delta_flux', 'luma'] if c in smear_data.columns]
        fig = px.line(smear_data, x="frame", y=cols, title="Baseline-Subtracted Radiometry")
        fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0), height=400)
        fig.add_vline(x=f_display, line_width=3, line_color="#FF00CC", annotation_text=f"F{f_display}")
        st.plotly_chart(fig, config={'displayModeBar': False, 'responsive': True})

with tab_spatial:
    with st.expander("🛠️ Enhancement Suite", expanded=True):
        e1, e2, e3, e4 = st.columns(4)
        clahe_on = e1.checkbox("Contrast Stretching (CLAHE)", value=True)
        zoom_on = e2.checkbox("4x Dilation Zoom", value=False)
        lookback = e3.slider("Lookback (Delta)", 1, 30, 5)
        noise_floor = e4.slider("Noise Floor", 0, 50, 15)

    col_l, col_r = st.columns(2)
    img = cv2.imread(os.path.join(frames_dir, files[f_idx]))
    ref_img = cv2.imread(os.path.join(frames_dir, files[max(0, f_idx - lookback)]))

    if img is not None:
        # 1. Visual Spectrum (Left)
        l_view = img.copy()
        if clahe_on:
            lab = cv2.cvtColor(l_view, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0).apply(l)
            l_view = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
        
        # 2. Illumination Delta (Right)
        if ref_img is not None:
            # We use absdiff and a threshold to kill sensor pixel-crawl
            diff = cv2.absdiff(img, ref_img)
            _, diff = cv2.threshold(diff, noise_floor, 255, cv2.THRESH_TOZERO)
            r_view = cv2.applyColorMap(cv2.convertScaleAbs(diff, alpha=8.0), cv2.COLORMAP_MAGMA)
        else:
            r_view = np.zeros_like(img)

        if zoom_on:
            for i, frame in enumerate([l_view, r_view]):
                h, w = frame.shape[:2]
                box = 200
                cropped = frame[h//2-box:h//2+box, w//2-box:w//2+box]
                resized = cv2.resize(cropped, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                if i == 0: l_view = resized
                else: r_view = resized

        col_l.image(cv2.cvtColor(l_view, cv2.COLOR_BGR2RGB), width="stretch", caption=f"Visual F{f_display}")
        col_r.image(cv2.cvtColor(r_view, cv2.COLOR_BGR2RGB), width="stretch", caption="BSD Delta (Magma)")

with tab_persist:
    streak_path = os.path.join(data_dir, "streak_map.png")
    if os.path.exists(streak_path):
        st.image(streak_path, width="stretch", caption="Streak Map Persistence")
    else:
        st.info("Run 'lumicron stack Target_C --mode diff' to visualize path.")
