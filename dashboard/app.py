"""
Fabric Defect Detection & Model Evaluation Dashboard.
Real-time camera inspection, model evaluation, and live defect alerts.
"""
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from dashboard.runtime import get_runtime

# Page Configuration
st.set_page_config(
    page_title="Fabric Defect Inspector AI",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism UI)
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1117 0%, #161b22 90%);
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Container */
    .header-box {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Stat Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(48, 54, 61, 0.9);
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 13px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Alert Banners */
    .alert-banner-danger {
        background: linear-gradient(90deg, rgba(218, 54, 51, 0.9) 0%, rgba(248, 81, 73, 0.95) 100%);
        color: #ffffff;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 15px;
        font-weight: 700;
        font-size: 19px;
        text-align: center;
        box-shadow: 0 0 25px rgba(248, 81, 73, 0.6);
        animation: pulse 1s infinite alternate;
    }
    @keyframes pulse {
        0% { transform: scale(1.0); }
        100% { transform: scale(1.015); }
    }
    
    .alert-banner-normal {
        background: rgba(35, 134, 54, 0.2);
        border: 1px solid rgba(46, 160, 67, 0.5);
        color: #3fb950;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 15px;
        font-weight: 600;
        font-size: 15px;
        text-align: center;
    }
    
    /* Video Box */
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid #30363d;
        background: #000;
    }
</style>
""", unsafe_allow_html=True)

# Audio alert generator
AUDIO_ALERT_HTML = """
<audio autoplay>
    <source src="data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU{data}" type="audio/wav">
</audio>
"""

# Access Runtime Singleton
runtime = get_runtime()

# ─────────────────── Sidebar Controls ────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Inspection Configuration")
    
    # Session Details
    fabric_id = st.text_input("Fabric / Roll ID", value="Roll-A-01")
    fabric_length = st.number_input("Fabric Length (Metres)", min_value=1.0, value=15.0, step=1.0)
    
    st.markdown("---")
    st.markdown("### 📹 Camera Feed")
    cam_option = st.selectbox(
        "Camera Source",
        options=[
            "Synthetic Simulation (Test Weave & Datasets)",
            "Webcam 0 (Default Laptop / USB Camera)",
            "Webcam 1 (Secondary Camera)",
            "Phone Stream / RTSP URL"
        ],
        index=0
    )
    
    source_val = "synthetic"
    if cam_option == "Webcam 0 (Default Laptop / USB Camera)":
        source_val = "0"
    elif cam_option == "Webcam 1 (Secondary Camera)":
        source_val = "1"
    elif cam_option == "Phone Stream / RTSP URL":
        source_val = st.text_input("Stream URL", value="http://192.168.1.10:8080/video")
        
    st.markdown("---")
    st.markdown("### 🧠 Model Selection")
    model_choice = st.selectbox(
        "Classifier Architecture",
        options=[
            "YOLO11n-cls (Recommended: 89.96% Acc, 1.58ms)",
            "YOLO11s-cls (Standard: 88.89% Acc, 2.17ms)",
            "YOLOv8s-cls (Prior Gen: 87.63% Acc, 1.54ms)",
            "Custom Weights Path"
        ],
        index=0
    )
    
    if "YOLO11n-cls" in model_choice:
        weights_path = "runs/classify/runs/cls_yolo11n/weights/best.pt"
    elif "YOLO11s-cls" in model_choice:
        weights_path = "runs/cls/weights/best.pt"
    elif "YOLOv8s-cls" in model_choice:
        weights_path = "runs/classify/runs/cls_yolov8s/weights/best.pt"
    else:
        weights_path = st.text_input("Weights File", value="models/fabric_classifier_best.pt")
        
    conf_thresh = st.slider("Detection Confidence Threshold", min_value=0.20, max_value=0.95, value=0.50, step=0.05)
    
    lighting_corr = st.selectbox(
        "Lighting Normalization",
        options=["clahe", "illumination_norm", "combined", "none"],
        index=0,
        help="Applies contrast/illumination correction before model inference"
    )
    
    beep_alert = st.checkbox("System Audio Beep on Defect", value=True)
    
    st.markdown("---")
    # Action buttons
    snap = runtime.snapshot()
    is_running = snap["running"]
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("▶ START", use_container_width=True, disabled=is_running, type="primary"):
            try:
                runtime.start_session(
                    fabric_id=fabric_id,
                    fabric_length_m=fabric_length,
                    source=source_val,
                    weights=weights_path,
                    mock_mode=(source_val == "synthetic"),
                    conf=conf_thresh,
                    lighting=lighting_corr,
                    beep=beep_alert
                )
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start: {e}")
                
    with col_s2:
        if st.button("⏹ STOP", use_container_width=True, disabled=not is_running):
            runtime.stop_session()
            st.rerun()

# ─────────────────── Main Header ─────────────────────────────────────────────
snap = runtime.snapshot()
is_running = snap["running"]
stats = snap["stats"]
alert = snap["alert"]
counts = snap["class_counts"]
total_defects = sum(counts.values())

st.markdown(f"""
<div class="header-box">
    <div>
        <h2 style="margin: 0; font-size: 24px; color: #58a6ff;">🧵 Autonomous Fabric Defect Inspection System</h2>
        <span style="color: #8b949e; font-size: 14px;">Edge Computer Vision • Real-Time Detection • RTX 3050 GPU Acceleration</span>
    </div>
    <div style="text-align: right;">
        <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: {'#2ea043' if is_running else '#8b949e'}; margin-right: 6px;"></span>
        <strong style="color: {'#3fb950' if is_running else '#8b949e'}; font-size: 15px;">{'SYSTEM INSPECTING' if is_running else 'SYSTEM STANDBY'}</strong>
        <div style="color: #8b949e; font-size: 12px;">Session: {snap.get('session_id') or 'None'}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────── Main Tabs ───────────────────────────────────────────────
tab_live, tab_eval, tab_history, tab_sessions = st.tabs([
    "📹 Live Camera & Alerts",
    "📊 Model Evaluation & Benchmarks",
    "📸 Defect Snapshots & Event Log",
    "📁 Session Records & Export"
])

# ─────────────────── Tab 1: Live Camera & Alerts ─────────────────────────────
with tab_live:
    # Fragment to auto-refresh live camera feed and alerts smoothly
    @st.fragment(run_every="0.25s")
    def render_live_inspection():
        s = runtime.snapshot()
        run_state = s["running"]
        stt = s["stats"]
        al = s["alert"]
        cts = s["class_counts"]
        t_def = sum(cts.values())
        
        # 1. Alert Banner
        if al["active"]:
            info = al["info"]
            cls_name = info.get("class", "DEFECT").upper()
            c_val = info.get("confidence", 0.0) * 100
            st.markdown(f"""
            <div class="alert-banner-danger">
                🚨 CRITICAL DEFECT DETECTED: {cls_name} ({c_val:.1f}%) — STOP MOVING FABRIC 🚨
            </div>
            """, unsafe_allow_html=True)
        elif run_state:
            st.markdown("""
            <div class="alert-banner-normal">
                🟢 SCANNING FABRIC FEED — SYSTEM NORMAL (NO DEFECTS IN CURRENT VIEW)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("System is in standby. Click 'START' in the sidebar to begin live inspection.")
            
        # 2. Defect Counters Row
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #f85149;">
                <div class="metric-label">Total Defects</div>
                <div class="metric-val" style="color: #f85149;">{t_def}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #da3633;">
                <div class="metric-label">Holes</div>
                <div class="metric-val">{cts.get('hole', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #d29922;">
                <div class="metric-label">Stains</div>
                <div class="metric-val">{cts.get('stain', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #e36209;">
                <div class="metric-label">Broken Threads</div>
                <div class="metric-val">{cts.get('broken_thread', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #bc8cff;">
                <div class="metric-label">Irregular Yarn</div>
                <div class="metric-val">{cts.get('irregular_yarn', 0)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # 3. Main Live Stream + Live Performance
        col_video, col_telemetry = st.columns([7, 3])
        
        with col_video:
            st.markdown("#### Live Inspection Footage")
            frame = s["frame"]
            if frame is not None and frame.size > 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_frame, channels="RGB", use_container_width=True)
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "WAITING FOR CAMERA STREAM...", (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
                st.image(placeholder, channels="BGR", use_container_width=True)
                
        with col_telemetry:
            st.markdown("#### Real-Time Telemetry")
            
            cam_fps = stt.get("camera_fps", 0.0)
            ai_fps = stt.get("inference_fps", 0.0)
            lat_ms = stt.get("inference_ms", 0.0)
            frames_proc = stt.get("frames_processed", 0)
            
            st.metric("Inference Latency", f"{lat_ms:.1f} ms", delta=f"{ai_fps:.1f} AI FPS")
            st.metric("Camera Capture Speed", f"{cam_fps:.1f} FPS")
            st.metric("Frames Processed", f"{frames_proc:,}")
            
            # Live Probability Distribution (for Classifiers)
            probs = s.get("class_probs", {})
            if probs:
                st.markdown("---")
                st.markdown("##### Live Class Probabilities")
                for c_name, p_val in sorted(probs.items(), key=lambda x: -x[1]):
                    col_bar, col_lbl = st.columns([7, 3])
                    with col_bar:
                        st.progress(min(1.0, max(0.0, p_val)))
                    with col_lbl:
                        st.caption(f"{c_name}: **{p_val*100:.1f}%**")

    render_live_inspection()

# ─────────────────── Tab 2: Model Evaluation & Benchmarks ────────────────────
with tab_eval:
    st.markdown("### 📊 Model Evaluation & Benchmarking Study")
    st.markdown("""
    To validate model efficacy rigorously for academic and industrial requirements, we evaluated our primary
    architecture against both lighter edge alternatives and previous generation models on **558 held-out test samples**.
    """)
    
    # Benchmark table
    benchmark_file = Path("logs/benchmark_results.txt")
    if benchmark_file.exists():
        st.code(benchmark_file.read_text(encoding="utf-8"), language="text")
    else:
        st.info("Running benchmarks will populate this section.")
        
    st.markdown("---")
    st.markdown("### 🏆 Architecture Trade-off Analysis")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #3fb950;">
            <h4 style="color: #3fb950; margin: 0;">🥇 YOLO11n-cls (Nano)</h4>
            <div style="font-size: 13px; color: #8b949e; margin-bottom: 10px;">Best Overall Model</div>
            <p>• <b>Top-1 Acc: 89.96%</b><br>
            • Top-5 Acc: 100.0%<br>
            • Parameters: <b>1.54M</b> (72% smaller)<br>
            • Model Size: <b>3.0 MB</b><br>
            • Latency: <b>1.58 ms / frame</b></p>
            <span style="color: #3fb950; font-size: 12px;"><b>Verdict:</b> Optimal for high-speed edge video feeds. Avoids weave overfitting.</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #58a6ff;">
            <h4 style="color: #58a6ff; margin: 0;">🥈 YOLO11s-cls (Small)</h4>
            <div style="font-size: 13px; color: #8b949e; margin-bottom: 10px;">High Capacity Baseline</div>
            <p>• <b>Top-1 Acc: 88.89%</b><br>
            • Top-5 Acc: 100.0%<br>
            • Parameters: 5.45M<br>
            • Model Size: 10.5 MB<br>
            • Latency: 2.17 ms / frame</p>
            <span style="color: #58a6ff; font-size: 12px;"><b>Verdict:</b> Rich feature representations; solid high-resolution performance.</span>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #8b949e;">
            <h4 style="color: #8b949e; margin: 0;">🥉 YOLOv8s-cls (YOLOv8)</h4>
            <div style="font-size: 13px; color: #8b949e; margin-bottom: 10px;">Prior Generation Benchmark</div>
            <p>• <b>Top-1 Acc: 87.63%</b><br>
            • Top-5 Acc: 100.0%<br>
            • Parameters: 5.09M<br>
            • Model Size: 9.8 MB<br>
            • Latency: 1.54 ms / frame</p>
            <span style="color: #8b949e; font-size: 12px;"><b>Verdict:</b> YOLO11 delivers a +1.3% to +2.3% accuracy increase over YOLOv8.</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────── Tab 3: Defect Snapshots & History ───────────────────────
with tab_history:
    st.markdown("### 📸 Live Defect Snapshot Gallery & Event History")
    
    events_df = runtime.logger.events_df(runtime.session_id)
    
    if not events_df.empty:
        # Gallery of snapshots
        snaps_with_img = events_df[events_df["snapshot_path"].str.len() > 0]
        if not snaps_with_img.empty:
            st.markdown("#### Captured Defect Evidence")
            gallery_cols = st.columns(4)
            for idx, (_, row) in enumerate(snaps_with_img.head(8).iterrows()):
                col = gallery_cols[idx % 4]
                img_p = row["snapshot_path"]
                if Path(img_p).exists():
                    col.image(img_p, caption=f"#{row['event_id']} {row['defect_class']} ({row['confidence']:.2f})")
        
        st.markdown("---")
        st.markdown("#### Event Audit Log")
        display_cols = ["event_id", "defect_class", "confidence", "first_seen", "frames_seen", "snapshot_path"]
        st.dataframe(events_df[display_cols], use_container_width=True)
    else:
        st.info("No defect events recorded in current session yet.")

# ─────────────────── Tab 4: Session Records & Export ─────────────────────────
with tab_sessions:
    st.markdown("### 📁 Historical Inspection Sessions")
    
    sess_df = runtime.logger.sessions_df()
    if not sess_df.empty:
        st.dataframe(sess_df, use_container_width=True)
        
        # CSV Export
        all_events_df = runtime.logger.events_df()
        if not all_events_df.empty:
            csv_data = all_events_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Defects CSV Report",
                data=csv_data,
                file_name=f"fabric_defects_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary"
            )
    else:
        st.info("No saved inspection sessions found in database.")
