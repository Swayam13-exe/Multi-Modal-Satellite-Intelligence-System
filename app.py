import os
import streamlit as st
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from inference import FusionPredictor
from utils.feature_engineering import approximate_ndvi, extract_meta_features
from utils.visualization import generate_vegetation_heatmap, overlay_heatmap
from utils.gradcam import generate_gradcam
from utils.temporal_analysis import compare_images, compute_change_mask, compute_ndvi_difference

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Modal Satellite Intelligence System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0;
    }
    .sub-header {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 0.2rem;
        margin-bottom: 1rem;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .metric-value { font-size: 22px; font-weight: 700; color: #10B981; }
    .metric-value-danger { font-size: 22px; font-weight: 700; color: #EF4444; }
    .metric-value-neutral { font-size: 22px; font-weight: 700; color: #3B82F6; }
    .metric-sub { font-size: 12px; color: #94a3b8; margin-top: 4px; }
    /* Architecture note */
    .arch-note {
        background: #EFF6FF;
        border-left: 3px solid #3B82F6;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        font-size: 13px;
        color: #1E40AF;
        margin: 10px 0;
    }
    /* Disclaimer */
    .disclaimer {
        background: #FFFBEB;
        border-left: 3px solid #F59E0B;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        font-size: 12px;
        color: #92400E;
        margin: 10px 0;
    }
    /* Run button */
    .stButton > button {
        width: 100%;
        background: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.55rem 1rem;
        font-size: 15px;
    }
    .stButton > button:hover { background: #1D4ED8; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    return FusionPredictor()


predictor = load_predictor()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🛰️ Navigation")
mode = st.sidebar.radio(
    "Select Mode",
    ["Single Image Analysis", "Temporal Change Detection", "Training Results"]
)
st.sidebar.divider()
st.sidebar.markdown("""
**About this system**

A multi-task deep learning pipeline that fuses:
- 🖼️ **RGB satellite patches** via ResNet18
- 📍 **Geographic coordinates** (lat/lon)
- 📅 **Temporal metadata** (acquisition month)

Built on the **EuroSAT** dataset for ISRO earth observation relevance.
""")

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("<p class='main-header'>🛰️ Multi-Modal Satellite Intelligence System</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Image · Geographic · Temporal Data Fusion  —  ResNet18 + MLP Encoder + Multi-Task Heads</p>", unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  MODE 1 — Single Image Analysis
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Single Image Analysis":
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.header("📥 Input Modalities")

        st.subheader("1. Satellite Image (RGB)")
        # Show demo images if available
        demo_dir = config.DEMO_DIR
        demo_options = []
        if os.path.isdir(demo_dir):
            demo_options = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        image = None
        input_src = st.radio("Image source", ["Upload your own", "Use demo image"], horizontal=True)

        if input_src == "Upload your own":
            uploaded_file = st.file_uploader(
                "Upload a satellite patch (.jpg, .png)", type=["jpg", "jpeg", "png"]
            )
            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
        else:
            if demo_options:
                selected_demo = st.selectbox("Select a demo image", demo_options)
                image = Image.open(os.path.join(demo_dir, selected_demo)).convert('RGB')
            else:
                st.info("No demo images found. Add `.jpg` files to the `demo/` folder.")

        if image is not None:
            st.image(image, caption='Input Satellite Patch', use_container_width=True)
            raw_tensor = predictor.transform(image)
            ndvi_val = approximate_ndvi(raw_tensor)
            ndvi_color = "🟢" if ndvi_val > 0.3 else "🟡" if ndvi_val > 0.1 else "🔴"
            st.info(f"{ndvi_color} **Approx. NDVI (RGB proxy):** `{ndvi_val:.3f}`  \n"
                    f"*Estimated from RGB channel balance — not a true multispectral NDVI.*")

        st.divider()
        st.subheader("2. Geographic & Temporal Metadata")
        col1, col2, col3 = st.columns(3)
        with col1:
            lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=20.59,
                                  help="Decimal degrees. India centre ≈ 20.59°N")
        with col2:
            lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=78.96,
                                  help="Decimal degrees. India centre ≈ 78.96°E")
        with col3:
            month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            month_name = st.selectbox("Acquisition Month", month_names, index=4)
            month = month_names.index(month_name) + 1

    with col_right:
        st.header("🧠 Analysis Pipeline")
        st.markdown("""
<div class='arch-note'>
<strong>Architecture:</strong> ResNet18 (512-D image features) + MLP Tabular Encoder (128-D metadata features)
→ 640-D Fusion Layer → 3 task heads: Land Classification · Vegetation Regression · Risk Detection
</div>
<div class='disclaimer'>
<strong>Note on auxiliary labels:</strong> Vegetation scores and risk labels are derived deterministically
from land cover class membership — they are proxy supervision signals, not real sensor measurements.
They demonstrate the multi-task fusion architecture's capability.
</div>
""", unsafe_allow_html=True)

        show_heatmap = st.checkbox("Show Vegetation Heatmap overlay", value=True)
        show_gradcam = st.checkbox("Show Grad-CAM explainability", value=True)

        if st.button("🚀 Run Intelligence Engine"):
            if image is None:
                st.warning("Please provide a satellite image first.")
            else:
                with st.spinner("Fusing modalities and running inference..."):
                    results = predictor.predict(image, lat, lon, month)

                st.success("✅ Analysis complete!")
                st.subheader("📊 Predictions")

                rcol1, rcol2, rcol3 = st.columns(3)
                with rcol1:
                    conf_pct = results['Confidence'] * 100
                    st.markdown(f"""
<div class='metric-card'>
  <div class='metric-label'>Land Use Class</div>
  <div class='metric-value'>{results['Land Use Class']}</div>
  <div class='metric-sub'>Confidence: {conf_pct:.1f}%</div>
</div>""", unsafe_allow_html=True)

                with rcol2:
                    veg_pct = results['Vegetation Score'] * 100
                    veg_color = "metric-value" if veg_pct > 50 else "metric-value-danger"
                    st.markdown(f"""
<div class='metric-card'>
  <div class='metric-label'>Vegetation Health</div>
  <div class='{veg_color}'>{results['Vegetation Score']:.2f} / 1.0</div>
  <div class='metric-sub'>Fusion feature index</div>
</div>""", unsafe_allow_html=True)

                with rcol3:
                    risk_color = "metric-value-danger" if results['Risk Indicator'] == "High Risk" else "metric-value"
                    risk_pct = results['Risk Probability'] * 100
                    st.markdown(f"""
<div class='metric-card'>
  <div class='metric-label'>Environmental Risk</div>
  <div class='{risk_color}'>{results['Risk Indicator']}</div>
  <div class='metric-sub'>Probability: {risk_pct:.1f}%</div>
</div>""", unsafe_allow_html=True)

                st.divider()
                if show_heatmap or show_gradcam:
                    st.subheader("👁️ Visualisations")
                    vcol1, vcol2 = st.columns(2)

                    if show_heatmap:
                        with vcol1:
                            st.markdown("**Vegetation Heatmap**")
                            heat_np = generate_vegetation_heatmap(image.size, results['Vegetation Score'])
                            overlay1 = overlay_heatmap(image, heat_np, alpha=0.5)
                            st.image(overlay1, use_container_width=True,
                                     caption=f"Vegetation density overlay (score={results['Vegetation Score']:.2f})")

                    if show_gradcam:
                        with vcol2:
                            st.markdown("**Grad-CAM — CNN Attention**")
                            with st.spinner("Computing gradient saliency..."):
                                raw_t = predictor.transform(image)
                                img_t = raw_t.unsqueeze(0).to(predictor.device)
                                meta_f = extract_meta_features(lat, lon, month, image_tensor=raw_t)
                                meta_t = torch.tensor(meta_f, dtype=torch.float32).unsqueeze(0).to(predictor.device)
                                saliency = generate_gradcam(predictor.model, img_t, meta_t)
                            overlay2 = overlay_heatmap(image, saliency, alpha=0.45)
                            st.image(overlay2, use_container_width=True,
                                     caption="Regions most influential for classification (red = high attention)")

# ══════════════════════════════════════════════════════════════════════════════
#  MODE 2 — Temporal Change Detection
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Temporal Change Detection":
    st.markdown("Upload two satellite patches from **different time periods** to detect land cover changes.")
    st.divider()

    tcol1, tcol2 = st.columns(2)
    img1, img2 = None, None

    with tcol1:
        st.subheader("🕐 Time 1 — Before")
        file1 = st.file_uploader("Upload image (T1)", type=["jpg", "png"], key="f1")
        if file1:
            img1 = Image.open(file1).convert('RGB')
            st.image(img1, use_container_width=True)

    with tcol2:
        st.subheader("🕑 Time 2 — After")
        file2 = st.file_uploader("Upload image (T2)", type=["jpg", "png"], key="f2")
        if file2:
            img2 = Image.open(file2).convert('RGB')
            st.image(img2, use_container_width=True)

    if img1 and img2:
        st.divider()
        if st.button("🔍 Run Temporal Analysis"):
            with st.spinner("Computing pixel-wise changes and NDVI delta..."):
                diff_img = compare_images(img1, img2)
                change_mask, change_pct = compute_change_mask(diff_img, threshold=25)
                t1_ndvi = approximate_ndvi(predictor.transform(img1))
                t2_ndvi = approximate_ndvi(predictor.transform(img2))
                ndvi_diff = compute_ndvi_difference(t1_ndvi, t2_ndvi)

            st.success("✅ Temporal comparison complete!")

            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            with pcol1:
                st.metric("Area Changed", f"{change_pct:.1f}%")
            with pcol2:
                st.metric("T1 NDVI (RGB proxy)", f"{t1_ndvi:.3f}")
            with pcol3:
                st.metric("T2 NDVI (RGB proxy)", f"{t2_ndvi:.3f}", delta=f"{ndvi_diff:+.3f}")
            with pcol4:
                trend = "🌱 Greening" if ndvi_diff > 0.02 else "🔥 Degrading" if ndvi_diff < -0.02 else "→ Stable"
                st.metric("Vegetation Trend", trend)

            st.divider()
            ccol1, ccol2 = st.columns(2)
            with ccol1:
                st.markdown("**Absolute Difference Map**")
                st.image(diff_img, use_container_width=True, caption="Pixel-wise intensity difference (T1 vs T2)")
            with ccol2:
                st.markdown("**Significant Change Mask**")
                overlay = Image.blend(img2.resize(change_mask.size), change_mask.convert('RGB'), alpha=0.6)
                st.image(overlay, use_container_width=True, caption="Changed pixels flagged in red (threshold=25)")

# ══════════════════════════════════════════════════════════════════════════════
#  MODE 3 — Training Results
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Training Results":
    st.header("📈 Model Training Results")
    results_dir = config.RESULTS_DIR

    curves_path = os.path.join(results_dir, 'training_curves.png')
    cm_path     = os.path.join(results_dir, 'confusion_matrix.png')

    if not os.path.exists(curves_path) and not os.path.exists(cm_path):
        st.info("No training results found yet. Run `python train.py` to generate them, then reload this page.")
    else:
        if os.path.exists(curves_path):
            st.subheader("Loss & Metric Curves")
            st.image(curves_path, use_container_width=True,
                     caption="Training vs Validation loss/metrics across epochs")
            st.divider()

        if os.path.exists(cm_path):
            st.subheader("Per-Class Evaluation")
            st.image(cm_path, use_container_width=True,
                     caption="Confusion matrix & per-class F1 score on validation set")

        # Show JSON report if available
        report_path = os.path.join(results_dir, 'classification_report.json')
        if os.path.exists(report_path):
            import json
            with open(report_path) as f:
                report = json.load(f)
            st.divider()
            st.subheader("Summary Metrics")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Overall Accuracy", f"{report.get('accuracy', 0)*100:.2f}%")
            with m2:
                st.metric("Macro Avg F1", f"{report['macro avg']['f1-score']:.4f}")
            with m3:
                st.metric("Weighted Avg F1", f"{report['weighted avg']['f1-score']:.4f}")