import streamlit as st
from PIL import Image
import numpy as np
import torch
from inference import FusionPredictor

from utils.feature_engineering import approximate_ndvi, extract_meta_features
from utils.visualization import generate_vegetation_heatmap, overlay_heatmap
from utils.gradcam import generate_gradcam
from utils.temporal_analysis import compare_images, compute_change_mask, compute_ndvi_difference

# Streamlit Page Config
st.set_page_config(
    page_title="Multi-Modal Satellite Intelligence System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium UI
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #1E3A8A;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #10B981;
    }
    .metric-value-danger {
        font-size: 24px;
        font-weight: bold;
        color: #EF4444;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return FusionPredictor()

predictor = load_predictor()

st.sidebar.title("🌍 Intelligence Modes")
mode = st.sidebar.radio("Select Mode", ["Single Image Analysis", "Temporal Comparison (Change Detection)"])
st.sidebar.divider()
st.sidebar.info("Upload imagery from the data/raw/eurosat directory to test inference.")

st.markdown("<h1 class='main-header'>🌍 Multi-Modal Satellite Intelligence System</h1>", unsafe_allow_html=True)
st.markdown("**Powered by Deep Learning Data Fusion (Image + Geo + Temporal Data)**")

if mode == "Single Image Analysis":
    st.markdown("Automated Land Classification, Vegetation Health Scoring, Risk Assessment, and Explainable AI.")
    st.divider()

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.header("📥 Input Modalities")
        
        # 1. Image Modality
        st.subheader("1. Satellite Image (RGB)")
        uploaded_file = st.file_uploader("Upload an optical satellite image patch (.jpg, .png)", type=["jpg", "jpeg", "png"])
        
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Captured Satellite Patch', use_container_width=True)
            
            # Feature: NDVI Approximation
            img_tensor = predictor.transform(image)
            ndvi_val = approximate_ndvi(img_tensor)
            st.info(f"🌿 **Approximate NDVI Score:** `{ndvi_val:.3f}`  \n*(Derived locally from RGB balance)*")
        
        st.divider()
        
        # 2. Tabular Modalities
        st.subheader("2. Subject Metadata")
        col1, col2, col3 = st.columns(3)
        with col1:
            lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=20.59) # Default to India
        with col2:
            lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=78.96)
        with col3:
            month = st.selectbox("Month (Temporal)", options=list(range(1, 13)), index=4) # Default to May

    with col_right:
        st.header("🧠 Analysis Pipeline")
        st.info("The fusion model combines a ResNet18 CNN encoder with an MLP-based Tabular Encoder to extract contextual features.")
        
        show_heatmap = st.checkbox("Show Vegetation Heatmap", value=False)
        show_gradcam = st.checkbox("Show Explainability (Grad-CAM)", value=False)
        
        if st.button("Run Intelligence Engine 🚀"):
            if image is None:
                st.warning("Please upload a satellite image first.")
            else:
                with st.spinner("Processing Modalities & Fusing Data..."):
                    results = predictor.predict(image, lat, lon, month)
                
                st.success("Analysis Complete!")
                
                st.subheader("📊 Primary Results")
                
                rcol1, rcol2, rcol3 = st.columns(3)
                
                with rcol1:
                    st.markdown("<div class='metric-card'>Land Use Class<br>"
                                f"<span class='metric-value'>{results['Land Use Class']}</span><br>"
                                f"Conf: {results['Confidence']*100:.1f}%</div>", unsafe_allow_html=True)
                    
                with rcol2:
                    st.markdown("<div class='metric-card'>Vegetation Score<br>"
                                f"<span class='metric-value'>{results['Vegetation Score']:.2f}/1.0</span><br>"
                                "Deep Feature Index</div>", unsafe_allow_html=True)
                    
                with rcol3:
                    risk_class = "metric-value-danger" if results['Risk Indicator'] == "High Risk" else "metric-value"
                    st.markdown(f"<div class='metric-card'>Risk Indicator<br>"
                                f"<span class='{risk_class}'>{results['Risk Indicator']}</span><br>"
                                f"Prob: {results['Risk Probability']*100:.1f}%</div>", unsafe_allow_html=True)

                st.divider()
                st.write("### 👁️ Advanced Visualizations")
                
                vcol1, vcol2 = st.columns(2)
                
                with vcol1:
                    if show_heatmap:
                        st.markdown("**Vegetation Heatmap**")
                        # Use the prediction score as the intensity base
                        heat_np = generate_vegetation_heatmap(image.size, results['Vegetation Score'])
                        overlay1 = overlay_heatmap(image, heat_np, alpha=0.5)
                        st.image(overlay1, use_container_width=True, caption="Model vegetation overlay")
                        
                with vcol2:
                    if show_gradcam:
                        st.markdown("**Grad-CAM (CNN Feature Focus)**")
                        with st.spinner("Executing backward pass for Explainability..."):
                            raw_tensor = predictor.transform(image)
                            img_t = raw_tensor.unsqueeze(0).to(predictor.device)
                            meta_f = extract_meta_features(lat, lon, month, image_tensor=raw_tensor)
                            meta_t = torch.tensor(meta_f, dtype=torch.float32).unsqueeze(0).to(predictor.device)
                            
                            saliency = generate_gradcam(predictor.model, img_t, meta_t)
                            overlay2 = overlay_heatmap(image, saliency, alpha=0.45)
                            st.image(overlay2, use_container_width=True, caption="High emphasis regions in red")

elif mode == "Temporal Comparison (Change Detection)":
    st.markdown("Compare satellite patches from two different time periods to track urban development or deforestation.")
    st.divider()
    
    tcol1, tcol2 = st.columns(2)
    img1, img2 = None, None
    with tcol1:
        st.subheader("Time 1 (Before)")
        file1 = st.file_uploader("Upload Image 1", type=["jpg", "png"], key="f1")
        if file1:
             img1 = Image.open(file1).convert('RGB')
             st.image(img1, use_container_width=True)
    with tcol2:
        st.subheader("Time 2 (After)")
        file2 = st.file_uploader("Upload Image 2", type=["jpg", "png"], key="f2")
        if file2:
             img2 = Image.open(file2).convert('RGB')
             st.image(img2, use_container_width=True)
             
    if file1 and file2:
        st.divider()
        if st.button("🔍 Run Temporal Analysis"):
            with st.spinner("Calculating deltas and structural changes..."):
                diff_img = compare_images(img1, img2)
                change_mask, change_pct = compute_change_mask(diff_img, threshold=25)
                
                t1_ndvi = approximate_ndvi(predictor.transform(img1))
                t2_ndvi = approximate_ndvi(predictor.transform(img2))
                ndvi_diff = compute_ndvi_difference(t1_ndvi, t2_ndvi)
            
            st.success("Temporal Comparison Complete!")
            
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                 st.metric("Total Area Changed", f"{change_pct:.1f}%")
            with pcol2:
                 st.metric("Time 1 NDVI", f"{t1_ndvi:.3f}")
            with pcol3:
                 st.metric("Time 2 NDVI", f"{t2_ndvi:.3f}", delta=f"{ndvi_diff:.3f}")
                 
            st.divider()
            
            ccol1, ccol2 = st.columns(2)
            with ccol1:
                 st.markdown("**Absolute Difference Map**")
                 st.image(diff_img, use_container_width=True, caption="Pixel-wise discrepancies")
            with ccol2:
                 st.markdown("**Significant Change Highlight**")
                 # Overlay the mask over img2
                 overlay = Image.blend(img2, change_mask.convert('RGB'), alpha=0.6)
                 st.image(overlay, use_container_width=True, caption="Changed areas flagged in red")
