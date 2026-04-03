import streamlit as st
from PIL import Image
from Multi-Modal-Satellite-Intelligence-System.inference import FusionPredictor

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

st.markdown("<h1 class='main-header'>🌍 Multi-Modal Satellite Intelligence System</h1>", unsafe_allow_html=True)
st.markdown("**Powered by Deep Learning Data Fusion (Image + Geo + Temporal Data)**")
st.markdown("Automated Land Classification, Vegetation Health Scoring, and Risk Assessment for ISRO Applications.")
st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.header("📥 Input Modalities")
    
    # 1. Image Modality
    st.subheader("1. Satellite Image (RGB)")
    uploaded_file = st.file_uploader("Upload an optical satellite image patch (.jpg, .png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Captured Satellite Patch', use_container_width=True)
    
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
    
    if st.button("Run Intelligence Engine 🚀"):
        if uploaded_file is None:
            st.warning("Please upload a satellite image first.")
        else:
            with st.spinner("Processing Modalities & Fusing Data..."):
                results = predictor.predict(image, lat, lon, month)
            
            st.success("Analysis Complete!")
            
            st.subheader("📊 Primary Results")
            
            rcol1, rcol2, rcol3 = st.columns(3)
            
            with rcol1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("Land Use Class")
                st.markdown(f"<p class='metric-value'>{results['Land Use Class']}</p>", unsafe_allow_html=True)
                st.markdown(f"Conf: {results['Confidence']*100:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with rcol2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("Vegetation Score")
                st.markdown(f"<p class='metric-value'>{results['Vegetation Score']:.2f}/1.0</p>", unsafe_allow_html=True)
                st.markdown("Index based on spectrum")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with rcol3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("Risk Indicator")
                risk_class = "metric-value-danger" if results['Risk Indicator'] == "High Risk" else "metric-value"
                st.markdown(f"<p class='{risk_class}'>{results['Risk Indicator']}</p>", unsafe_allow_html=True)
                st.markdown(f"Prob: {results['Risk Probability']*100:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)

            st.divider()
            st.write("### Model Architecture Summary")
            st.write("""
            - **Image Encoder**: Pretrained ResNet18 -> 512d Vector
            - **Metadata Encoder**: MLP (Lat, Lon, Sin(Month), Cos(Month)) -> 128d Vector
            - **Fusion**: Concatenation -> Dense Block (256 -> 128) -> Multi-task Heads
            """)
