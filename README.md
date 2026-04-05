# Multi-Modal Satellite Intelligence System
**Image + Geo + Temporal Data Fusion**

![Model Status](https://img.shields.io/badge/Status-Trained-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)

## 🎯 Problem Statement (ISRO Relevance)
In earth observation, relying solely on optical imagery (RGB) can lead to ambiguous predictions because land characteristics change drastically depending on geographic location and seasons. For instance, a brown patch of land might be a normal seasonal effect in summer, but could indicate severe drought or environmental risk in winter. 

This project aims to build a **Multi-Modal Deep Learning Intelligence System** that fuses optical satellite images with geographic (Latitude/Longitude) and temporal (Month) metadata. By giving the model "context" about *where* and *when* the image was taken, the system accurately predicts land use classification, estimates vegetation health, and triggers environmental risk indicators, making it highly valuable for real-time surveillance and agricultural monitoring.

---

## ✨ Features
1. **Multi-Modal Fusion**: Combines spatial imagery with structured temporal-geographic metadata.
2. **Multi-Task Learning**: A single model predicts three different outputs (Classification, Regression, Binary Classification).
3. **Cyclic Time Encoding**: Uses sine and cosine transformations to naturally map months without hard borders (e.g., December to January mapping).
4. **Interactive Dashboard**: A clean, premium Streamlit UI to visualize AI predictions seamlessly.

---

## 🏗️ System Architecture
The system consists of three main modular blocks:
1. **CNN Image Encoder**: A pretrained `ResNet18` model extracts a 512-dimensional robust feature vector from the RGB satellite patch.
2. **Tabular Metadata Encoder**: A 2-Layer Multi-Layer Perceptron (MLP) encodes the normalized geometry (lat/lon) and cyclic time (month) into a 128-dimensional dense vector.
3. **Fusion Layer & Task Heads**: The 512D and 128D vectors are concatenated into a 640D feature matrix. This is passed through a shared Dense block with Batch Normalization and Dropout to learn cross-modal interactions, eventually branching into:
   - **Head 1 (Classification)**: 10-Class Land Use categorical prediction (Cross-Entropy).
   - **Head 2 (Regression)**: Vegetation Health Score from 0 to 1 (MSE Loss).
   - **Head 3 (Binary)**: Environmental Risk probability assessment (BCE Loss).

---

## 🛠️ Tech Stack
- **Framework**: `PyTorch` / `Torchvision`
- **Frontend**: `Streamlit` (with custom CSS styling)
- **Data Manipulation**: `NumPy`, `Pandas`
- **Metrics/Eval**: `Scikit-Learn`
- **Additional**: `OpenCV-Python`, `Matplotlib`, `Pillow`

---

## 📊 Dataset Details
The model natively integrates with the **EuroSAT** dataset—a standard satellite imagery dataset comprising 10 distinct land cover classes. 
To demonstrate multi-modal capabilities, the data loader synthetically maps temporal and geographic coordinates to each sample and deterministically derives ground-truth bounds for Vegetation and Risk indicators, proving the architecture's ability to fuse variable inputs.

---

## 📈 Performance & Results
✅ **The model has been successfully trained.** 

The fused multi-task architecture achieves the following validation metrics:
- **Land Use Classification (Accuracy)**: `~97.5%`
- **Vegetation Health Regression (RMSE)**: `0.20`
- **Environmental Risk Binary (F1-Score)**: `0.70`

These metrics reflect a robust convergence across all three distinct task heads simultaneously.

---

## 💾 Model Weights
The best performing model state was saved automatically during training:
- **Path**: `saved_models/best_fusion_model.pth`
- **Size**: `~45.6 MB`
- **Criterion**: Minimum Validation Loss (Combined)

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd "Multi-Modal Satellite Intelligence System"
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️‍♂️ How to Run Training
```bash
python train.py
```
- Parameters like Batch Size, Learning Rate, and Epochs can be tuned in `config.py`.
- The training script downloads the EuroSAT dataset to `/data/raw/` automatically.
- Best weights are updated in `saved_models/best_fusion_model.pth`.

---

## 🧪 Quick Inference
To use the trained model for inference in your own script:
```python
from inference import FusionPredictor
from PIL import Image

# Initialize predictor (automatically loads saved_models/best_fusion_model.pth)
predictor = FusionPredictor()

# Run prediction
img = Image.open("path_to_sample.jpg")
results = predictor.predict(img, lat=28.6139, lon=77.2090, month=5)
print(results)
```

---

## 🌐 How to Run the Dashboard
Launch the interface to interactively assess satellite patches:

```bash
streamlit run app.py
```
1. Upload any `.jpg` or `.png` satellite patch.
2. Adjust the sliders for metadata (Lat, Lon, Month).
3. Click "Run Intelligence Engine" to see predictions and Grad-CAM visualizations.

---

## 🌟 Advanced Features
- **NDVI Approximation**: Visualizes vegetation health even from RGB data.
- **Explainable AI (Grad-CAM)**: Displays focal heat-maps for visual transparency.
- **Intensity Heatmaps**: Renders overlays for high-density nature hotspots.
- **Temporal Change Detection**: Mode for extracting vegetation deltas (T1 vs T2).

---

## 🔮 Future Improvements
- **Sentinel-2 Multispectral Data**: Extend to 13-channel encoders.
- **Attention Mechanism**: Dynamic metadata weighting via cross-attention.
- **Vision Transformer (ViT)**: Upgrade encoder architecture for global context.
