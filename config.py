import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
DATA_DIR           = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR       = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR          = os.path.join(BASE_DIR, 'saved_models')
RESULTS_DIR        = os.path.join(BASE_DIR, 'results')
DEMO_DIR           = os.path.join(BASE_DIR, 'demo')

# ── Training Hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4
EPOCHS        = 10
NUM_WORKERS   = 4

# ── Loss Weights ──────────────────────────────────────────────────────────────
# These weights balance the three task losses:
#   total_loss = L_cls + LAMBDA_REGRESSION * L_reg + LAMBDA_RISK * L_risk
LAMBDA_REGRESSION = 1.0
LAMBDA_RISK       = 1.0

# ── Model Architecture ────────────────────────────────────────────────────────
NUM_CLASSES    = 10
META_INPUT_DIM = 5   # [lat_norm, lon_norm, sin_month, cos_month, ndvi_approx]
META_HIDDEN    = 128
IMAGE_FEAT_DIM = 512  # ResNet18 output dimension
FUSION_DIM     = 640  # IMAGE_FEAT_DIM + META_HIDDEN
DROPOUT_RATE   = 0.4

# ── EuroSAT Class Labels ──────────────────────────────────────────────────────
EUROSAT_CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

# Vegetation health scores per class (0=low, 1=high vegetation health)
# Used deterministically to generate auxiliary supervision signal.
# Note: These are proxy labels derived from domain knowledge about each land
# cover type — not measured from real NDVI sensor data.
CLASS_VEG_SCORES = {
    'AnnualCrop': 0.7, 'Forest': 0.95, 'HerbaceousVegetation': 0.85,
    'Highway': 0.05, 'Industrial': 0.05, 'Pasture': 0.75,
    'PermanentCrop': 0.80, 'Residential': 0.15, 'River': 0.30,
    'SeaLake': 0.10
}

# Binary environmental risk per class (1 = high risk / human impact)
CLASS_RISK_LABELS = {
    'AnnualCrop': 0, 'Forest': 0, 'HerbaceousVegetation': 0,
    'Highway': 1, 'Industrial': 1, 'Pasture': 0,
    'PermanentCrop': 0, 'Residential': 1, 'River': 0,
    'SeaLake': 0
}

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42