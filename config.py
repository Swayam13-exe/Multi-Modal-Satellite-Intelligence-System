import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
NUM_WORKERS = 4

# Loss Weights
LAMBDA_REGRESSION = 1.0
LAMBDA_RISK = 1.0

# Class maps
NUM_CLASSES = 10
EUROSAT_CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
    'River', 'SeaLake'
]

# Random seed for reproducibility
SEED = 42
