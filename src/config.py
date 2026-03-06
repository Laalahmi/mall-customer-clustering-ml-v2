from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Files
DATA_FILE = DATA_DIR / "mall_customers.csv"
MODEL_BUNDLE_FILE = MODELS_DIR / "clustering_bundle.joblib"

# Dataset columns
REQUIRED_COLUMNS = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual_Income",
    "Spending_Score",
]

# Features for clustering
CLUSTER_FEATURES = ["Age", "Annual_Income", "Spending_Score"]

# KMeans settings
K_RANGE = range(2, 11)
RANDOM_STATE = 42
N_INIT = 10

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "project.log"