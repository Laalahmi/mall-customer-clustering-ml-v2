import pandas as pd
from src.config import DATA_FILE, REQUIRED_COLUMNS
from src.logger import get_logger

logger = get_logger(__name__)

COLUMN_RENAME_MAP = {
    "Customer_ID": "CustomerID",
    "Annual Income (k$)": "Annual_Income",
    "Spending Score (1-100)": "Spending_Score",
    "Genre": "Gender",
}


def load_data(file_path=DATA_FILE) -> pd.DataFrame:
    """
    Load dataset from CSV file and standardize known column names.
    """
    try:
        logger.info(f"Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)

        df = df.rename(columns=COLUMN_RENAME_MAP)

        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns found: {df.columns.tolist()}")
        return df

    except FileNotFoundError as exc:
        logger.error(f"Dataset file not found: {file_path}")
        raise FileNotFoundError(f"Dataset file not found: {file_path}") from exc
    except Exception as exc:
        logger.exception("Unexpected error while loading dataset.")
        raise RuntimeError("Unexpected error while loading dataset.") from exc


def validate_columns(df: pd.DataFrame) -> None:
    """
    Validate that required columns exist in dataset.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info("Dataset column validation passed.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - remove duplicates
    """
    logger.info("Starting dataset cleaning.")

    initial_shape = df.shape
    df = df.drop_duplicates().copy()

    logger.info(f"Dropped duplicates. Shape changed from {initial_shape} to {df.shape}")
    return df