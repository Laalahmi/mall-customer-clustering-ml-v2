import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import CLUSTER_FEATURES
from src.logger import get_logger

logger = get_logger(__name__)


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select clustering features and ensure they are numeric.
    """
    try:
        X = df[CLUSTER_FEATURES].copy()

        for col in CLUSTER_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        missing_before = X.isnull().sum().sum()
        if missing_before > 0:
            logger.warning(
                f"Found {missing_before} missing values in clustering features. Dropping rows with missing values."
            )
            X = X.dropna()

        logger.info(f"Selected clustering features. Shape: {X.shape}")
        return X

    except KeyError as exc:
        logger.error("One or more clustering features are missing from dataset.")
        raise ValueError("One or more clustering features are missing from dataset.") from exc
    except Exception as exc:
        logger.exception("Unexpected error during feature selection.")
        raise RuntimeError("Unexpected error during feature selection.") from exc


def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    """
    Fit a StandardScaler on the selected features.
    """
    try:
        scaler = StandardScaler()
        scaler.fit(X)
        logger.info("Scaler fitted successfully.")
        return scaler
    except Exception as exc:
        logger.exception("Failed to fit scaler.")
        raise RuntimeError("Failed to fit scaler.") from exc


def transform_features(scaler: StandardScaler, X: pd.DataFrame):
    """
    Transform selected features using fitted scaler.
    """
    try:
        X_scaled = scaler.transform(X)
        logger.info("Features transformed successfully.")
        return X_scaled
    except Exception as exc:
        logger.exception("Failed to transform features.")
        raise RuntimeError("Failed to transform features.") from exc


def fit_transform_features(X: pd.DataFrame):
    """
    Fit scaler and transform features in one step.
    Returns:
        scaler, X_scaled
    """
    scaler = fit_scaler(X)
    X_scaled = transform_features(scaler, X)
    return scaler, X_scaled