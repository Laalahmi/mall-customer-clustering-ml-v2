import pandas as pd
from src.config import CLUSTER_FEATURES
from src.logger import get_logger

logger = get_logger(__name__)


def create_cluster_summary(df, labels):
    """
    Create summary statistics for each cluster.
    """
    try:
        df_clustered = df.copy()
        df_clustered["Cluster"] = labels

        summary = df_clustered.groupby("Cluster")[CLUSTER_FEATURES].mean().round(2)

        logger.info("Cluster summary created successfully.")

        return summary

    except Exception as exc:
        logger.exception("Failed to generate cluster summary.")
        raise RuntimeError("Failed to generate cluster summary.") from exc