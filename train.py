from src.data_loader import load_data, validate_columns, clean_data
from src.preprocess import select_features, fit_transform_features
from src.clustering import search_best_k
from src.utils import create_cluster_summary
from src.artifacts import save_model_bundle
from src.config import CLUSTER_FEATURES
from src.logger import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Starting clustering training pipeline.")

    df = load_data()
    validate_columns(df)
    df = clean_data(df)

    X = select_features(df)
    scaler, X_scaled = fit_transform_features(X)

    model, labels, best_k, results = search_best_k(X_scaled)

    cluster_summary = create_cluster_summary(X, labels)

    logger.info(f"\nCluster Summary:\n{cluster_summary}")

    save_model_bundle(
        model=model,
        scaler=scaler,
        features=CLUSTER_FEATURES,
        cluster_summary=cluster_summary,
        best_k=best_k,
        search_results=results
    )

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()