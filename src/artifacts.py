import joblib
from src.config import MODEL_BUNDLE_FILE
from src.logger import get_logger

logger = get_logger(__name__)


def save_model_bundle(model, scaler, features, cluster_summary, best_k, search_results):
    """
    Save model artifacts for deployment.
    """
    try:
        bundle = {
            "model": model,
            "scaler": scaler,
            "features": features,
            "cluster_summary": cluster_summary,
            "best_k": best_k,
            "search_results": search_results
        }

        joblib.dump(bundle, MODEL_BUNDLE_FILE)

        logger.info(f"Model bundle saved to {MODEL_BUNDLE_FILE}")

    except Exception as exc:
        logger.exception("Failed to save model bundle.")
        raise RuntimeError("Failed to save model bundle.") from exc


def load_model_bundle():
    """
    Load model artifacts.
    """
    try:
        bundle = joblib.load(MODEL_BUNDLE_FILE)
        logger.info("Model bundle loaded successfully.")
        return bundle
    except Exception as exc:
        logger.exception("Failed to load model bundle.")
        raise RuntimeError("Failed to load model bundle.") from exc