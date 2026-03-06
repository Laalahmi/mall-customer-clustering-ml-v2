from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.config import K_RANGE, RANDOM_STATE, N_INIT
from src.logger import get_logger

logger = get_logger(__name__)


def train_kmeans(X_scaled, k):
    """
    Train a KMeans model with k clusters.
    """
    try:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT
        )

        labels = model.fit_predict(X_scaled)

        inertia = model.inertia_
        silhouette = silhouette_score(X_scaled, labels)

        logger.info(f"K={k} | Inertia={inertia:.2f} | Silhouette={silhouette:.4f}")

        return model, labels, inertia, silhouette

    except Exception as exc:
        logger.exception("Error training KMeans model.")
        raise RuntimeError("Error training KMeans model.") from exc


def search_best_k(X_scaled):
    """
    Evaluate multiple K values and return the best model.
    """
    best_model = None
    best_labels = None
    best_k = None
    best_score = -1
    results = []

    logger.info("Searching for optimal K using silhouette score.")

    for k in K_RANGE:
        model, labels, inertia, silhouette = train_kmeans(X_scaled, k)

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette": silhouette
        })

        if silhouette > best_score:
            best_score = silhouette
            best_model = model
            best_labels = labels
            best_k = k

    logger.info(f"Best K selected: {best_k} (silhouette={best_score:.4f})")

    return best_model, best_labels, best_k, results