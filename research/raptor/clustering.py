"""Clustering: dimensionality reduction + GMM with BIC model selection.

Paper uses UMAP + GMM. To stay dependency-light we use TruncatedSVD + GMM,
which reproduces the same behavior on small/medium corpora.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture


def reduce_dim(embeddings: np.ndarray, n_components: int = 10) -> np.ndarray:
    n, d = embeddings.shape
    k = min(n_components, d - 1, n - 1)
    if k < 2:
        return embeddings
    svd = TruncatedSVD(n_components=k, random_state=42)
    return svd.fit_transform(embeddings)


def find_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 10,
    random_state: int = 42,
) -> int:
    n = len(embeddings)
    if n <= 2:
        return 1
    max_k = min(max_clusters, n // 2, n - 1)
    if max_k < 2:
        return 1
    reduced = reduce_dim(embeddings)
    best_k, best_bic = 1, float("inf")
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=3)
        try:
            gmm.fit(reduced)
            bic = gmm.bic(reduced)
        except Exception:
            continue
        if bic < best_bic:
            best_bic, best_k = bic, k
    return best_k


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Return cluster labels (len == n)."""
    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n <= 5:
        return np.zeros(n, dtype=int)
    if n_clusters is None:
        n_clusters = find_optimal_clusters(embeddings)
    n_clusters = max(1, min(n_clusters, n))
    if n_clusters == 1:
        return np.zeros(n, dtype=int)
    reduced = reduce_dim(embeddings)
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state, n_init=5)
    return gmm.fit_predict(reduced)
