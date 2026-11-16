"""Offline evaluation for the recommender.

The dataset has no user interaction logs, so quality is measured with
content-based proxies against baselines, not with user-satisfaction metrics:

* artist hit-rate@k: for a query track whose primary artist has other tracks,
  does the top-k contain another track by that artist?
* language precision@k: share of top-k with the query's language label.
* intra-list diversity: mean pairwise cosine distance inside the top-k.
* artist diversity: distinct primary artists / k.
* catalogue coverage: share of the catalogue appearing in any top-k list.
* K-Means model selection: silhouette (sampled), Davies-Bouldin, inertia.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score


def _list_diversity(Xn: np.ndarray, idx: np.ndarray) -> float:
    if len(idx) < 2:
        return 0.0
    S = Xn[idx] @ Xn[idx].T
    iu = np.triu_indices(len(idx), k=1)
    return float(np.mean(1.0 - S[iu]))


def evaluate_neighbors(engine, k: int = 10, n_queries: int = 2000, seed: int = 0) -> dict:
    """Compare cosine k-NN with random and popularity baselines on the same
    sampled queries."""
    rng = np.random.default_rng(seed)
    df, Xn = engine.df, engine.Xn
    n = len(df)
    artist = df["primary_artist"].to_numpy()
    lang = df["language"].fillna("unknown").to_numpy()
    pop = np.nan_to_num(df["popularity"].to_numpy(float))

    # Only queries whose artist has another track can score an artist hit.
    counts = df["primary_artist"].map(df["primary_artist"].value_counts()).to_numpy()
    eligible = np.flatnonzero((counts >= 2) & (artist != ""))
    queries = rng.choice(eligible, size=min(n_queries, len(eligible)), replace=False)
    top_pop = np.argsort(-pop, kind="stable")[: k + 1]

    def blank():
        return {"artist_hit": [], "lang_prec": [], "ild": [], "artist_div": [], "seen": set()}

    res = {"knn": blank(), "random": blank(), "popularity": blank()}
    for q in queries:
        lists = {
            "knn": engine.neighbors(int(q), k),
            "random": rng.choice(np.delete(np.arange(n), q), size=k, replace=False),
            "popularity": np.array([j for j in top_pop if j != q][:k]),
        }
        for name, idx in lists.items():
            r = res[name]
            r["artist_hit"].append(float(np.any(artist[idx] == artist[q])))
            r["lang_prec"].append(float(np.mean(lang[idx] == lang[q])))
            r["ild"].append(_list_diversity(Xn, idx))
            r["artist_div"].append(len(set(artist[idx])) / k)
            r["seen"].update(idx.tolist())

    out = {"k": k, "n_queries": int(len(queries)), "catalogue_size": int(n), "seed": seed}
    for name, r in res.items():
        out[name] = {
            "artist_hit_rate": float(np.mean(r["artist_hit"])),
            "language_precision": float(np.mean(r["lang_prec"])),
            "intra_list_diversity": float(np.mean(r["ild"])),
            "artist_diversity": float(np.mean(r["artist_div"])),
            "catalogue_coverage": len(r["seen"]) / n,
        }
    return out


def kmeans_selection(X: np.ndarray, k_values=range(2, 13), sample_size: int = 10000,
                     seed: int = 42) -> list[dict]:
    """Silhouette / Davies-Bouldin / inertia per k. Silhouette and DB are
    computed on a fixed random subsample (O(n^2) otherwise)."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit(X)
        labels = km.labels_
        rows.append({
            "k": k,
            "silhouette": float(silhouette_score(X[sub], labels[sub])),
            "davies_bouldin": float(davies_bouldin_score(X[sub], labels[sub])),
            "inertia": float(km.inertia_),
        })
    return rows
