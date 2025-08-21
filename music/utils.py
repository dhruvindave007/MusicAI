# utils.py
import re
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Audio features used for similarity
FEATURE_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence"
]

# Light, music-minded weights (can tweak later)
FEATURE_WEIGHTS = {
    "acousticness": 1.0,
    "danceability": 1.4,
    "energy": 1.4,
    "instrumentalness": 1.0,
    "liveness": 0.8,
    "loudness": 1.1,
    "speechiness": 0.8,
    "tempo": 1.0,
    "valence": 1.4,
}


def _safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with just the feature columns (filled & cast)."""
    f = df[FEATURE_COLS].copy()
    return f.fillna(0).astype(float)


def _apply_feature_weights(X: pd.DataFrame) -> np.ndarray:
    """Multiply each feature column by its weight (vectorized)."""
    weights = np.array([FEATURE_WEIGHTS[c] for c in FEATURE_COLS], dtype=float)
    return (X.values * weights)


def get_similar_songs(
    df: pd.DataFrame,
    track_id: str,
    top_n: int = 10,
    artist_boost: float = 0.05,   # 5% boost to same-artist tracks
    alpha: float = 0.75           # cosine weight vs euclidean
) -> List[Dict]:
    """
    Return top_n similar songs using a hybrid of (weighted, scaled) cosine + euclidean,
    plus a small same-artist boost. Results include 'match_pct' (0..100).
    """
    if track_id not in df["track_id"].values:
        return []

    # Feature prep: weight → scale
    X = _safe_features(df)
    Xw = _apply_feature_weights(X)
    Xs = StandardScaler().fit_transform(Xw)

    # Locate query row
    idx = df.index[df["track_id"] == track_id][0]

    # Similarities
    cos = cosine_similarity(Xs[idx:idx + 1], Xs)[0]                        # [0..1]
    euc = euclidean_distances(Xs[idx:idx + 1], Xs)[0]
    euc_sim = 1.0 / (1.0 + euc)                                            # ~[0..1]

    # Hybrid
    sim = alpha * cos + (1.0 - alpha) * euc_sim

    # Build local results (do NOT mutate original df)
    results = df.copy()
    results["similarity_raw"] = sim

    # Exclude the track itself
    results = results[results["track_id"] != track_id]

    # Same-artist boost (handles collabs: comma-separated)
    base_artists = []
    base_artist_field = str(df.loc[idx, "artist_name"]) if "artist_name" in df.columns else ""
    if base_artist_field:
        base_artists = [a.strip() for a in base_artist_field.split(",") if a.strip()]

    if base_artists and "artist_name" in results.columns:
        pat = "|".join([re.escape(a) for a in base_artists])
        mask_same_artist = results["artist_name"].fillna("").str.contains(pat, case=False, na=False)
        results.loc[mask_same_artist, "similarity_raw"] *= (1.0 + artist_boost)

    # Tiny popularity bias if available
    if "popularity" in results.columns:
        pop = results["popularity"].fillna(0).astype(float)
        if pop.max() > 0:
            pop_norm = (pop / pop.max())
            results["similarity_raw"] = 0.9 * results["similarity_raw"] + 0.1 * pop_norm

    # Normalize similarities to 0..1 for stable match pct
    sim_vals = results["similarity_raw"].astype(float)
    sim_min, sim_max = sim_vals.min(), sim_vals.max()
    if sim_max > sim_min:
        sim_norm = (sim_vals - sim_min) / (sim_max - sim_min)
    else:
        sim_norm = sim_vals.clip(lower=0.0, upper=1.0)

    results["similarity"] = sim_norm
    results = results.sort_values("similarity", ascending=False)

    # Display fields
    results["match_pct"] = (results["similarity"] * 100.0).round(1)

    # Return top N as dicts
    cols_to_send = list(df.columns) + ["match_pct"]  # keep original fields + match_pct
    return results.head(top_n)[cols_to_send].to_dict(orient="records")


def get_more_from_artists(
    df: pd.DataFrame,
    track_id: str,
    top_n_per_artist: int = 5
) -> Dict[str, List[Dict]]:
    """
    For each artist on the provided track, return up to N other tracks by that artist.
    Returns a dict: { artist_name: [track_dict, ...], ... }
    """
    if track_id not in df["track_id"].values:
        return {}

    row = df[df["track_id"] == track_id].iloc[0]
    artists_field = str(row.get("artist_name", "") or "")
    artists = [a.strip() for a in artists_field.split(",") if a.strip()]
    if not artists:
        return {}

    results: Dict[str, List[Dict]] = {}

    for artist in artists:
        # Match anywhere in artist_name (case-insensitive), exclude the current track
        mask = (
            df["artist_name"].fillna("").str.contains(re.escape(artist), case=False, na=False)
            & (df["track_id"] != track_id)
        )
        subset = df[mask].copy()

        # Optional: language filter (kept minimal—comment out if not needed)
        if "language" in subset.columns:
            subset = subset[~subset["language"].fillna("").str.lower().isin(["unknown", "tamil"])]

        # Prefer popular/recent if present
        sort_cols, ascending = [], []
        if "popularity" in subset.columns:
            sort_cols.append("popularity"); ascending.append(False)
        if "year" in subset.columns:
            sort_cols.append("year"); ascending.append(False)
        if sort_cols:
            subset = subset.sort_values(by=sort_cols, ascending=ascending)

        results[artist] = subset.head(top_n_per_artist).to_dict(orient="records")

    return results
