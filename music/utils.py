import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA

FEATURE_COLS = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
]

# Assign feature weights (tuned for better balance)
FEATURE_WEIGHTS = {
    'acousticness': 1.0,
    'danceability': 1.5,
    'energy': 1.5,
    'instrumentalness': 1.0,
    'liveness': 0.8,
    'loudness': 1.2,
    'speechiness': 0.8,
    'tempo': 1.2,
    'valence': 1.5
}


def get_song(track_id):
    """Fetch a single song by ID"""
    song = df[df['track_id'] == track_id]
    if song.empty:
        return None
    return song.iloc[0].to_dict()


def get_similar_songs(df, track_id, top_n=10, use_pca=True):
    """
    Returns top_n similar songs for a given track_id
    based on weighted audio features and hybrid similarity.
    """

    if track_id not in df['track_id'].values:
        return []

    # Copy df to avoid modifying original
    features = df.copy()

    # Apply weights to feature columns
    for col in FEATURE_COLS:
        if col in features:
            features[col] = features[col] * FEATURE_WEIGHTS[col]

    # Extract only feature matrix
    X = features[FEATURE_COLS].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional PCA to denoise redundant features
    if use_pca and X_scaled.shape[1] > 2:
        pca = PCA(n_components=0.95, random_state=42)  # keep 95% variance
        X_scaled = pca.fit_transform(X_scaled)

    # Get index of the query track
    idx = df.index[df['track_id'] == track_id][0]

    # Compute cosine similarity
    cos_sim = cosine_similarity(X_scaled[idx:idx+1], X_scaled)[0]

    # Compute euclidean similarity
    euc_dist = euclidean_distances(X_scaled[idx:idx+1], X_scaled)[0]
    euc_sim = 1 / (1 + euc_dist)  # convert distance → similarity

    # Hybrid similarity score
    alpha = 0.7  # weight for cosine vs euclidean
    final_sim = alpha * cos_sim + (1 - alpha) * euc_sim

    # Build result DataFrame
    results = df.copy()
    results['similarity'] = final_sim

    # Exclude the same song
    results = results[results['track_id'] != track_id]

    # Boost songs from the same artist slightly
    base_song = get_song(track_id)
    if base_song and 'artist_name' in df.columns:
        artist = str(base_song.get('artist_name', '')).lower()
        results.loc[
            results['artist_name'].str.lower().str.contains(artist, na=False),
            'similarity'
        ] *= 1.05

    # Normalize similarity (so percentages are consistent)
    results['similarity'] = (results['similarity'] - results['similarity'].min()) / (
        results['similarity'].max() - results['similarity'].min() + 1e-9
    )

    # Sort by similarity
    results = results.sort_values('similarity', ascending=False)

    # Absolute similarity percentage
    results['match_pct'] = (results['similarity'] * 100).round(1)

    return results.head(top_n).to_dict(orient='records')


def get_more_from_artists(track_id, top_n_per_artist=5):
    """For each artist on the provided track, return other top tracks.

    - Splits the track's `artist_name` by commas to support collaborations.
    - For each artist, finds tracks where `artist_name` contains that artist (case-insensitive).
    - Excludes the current track and returns up to `top_n_per_artist` tracks per artist.
    - Sorts by 'popularity' if available, descending, as a simple relevance heuristic.
    - Returns a dict: { artist_display_name: [track_dict, ...], ... }
    """
    song = get_song(track_id)
    if not song:
        return {}

    artists_field = song.get('artist_name', '') or ''
    artists = [a.strip() for a in artists_field.split(',') if a.strip()]

    results = {}
    for artist in artists:
        # Case-insensitive substring match within the artist_name field
        mask = df['artist_name'].fillna('').str.contains(re.escape(artist), case=False, na=False)
        subset = df[mask].copy()
        # Exclude the current track
        subset = subset[subset['track_id'] != track_id]

        # Exclude tracks with language unknown or tamil
        if 'language' in subset.columns:
            subset = subset[~subset['language'].fillna('').str.lower().isin(['unknown', 'tamil'])]

        # Sort by popularity if present, otherwise by year then track_name
        sort_cols = []
        if 'popularity' in subset.columns:
            sort_cols.append('popularity')
        if 'year' in subset.columns:
            sort_cols.append('year')
        if sort_cols:
            subset = subset.sort_values(by=sort_cols, ascending=False)

        results[artist] = subset.head(top_n_per_artist).to_dict(orient='records')

    return results