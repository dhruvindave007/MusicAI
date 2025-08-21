# music/recommender.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

FEATURE_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms"
]

META_COLS = ["track_name", "artist_name", "album_name", "id", "popularity", "year"]

class RecommendationEngine:
    def __init__(self):
        self.df = None                 # full cleaned dataframe
        self.X = None                  # scaled feature matrix (np.array)
        self.scaler = None             # StandardScaler
        self.knn = None                # NearestNeighbors model
        self.kmeans = None             # KMeans for mood-ish clusters
        self.cluster_names = None      # optional mapping from cluster -> mood label

    # ---------- Training ----------
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        # Keep only the needed columns that exist
        have = [c for c in FEATURE_COLS + META_COLS if c in df.columns]
        df = df[have].copy()

        # Basic cleaning
        df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns], inplace=True)
        df.drop_duplicates(subset=[c for c in ["track_name", "artist_name", "id"] if c in df.columns], keep="first", inplace=True)

        # Ensure numeric types
        for c in FEATURE_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns], inplace=True)

        # Some datasets have loudness negative large numbers; it's fine. We’ll scale.
        # Make sure popularity/year exist even if not provided
        for c in ["popularity", "year"]:
            if c not in df.columns:
                df[c] = np.nan

        # Ensure `id` exists (Spotify track id). If not, try to derive from a link column:
        if "id" not in df.columns:
            if "spotify_track_link" in df.columns:
                df["id"] = df["spotify_track_link"].str.extract(r"track/([A-Za-z0-9]+)")
            else:
                # As a fallback, create synthetic ids (won’t be embeddable)
                df["id"] = np.arange(len(df)).astype(str)

        # Normalize text for search
        if "track_name" in df.columns:
            df["track_name_norm"] = df["track_name"].astype(str).str.strip().str.lower()
        else:
            df["track_name_norm"] = ""

        if "artist_name" in df.columns:
            df["artist_name_norm"] = df["artist_name"].astype(str).str.strip().str.lower()
        else:
            df["artist_name_norm"] = ""

        self.df = df.reset_index(drop=True)
        return self.df

    def build_features(self) -> np.ndarray:
        missing = [c for c in FEATURE_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in CSV: {missing}")

        X = self.df[FEATURE_COLS].values.astype(float)

        # Scale: Standardize most features; min-max tempo/duration to balance magnitudes
        std_cols = ["danceability","energy","loudness","speechiness","acousticness",
                    "instrumentalness","liveness","valence"]
        mm_cols = ["tempo","duration_ms"]

        self.scaler = {}
        X_parts = []

        if set(std_cols).issubset(self.df.columns):
            std_scaler = StandardScaler()
            X_std = std_scaler.fit_transform(self.df[std_cols].values)
            self.scaler["std"] = (std_cols, std_scaler)
            X_parts.append(X_std)

        if set(mm_cols).issubset(self.df.columns):
            mm_scaler = MinMaxScaler()
            X_mm = mm_scaler.fit_transform(self.df[mm_cols].values)
            self.scaler["mm"] = (mm_cols, mm_scaler)
            X_parts.append(X_mm)

        self.X = np.concatenate(X_parts, axis=1)
        return self.X

    def fit_models(self, n_neighbors: int = 15, n_clusters: int = 6, random_state: int = 42):
        # KNN over features
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self.knn.fit(self.X)

        # KMeans clusters to act as “moods”
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        clusters = self.kmeans.fit_predict(self.X)
        self.df["cluster"] = clusters

        # Optional: name clusters by average (valence, energy) — heuristic
        cluster_centers = self.kmeans.cluster_centers_
        # We need to recover valence/energy positions in X:
        # Build a map from feature to column index in self.X
        idx_map = {}
        start = 0
        if "std" in self.scaler:
            cols, _ = self.scaler["std"]
            for i, c in enumerate(cols):
                idx_map[c] = start + i
            start += len(cols)
        if "mm" in self.scaler:
            cols, _ = self.scaler["mm"]
            for i, c in enumerate(cols):
                idx_map[c] = start + i

        names = []
        for k in range(n_clusters):
            center = cluster_centers[k]
            # Use valence & energy scaled values as proxy
            v = center[idx_map.get("valence")]
            e = center[idx_map.get("energy")]
            if v is None or e is None:
                label = f"Cluster {k}"
            else:
                # simple naming heuristic
                if v > 0.5 and e > 0.5:
                    label = "Happy / High Energy"
                elif v > 0.5 and e <= 0.5:
                    label = "Warm / Relaxed"
                elif v <= 0.5 and e > 0.5:
                    label = "Intense / Dark"
                else:
                    label = "Melancholic / Low Energy"
            names.append(label)
        self.cluster_names = {k: names[k] for k in range(n_clusters)}

    # ---------- Persistence ----------
    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.knn, os.path.join(dir_path, "knn.joblib"))
        joblib.dump(self.kmeans, os.path.join(dir_path, "kmeans.joblib"))
        joblib.dump(self.scaler, os.path.join(dir_path, "scalers.joblib"))
        # Save metadata and a slimmed frame for fast lookup
        self.df.to_parquet(os.path.join(dir_path, "tracks.parquet"))
        joblib.dump(self.cluster_names, os.path.join(dir_path, "cluster_names.joblib"))

    @classmethod
    def load(cls, dir_path: str):
        eng = cls()
        eng.knn = joblib.load(os.path.join(dir_path, "knn.joblib"))
        eng.kmeans = joblib.load(os.path.join(dir_path, "kmeans.joblib"))
        eng.scaler = joblib.load(os.path.join(dir_path, "scalers.joblib"))
        eng.df = pd.read_parquet(os.path.join(dir_path, "tracks.parquet"))
        eng.cluster_names = joblib.load(os.path.join(dir_path, "cluster_names.joblib"))
        # rebuild X to support neighbor queries
        eng.X = eng._transform_existing(eng.df)
        return eng

    def _transform_existing(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        if "std" in self.scaler:
            cols, std = self.scaler["std"]
            parts.append(std.transform(df[cols].values))
        if "mm" in self.scaler:
            cols, mm = self.scaler["mm"]
            parts.append(mm.transform(df[cols].values))
        return np.concatenate(parts, axis=1)

    # ---------- Query helpers ----------
    def _row_indices_for_search(self, query: str, artist: str | None = None, limit: int = 20):
        q = (query or "").strip().lower()
        a = (artist or "").strip().lower()
        mask = self.df["track_name_norm"].str.contains(q, na=False)
        if a:
            mask &= self.df["artist_name_norm"].str.contains(a, na=False)
        idx = np.where(mask.values)[0][:limit]
        return idx

    def recommend_by_track(self, query: str, artist: str | None = None, top_n: int = 10):
        idxs = self._row_indices_for_search(query, artist, limit=1)
        if len(idxs) == 0:
            return []
        i = idxs[0]

        distances, neighbors = self.knn.kneighbors(self.X[i:i+1], n_neighbors=top_n+1)
        neighbors = neighbors[0].tolist()
        # drop self
        neighbors = [j for j in neighbors if j != i][:top_n]
        return self._rows_to_payload(neighbors)

    def recommend_same_mood(self, query: str, artist: str | None = None, top_n: int = 10):
        idxs = self._row_indices_for_search(query, artist, limit=1)
        if len(idxs) == 0:
            return [], None
        i = idxs[0]
        cluster_id = int(self.df.loc[i, "cluster"])
        mood_name = self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        # pick top_n from same cluster, prefer popular and close tempo
        cluster_df = self.df[self.df["cluster"] == cluster_id].copy()
        # simple re-rank: popularity desc, abs tempo diff asc
        if "tempo" in cluster_df.columns:
            cluster_df["tempo_diff"] = (cluster_df["tempo"] - self.df.loc[i, "tempo"]).abs()
        else:
            cluster_df["tempo_diff"] = 0
        cluster_df = cluster_df.sort_values(by=["popularity", "tempo_diff"], ascending=[False, True])
        # remove the query itself
        cluster_df = cluster_df.iloc[: 500]  # cap
        cluster_df = cluster_df[self.df.index.isin(cluster_df.index) & (self.df.index != i)]
        return self._rows_to_payload(cluster_df.index[:top_n].tolist()), mood_name

    def more_from_same_artist(self, query: str, artist: str | None = None, top_n: int = 6):
        idxs = self._row_indices_for_search(query, artist, limit=1)
        if len(idxs) == 0:
            return []
        i = idxs[0]
        artist_name = self.df.loc[i, "artist_name"]
        same = self.df[self.df["artist_name"] == artist_name].index.tolist()
        # exclude the source track
        same = [j for j in same if j != i][:top_n]
        return self._rows_to_payload(same)

    def _rows_to_payload(self, rows: list[int]):
        out = []
        for i in rows:
            r = self.df.iloc[i]
            out.append({
                "track_name": r.get("track_name") or "",
                "artist_name": r.get("artist_name") or "",
                "album_name": r.get("album_name") or "",
                "id": r.get("id") or "",  # Spotify track id
                "popularity": int(r.get("popularity")) if not pd.isna(r.get("popularity")) else None,
                "year": int(r.get("year")) if not pd.isna(r.get("year")) else None,
                "spotify_url": f"https://open.spotify.com/track/{r.get('id')}" if pd.notna(r.get("id")) else None,
            })
        return out
