"""Content-based music recommender over Spotify audio features.

* Exact cosine k-NN: rows of the standardised feature matrix are L2-normalised
  once, so similarity to every track is a single matrix-vector product.
* K-Means on the same matrix groups tracks into "mood" clusters.
* Artifacts are loaded once per process (see ``get_engine``); nothing is
  retrained at request time.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms",
]
# Columns carried through to the UI / evaluation.
META_COLS = [
    "track_id", "track_name", "artist_name", "album_name", "year",
    "popularity", "artwork_url", "language",
]


def _primary_artist(name: str) -> str:
    return str(name).split(",")[0].strip().lower()


def load_tracks(csv_path: str | os.PathLike) -> pd.DataFrame:
    """Read and clean the raw Spotify CSV."""
    df = pd.read_csv(csv_path)
    if "track_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "track_id"})
    missing = [c for c in FEATURE_COLS + ["track_id", "track_name", "artist_name"]
               if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    for c in META_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[list(dict.fromkeys(META_COLS + FEATURE_COLS))].copy()
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS + ["track_id", "track_name"])
    df = df.drop_duplicates(subset="track_id", keep="first")
    # The same song re-released on several albums would crowd out results:
    # keep the most popular copy.
    df = df.sort_values("popularity", ascending=False, na_position="last", kind="stable")
    df = df.drop_duplicates(subset=["track_name", "artist_name"], keep="first")
    df = df.sort_index().reset_index(drop=True)
    df["track_name_norm"] = df["track_name"].astype(str).str.strip().str.lower()
    df["artist_name_norm"] = df["artist_name"].fillna("").astype(str).str.strip().str.lower()
    df["primary_artist"] = df["artist_name"].fillna("").map(_primary_artist)
    return df


def standardise(df: pd.DataFrame) -> tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    return scaler, scaler.fit_transform(df[FEATURE_COLS].to_numpy(float))


class RecommendationEngine:
    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.X: np.ndarray | None = None        # standardised features
        self.Xn: np.ndarray | None = None       # L2-normalised rows (cosine)
        self.scaler: StandardScaler | None = None
        self.kmeans: KMeans | None = None
        self.cluster_names: dict[int, str] | None = None
        self._id_to_idx: dict[str, int] = {}

    # ------------------------------------------------------------- training
    load_csv = staticmethod(load_tracks)

    def fit(self, df: pd.DataFrame, n_clusters: int = 6,
            random_state: int = 42) -> "RecommendationEngine":
        self.df = df.reset_index(drop=True)
        self.scaler, self.X = standardise(self.df)
        self._finalise_matrix()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        self.df["cluster"] = self.kmeans.fit_predict(self.X)
        self.cluster_names = self._name_clusters()
        return self

    def _finalise_matrix(self):
        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        self.Xn = self.X / np.where(norms == 0, 1.0, norms)
        self._id_to_idx = {t: i for i, t in enumerate(self.df["track_id"].astype(str))}

    def _name_clusters(self) -> dict[int, str]:
        """Label clusters by centre valence/energy relative to the dataset mean
        (features are standardised, so 0 == average)."""
        v_i, e_i = FEATURE_COLS.index("valence"), FEATURE_COLS.index("energy")
        labels = {
            (True, True): "Happy / High Energy",
            (True, False): "Warm / Relaxed",
            (False, True): "Intense / Dark",
            (False, False): "Melancholic / Low Energy",
        }
        return {k: labels[(bool(c[v_i] > 0), bool(c[e_i] > 0))]
                for k, c in enumerate(self.kmeans.cluster_centers_)}

    # ---------------------------------------------------------- persistence
    def save(self, dir_path: str | os.PathLike):
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, d / "scaler.joblib")
        joblib.dump(self.kmeans, d / "kmeans.joblib")
        self.df.to_parquet(d / "tracks.parquet")
        (d / "meta.json").write_text(json.dumps({
            "cluster_names": {str(k): v for k, v in self.cluster_names.items()},
            "feature_cols": FEATURE_COLS,
        }, indent=2))

    @classmethod
    def load(cls, dir_path: str | os.PathLike) -> "RecommendationEngine":
        d = Path(dir_path)
        eng = cls()
        eng.scaler = joblib.load(d / "scaler.joblib")
        eng.kmeans = joblib.load(d / "kmeans.joblib")
        eng.df = pd.read_parquet(d / "tracks.parquet")
        meta = json.loads((d / "meta.json").read_text())
        eng.cluster_names = {int(k): v for k, v in meta["cluster_names"].items()}
        eng.X = eng.scaler.transform(eng.df[FEATURE_COLS].to_numpy(float))
        eng._finalise_matrix()
        return eng

    # -------------------------------------------------------------- queries
    def index_of(self, track_id: str) -> int | None:
        return self._id_to_idx.get(str(track_id))

    def search(self, query: str, limit: int = 50) -> list[dict]:
        """Case-insensitive literal substring search on track / artist / album."""
        q = (query or "").strip().lower()
        if not q:
            return []
        mask = (
            self.df["track_name_norm"].str.contains(q, regex=False)
            | self.df["artist_name_norm"].str.contains(q, regex=False)
            | self.df["album_name"].fillna("").str.lower().str.contains(q, regex=False)
        ).to_numpy()
        idx = np.flatnonzero(mask)
        if len(idx) > limit:  # keep the most popular matches
            pop = np.nan_to_num(self.df["popularity"].to_numpy(float)[idx])
            idx = idx[np.argsort(-pop, kind="stable")[:limit]]
        return self._payload(idx)

    def get_track(self, track_id: str) -> dict | None:
        i = self.index_of(track_id)
        return None if i is None else self._payload([i])[0]

    def similarities(self, i: int) -> np.ndarray:
        """Cosine similarity of track ``i`` to every track (vectorised)."""
        return self.Xn @ self.Xn[i]

    def neighbors(self, i: int, top_n: int = 10, mask: np.ndarray | None = None) -> np.ndarray:
        """Indices of the ``top_n`` most similar tracks, best first. Excludes
        track ``i`` itself and other copies of the same song."""
        sim = self.similarities(i)
        same_song = ((self.df["track_name_norm"].to_numpy() == self.df["track_name_norm"].iat[i])
                     & (self.df["primary_artist"].to_numpy() == self.df["primary_artist"].iat[i]))
        sim[same_song] = -np.inf
        sim[i] = -np.inf
        if mask is not None:
            sim[~mask] = -np.inf
        top_n = min(top_n, int(np.isfinite(sim).sum()))
        if top_n <= 0:
            return np.array([], dtype=int)
        part = np.argpartition(-sim, top_n - 1)[:top_n]
        return part[np.argsort(-sim[part], kind="stable")]

    def recommend_similar(self, track_id: str, top_n: int = 10) -> list[dict]:
        i = self.index_of(track_id)
        if i is None:
            return []
        idx = self.neighbors(i, top_n)
        return self._payload(idx, sims=self.similarities(i)[idx])

    def recommend_same_mood(self, track_id: str, top_n: int = 10):
        """Nearest tracks restricted to the query's K-Means cluster."""
        i = self.index_of(track_id)
        if i is None:
            return [], None
        cid = int(self.df["cluster"].iat[i])
        idx = self.neighbors(i, top_n, mask=(self.df["cluster"].to_numpy() == cid))
        mood = self.cluster_names.get(cid, f"Cluster {cid}")
        return self._payload(idx, sims=self.similarities(i)[idx]), mood

    def more_from_artists(self, track_id: str, top_n_per_artist: int = 5) -> dict[str, list[dict]]:
        i = self.index_of(track_id)
        if i is None:
            return {}
        out = {}
        names = self.df["artist_name"].fillna("").astype(str)
        lowered = names.str.lower()
        for artist in [a.strip() for a in names.iat[i].split(",") if a.strip()]:
            mask = lowered.str.contains(artist.lower(), regex=False).to_numpy()
            mask[i] = False
            sub = self.df[mask].sort_values(["popularity", "year"], ascending=False,
                                            na_position="last")
            if len(sub):
                out[artist] = self._payload(sub.index[:top_n_per_artist])
        return out

    # --------------------------------------------------------------- output
    def _payload(self, idx, sims=None) -> list[dict]:
        rows = self.df.iloc[list(idx)][META_COLS].astype(object)
        rows = rows.where(rows.notna(), None)
        recs = rows.to_dict(orient="records")
        for k, r in enumerate(recs):
            for c in ("year", "popularity"):
                if r[c] is not None:
                    r[c] = int(r[c])
            r["match_pct"] = None if sims is None else round(float(max(sims[k], 0.0)) * 100, 1)
        return recs


# ------------------------------------------------- process-wide singleton
_engine: RecommendationEngine | None = None
_lock = threading.Lock()


def get_engine(artifacts_dir: str | os.PathLike | None = None) -> RecommendationEngine:
    """Load artifacts once and reuse. Raises FileNotFoundError if untrained."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                if artifacts_dir is None:
                    from django.conf import settings
                    artifacts_dir = settings.ARTIFACTS_DIR
                _engine = RecommendationEngine.load(artifacts_dir)
    return _engine


def reset_engine():
    global _engine
    _engine = None
