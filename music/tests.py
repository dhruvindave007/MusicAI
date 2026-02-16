from django.test import TestCase
import os
import pandas as pd
import numpy as np
from pathlib import Path
from django.conf import settings
from .recommender import RecommendationEngine, FEATURE_COLS
from .utils import get_similar_songs, get_more_from_artists, FEATURE_COLS as UTILS_FEATURE_COLS


class RecommenderEngineTests(TestCase):
    """Test the machine learning recommendation engine."""
    
    def setUp(self):
        """Set up test data."""
        self.csv_path = os.path.join(settings.BASE_DIR, "music", "data", "spotify_tracks.csv")
        self.artifacts_dir = os.path.join(settings.BASE_DIR, "music", "artifacts")
        
    def test_csv_exists(self):
        """Verify the Spotify tracks CSV file exists."""
        self.assertTrue(
            os.path.exists(self.csv_path),
            f"CSV file not found at {self.csv_path}"
        )
    
    def test_csv_loads(self):
        """Test that CSV loads without errors."""
        df = pd.read_csv(self.csv_path)
        self.assertGreater(len(df), 0, "CSV is empty")
        
    def test_csv_has_required_columns(self):
        """Verify CSV contains required columns."""
        df = pd.read_csv(self.csv_path)
        required_cols = ["track_name", "artist_name"]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required column: {col}")
    
    def test_csv_has_audio_features(self):
        """Verify CSV contains audio feature columns."""
        df = pd.read_csv(self.csv_path)
        for feature in FEATURE_COLS:
            self.assertIn(
                feature, df.columns,
                f"Missing audio feature column: {feature}"
            )
    
    def test_engine_initialization(self):
        """Test RecommendationEngine can be instantiated."""
        engine = RecommendationEngine()
        self.assertIsNone(engine.df)
        self.assertIsNone(engine.knn)
        self.assertIsNone(engine.kmeans)
    
    def test_load_csv_processing(self):
        """Test CSV loading and preprocessing."""
        engine = RecommendationEngine()
        df = engine.load_csv(self.csv_path)
        
        # Check data is loaded
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        
        # Check normalization columns created
        self.assertIn("track_name_norm", df.columns)
        self.assertIn("artist_name_norm", df.columns)
        
        # Check no NaNs in feature columns
        for col in FEATURE_COLS:
            if col in df.columns:
                self.assertEqual(
                    df[col].isna().sum(), 0,
                    f"Found NaN values in {col} after cleaning"
                )
    
    def test_build_features(self):
        """Test feature matrix construction."""
        engine = RecommendationEngine()
        engine.load_csv(self.csv_path)
        X = engine.build_features()
        
        # Check shape
        self.assertEqual(X.shape[0], len(engine.df))
        self.assertEqual(X.shape[1], len(FEATURE_COLS))
        
        # Check no NaNs or infinities
        self.assertFalse(np.any(np.isnan(X)))
        self.assertFalse(np.any(np.isinf(X)))
    
    def test_artifacts_exist(self):
        """Verify trained model artifacts exist."""
        required_files = [
            "knn.joblib",
            "kmeans.joblib",
            "scalers.joblib",
            "cluster_names.joblib",
            "tracks.parquet"
        ]
        
        for filename in required_files:
            filepath = os.path.join(self.artifacts_dir, filename)
            self.assertTrue(
                os.path.exists(filepath),
                f"Missing artifact: {filename}. Run 'python manage.py train_recommender'"
            )
    
    def test_load_saved_engine(self):
        """Test loading a pre-trained engine from artifacts."""
        try:
            engine = RecommendationEngine.load(self.artifacts_dir)
            
            # Check models loaded
            self.assertIsNotNone(engine.knn)
            self.assertIsNotNone(engine.kmeans)
            self.assertIsNotNone(engine.scaler)
            self.assertIsNotNone(engine.df)
            self.assertIsNotNone(engine.cluster_names)
            
            # Check data shape
            self.assertGreater(len(engine.df), 0)
            
        except FileNotFoundError:
            self.skipTest("Artifacts not found. Run 'python manage.py train_recommender' first.")
    
    def test_recommendations(self):
        """Test that recommendations can be generated."""
        try:
            engine = RecommendationEngine.load(self.artifacts_dir)
            
            # Get first track name
            sample_track = engine.df.iloc[0]["track_name"]
            
            # Generate recommendations
            recs = engine.recommend_by_track(sample_track, top_n=5)
            
            # Check results
            self.assertIsInstance(recs, list)
            self.assertGreater(len(recs), 0)
            self.assertLessEqual(len(recs), 5)
            
            # Check recommendation structure
            if recs:
                rec = recs[0]
                self.assertIn("track_name", rec)
                self.assertIn("artist_name", rec)
                
        except FileNotFoundError:
            self.skipTest("Artifacts not found. Run 'python manage.py train_recommender' first.")


class SimilarityUtilsTests(TestCase):
    """Test similarity calculation utilities."""
    
    def setUp(self):
        """Set up test dataframe."""
        self.csv_path = os.path.join(settings.BASE_DIR, "music", "data", "spotify_tracks.csv")
        
    def test_utils_feature_columns_match(self):
        """Verify utils.py and recommender.py use compatible features."""
        # They might not be identical but should have overlap
        overlap = set(FEATURE_COLS) & set(UTILS_FEATURE_COLS)
        self.assertGreater(
            len(overlap), 5,
            "Feature columns in utils.py and recommender.py should overlap"
        )
    
    def test_get_similar_songs_basic(self):
        """Test basic similarity calculation."""
        df = pd.read_csv(self.csv_path)
        
        # Need track_id column
        if "track_id" not in df.columns:
            if "id" in df.columns:
                df["track_id"] = df["id"]
            else:
                self.skipTest("No track_id or id column in CSV")
        
        # Get first track
        track_id = df.iloc[0]["track_id"]
        
        # Calculate similar songs
        similar = get_similar_songs(df, track_id, top_n=5)
        
        # Check results
        self.assertIsInstance(similar, list)
        self.assertLessEqual(len(similar), 5)
        
        # Check match_pct exists
        if similar:
            self.assertIn("match_pct", similar[0])
            match_pct = similar[0]["match_pct"]
            self.assertGreaterEqual(match_pct, 0)
            self.assertLessEqual(match_pct, 100)
    
    def test_get_similar_songs_excludes_self(self):
        """Verify similar songs doesn't include the query track itself."""
        df = pd.read_csv(self.csv_path)
        
        if "track_id" not in df.columns:
            if "id" in df.columns:
                df["track_id"] = df["id"]
            else:
                self.skipTest("No track_id column")
        
        track_id = df.iloc[0]["track_id"]
        similar = get_similar_songs(df, track_id, top_n=10)
        
        # Make sure query track is not in results
        similar_ids = [s.get("track_id") for s in similar]
        self.assertNotIn(track_id, similar_ids)
    
    def test_get_more_from_artists(self):
        """Test artist-based recommendations."""
        df = pd.read_csv(self.csv_path)
        
        if "track_id" not in df.columns:
            if "id" in df.columns:
                df["track_id"] = df["id"]
            else:
                self.skipTest("No track_id column")
        
        track_id = df.iloc[0]["track_id"]
        more_from = get_more_from_artists(df, track_id, top_n_per_artist=3)
        
        # Check results
        self.assertIsInstance(more_from, dict)
        
        # If there are results, verify structure
        if more_from:
            for artist, tracks in more_from.items():
                self.assertIsInstance(tracks, list)
                self.assertLessEqual(len(tracks), 3)


class AccuracyMetricsTests(TestCase):
    """Test recommendation accuracy and quality."""
    
    def setUp(self):
        """Set up test data."""
        self.artifacts_dir = os.path.join(settings.BASE_DIR, "music", "artifacts")
    
    def test_recommendation_diversity(self):
        """Verify recommendations have diversity (not all same artist)."""
        try:
            engine = RecommendationEngine.load(self.artifacts_dir)
            
            # Get recommendations for a sample track
            sample_track = engine.df.iloc[100]["track_name"]
            recs = engine.recommend_by_track(sample_track, top_n=10)
            
            if len(recs) > 5:
                # Get unique artists
                artists = set(r["artist_name"] for r in recs)
                
                # Expect at least 3 different artists in top 10
                self.assertGreaterEqual(
                    len(artists), 3,
                    "Recommendations should include diverse artists"
                )
                
        except FileNotFoundError:
            self.skipTest("Artifacts not found")
    
    def test_cluster_assignment(self):
        """Verify all tracks are assigned to clusters."""
        try:
            engine = RecommendationEngine.load(self.artifacts_dir)
            
            # Check cluster column exists
            self.assertIn("cluster", engine.df.columns)
            
            # Check all tracks have cluster assignments
            self.assertEqual(engine.df["cluster"].isna().sum(), 0)
            
            # Check clusters are in expected range
            max_cluster = engine.df["cluster"].max()
            self.assertGreater(max_cluster, 0)
            self.assertLess(max_cluster, 20)  # Reasonable cluster count
            
        except FileNotFoundError:
            self.skipTest("Artifacts not found")
    
    def test_similarity_scores_reasonable(self):
        """Test that similarity scores are in reasonable ranges."""
        df = pd.read_csv(os.path.join(settings.BASE_DIR, "music", "data", "spotify_tracks.csv"))
        
        if "track_id" not in df.columns:
            if "id" in df.columns:
                df["track_id"] = df["id"]
            else:
                self.skipTest("No track_id column")
        
        # Sample a few tracks
        sample_ids = df["track_id"].sample(min(5, len(df))).tolist()
        
        for track_id in sample_ids:
            similar = get_similar_songs(df, track_id, top_n=3)
            
            if similar:
                # Top match should be reasonably high
                top_match = similar[0]["match_pct"]
                self.assertGreater(
                    top_match, 30,
                    f"Top similarity score too low: {top_match}%"
                )
