from unittest import mock

import numpy as np
import pandas as pd
from django.test import SimpleTestCase
from django.urls import reverse

from music.evaluation import evaluate_neighbors, kmeans_selection
from music.recommender import FEATURE_COLS, RecommendationEngine, load_tracks


def make_csv(path, n=120, seed=0):
    """Synthetic catalogue: 3 well-separated feature blobs, 2 artists per blob."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        blob = i % 3
        feats = {c: rng.normal(blob * 3.0, 0.2) for c in FEATURE_COLS}
        rows.append({
            "track_id": f"id{i:03d}", "track_name": f"Song {i}",
            "artist_name": f"Artist {blob}{i % 2}" + (", Guest" if i == 0 else ""),
            "album_name": f"Album {blob}", "year": 2020, "popularity": i % 100,
            "artwork_url": "http://x/y.jpg", "language": ["en", "hi", "ta"][blob], **feats,
        })
    rows.append(dict(rows[5], track_id="dup", popularity=0))  # same song, other release
    pd.DataFrame(rows).to_csv(path, index=False)


class EngineTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import tempfile, os
        cls.tmp = tempfile.mkdtemp()
        csv = os.path.join(cls.tmp, "t.csv")
        make_csv(csv)
        cls.df = load_tracks(csv)
        cls.engine = RecommendationEngine().fit(cls.df, n_clusters=3)

    def test_cleaning_drops_duplicate_song(self):
        self.assertEqual(len(self.df), 120)
        self.assertNotIn("dup", set(self.df.track_id))

    def test_similar_excludes_self_and_is_sorted(self):
        recs = self.engine.recommend_similar("id005", top_n=10)
        self.assertEqual(len(recs), 10)
        self.assertNotIn("id005", [r["track_id"] for r in recs])
        pct = [r["match_pct"] for r in recs]
        self.assertEqual(pct, sorted(pct, reverse=True))

    def test_similar_stays_in_blob(self):
        recs = self.engine.recommend_similar("id000", top_n=10)
        self.assertTrue(all(r["language"] == "en" for r in recs))

    def test_matches_brute_force(self):
        i = self.engine.index_of("id010")
        sims = self.engine.Xn @ self.engine.Xn[i]
        sims[i] = -np.inf
        got = self.engine.neighbors(i, 5)
        self.assertEqual(set(got), set(np.argsort(-sims)[:5]))

    def test_unknown_track(self):
        self.assertEqual(self.engine.recommend_similar("nope"), [])
        self.assertEqual(self.engine.recommend_same_mood("nope"), ([], None))
        self.assertIsNone(self.engine.get_track("nope"))

    def test_same_mood_shares_cluster(self):
        recs, mood = self.engine.recommend_same_mood("id001", top_n=5)
        df = self.engine.df
        cid = df.loc[self.engine.index_of("id001"), "cluster"]
        got = {df.loc[self.engine.index_of(r["track_id"]), "cluster"] for r in recs}
        self.assertEqual(got, {cid})
        self.assertIn(mood, set(self.engine.cluster_names.values()))

    def test_more_from_artists_handles_collab(self):
        more = self.engine.more_from_artists("id000")
        self.assertIn("Artist 00", more)
        self.assertNotIn("id000", [t["track_id"] for t in more["Artist 00"]])

    def test_search_is_literal_not_regex(self):
        self.assertEqual(self.engine.search("(")[:], [])
        self.assertEqual(len(self.engine.search("song 7", limit=50)), 11)  # 7, 70-79
        self.assertEqual(len(self.engine.search("song", limit=5)), 5)

    def test_save_load_roundtrip(self):
        import os
        d = os.path.join(self.tmp, "art")
        self.engine.save(d)
        loaded = RecommendationEngine.load(d)
        np.testing.assert_allclose(loaded.Xn, self.engine.Xn)
        self.assertEqual(loaded.cluster_names, self.engine.cluster_names)
        self.assertEqual(loaded.recommend_similar("id003"), self.engine.recommend_similar("id003"))


class EvaluationTests(SimpleTestCase):
    def test_metrics_in_range_and_knn_beats_random(self):
        import tempfile, os
        csv = os.path.join(tempfile.mkdtemp(), "t.csv")
        make_csv(csv)
        eng = RecommendationEngine().fit(load_tracks(csv), n_clusters=3)
        res = evaluate_neighbors(eng, k=5, n_queries=50, seed=1)
        self.assertGreater(res["knn"]["language_precision"], res["random"]["language_precision"])
        for m in res.values():
            if isinstance(m, dict):
                self.assertTrue(all(0 <= v <= 2 for v in m.values()))  # ILD is a cosine distance

    def test_kmeans_selection_finds_three_blobs(self):
        import tempfile, os
        csv = os.path.join(tempfile.mkdtemp(), "t.csv")
        make_csv(csv)
        eng = RecommendationEngine().fit(load_tracks(csv), n_clusters=3)
        rows = kmeans_selection(eng.X, range(2, 6), sample_size=100)
        self.assertEqual(max(rows, key=lambda r: r["silhouette"])["k"], 3)


class ViewTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import tempfile, os
        csv = os.path.join(tempfile.mkdtemp(), "t.csv")
        make_csv(csv)
        cls.engine = RecommendationEngine().fit(load_tracks(csv), n_clusters=3)

    def setUp(self):
        p = mock.patch("music.views.get_engine", return_value=self.engine)
        p.start()
        self.addCleanup(p.stop)

    def test_index(self):
        self.assertEqual(self.client.get(reverse("index")).status_code, 200)

    def test_search_results(self):
        r = self.client.get(reverse("search"), {"q": "song 12"})
        self.assertContains(r, "Song 12")

    def test_search_regex_chars_do_not_crash(self):
        self.assertEqual(self.client.get(reverse("search"), {"q": "[("}).status_code, 200)

    def test_search_no_results(self):
        self.assertContains(self.client.get(reverse("search"), {"q": "zzzz"}), "0 Results")

    def test_detail(self):
        r = self.client.get(reverse("song_detail", args=["id004"]))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Song 4")
        self.assertEqual(len(r.context["similar_songs"]), 10)
        self.assertIn("mood", r.context)

    def test_detail_unknown_is_404(self):
        self.assertEqual(self.client.get(reverse("song_detail", args=["nope"])).status_code, 404)

    def test_missing_artifacts_gives_503(self):
        with mock.patch("music.views.get_engine", side_effect=FileNotFoundError):
            self.assertEqual(self.client.get(reverse("search"), {"q": "a"}).status_code, 503)
            self.assertEqual(self.client.get(reverse("song_detail", args=["x"])).status_code, 503)


class ShippedArtifactsTests(SimpleTestCase):
    def test_real_artifacts_load_and_recommend(self):
        from django.conf import settings
        if not (settings.ARTIFACTS_DIR / "tracks.parquet").exists():
            self.skipTest("run `python manage.py train_recommender` first")
        eng = RecommendationEngine.load(settings.ARTIFACTS_DIR)
        tid = eng.df.track_id.iat[100]
        recs = eng.recommend_similar(tid, 10)
        self.assertEqual(len(recs), 10)
        self.assertNotIn(tid, [r["track_id"] for r in recs])
        self.assertEqual(eng.df["cluster"].isna().sum(), 0)
