import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from music.evaluation import evaluate_neighbors, kmeans_selection
from music.recommender import RecommendationEngine, load_tracks, standardise


class Command(BaseCommand):
    help = "Offline evaluation: k-NN hit-rate/diversity vs baselines and K-Means k selection."

    def add_arguments(self, parser):
        parser.add_argument("--csv", default=str(settings.DATA_CSV))
        parser.add_argument("--k", type=int, default=10)
        parser.add_argument("--queries", type=int, default=2000)
        parser.add_argument("--max-clusters", type=int, default=12)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--out", default="docs/eval_results.json")

    def handle(self, *args, **o):
        df = load_tracks(o["csv"])
        eng = RecommendationEngine().fit(df, n_clusters=6)
        self.stdout.write(f"Catalogue: {len(df)} tracks")
        neighbors = evaluate_neighbors(eng, k=o["k"], n_queries=o["queries"], seed=o["seed"])
        _, X = standardise(df)
        clustering = kmeans_selection(X, range(2, o["max_clusters"] + 1))
        result = {"neighbors": neighbors, "kmeans": clustering}
        out = Path(o["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        self.stdout.write(json.dumps(result, indent=2))
        self.stdout.write(self.style.SUCCESS(f"Wrote {out}"))
