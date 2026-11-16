from django.conf import settings
from django.core.management.base import BaseCommand

from music.recommender import RecommendationEngine, load_tracks


class Command(BaseCommand):
    help = "Train the recommender from the CSV and write artifacts (run after changing data/features)."

    def add_arguments(self, parser):
        parser.add_argument("--csv", default=str(settings.DATA_CSV))
        parser.add_argument("--out", default=str(settings.ARTIFACTS_DIR))
        parser.add_argument("--clusters", type=int, default=6)
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **o):
        self.stdout.write(f"Loading {o['csv']}")
        df = load_tracks(o["csv"])
        self.stdout.write(f"{len(df)} tracks after cleaning; fitting k={o['clusters']}")
        eng = RecommendationEngine().fit(df, n_clusters=o["clusters"], random_state=o["seed"])
        eng.save(o["out"])
        self.stdout.write(self.style.SUCCESS(f"Artifacts saved to {o['out']}"))
