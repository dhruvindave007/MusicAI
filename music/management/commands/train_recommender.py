# music/management/commands/train_recommender.py
from django.core.management.base import BaseCommand
from django.conf import settings
import os
from music.recommender import RecommendationEngine

class Command(BaseCommand):
    help = "Train and save the recommendation engine artifacts."

    def add_arguments(self, parser):
        parser.add_argument("--csv", default=os.path.join(settings.BASE_DIR, "music", "data", "spotify_tracks.csv"))
        parser.add_argument("--out", default=os.path.join(settings.BASE_DIR, "music", "artifacts"))
        parser.add_argument("--neighbors", type=int, default=15)
        parser.add_argument("--clusters", type=int, default=6)

    def handle(self, *args, **opts):
        csv_path = opts["csv"]
        out_dir = opts["out"]
        neighbors = opts["neighbors"]
        clusters = opts["clusters"]

        eng = RecommendationEngine()
        self.stdout.write(self.style.WARNING(f"Loading CSV: {csv_path}"))
        eng.load_csv(csv_path)

        self.stdout.write(self.style.WARNING("Building features..."))
        eng.build_features()

        self.stdout.write(self.style.WARNING(f"Fitting models (neighbors={neighbors}, clusters={clusters})..."))
        eng.fit_models(n_neighbors=neighbors, n_clusters=clusters)

        self.stdout.write(self.style.WARNING(f"Saving artifacts -> {out_dir}"))
        eng.save(out_dir)

        self.stdout.write(self.style.SUCCESS("Training complete."))
