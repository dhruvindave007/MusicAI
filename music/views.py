import logging

from django.http import Http404
from django.shortcuts import render

from .recommender import get_engine

logger = logging.getLogger(__name__)

SEARCH_LIMIT = 50


def _engine_or_none():
    try:
        return get_engine()
    except FileNotFoundError:
        logger.error("Recommender artifacts missing; run `python manage.py train_recommender`.")
        return None


def _unavailable(request):
    return render(request, "music/search.html",
                  {"query": "", "results": [], "error": "Recommender is not trained yet."},
                  status=503)


def index(request):
    return render(request, "music/search.html", {"query": "", "results": []})


def search(request):
    q = request.GET.get("q", "").strip()
    engine = _engine_or_none()
    if engine is None:
        return _unavailable(request)
    results = engine.search(q, limit=SEARCH_LIMIT) if q else []
    return render(request, "music/search.html", {"query": q, "results": results})


def song_detail(request, track_id):
    engine = _engine_or_none()
    if engine is None:
        return _unavailable(request)
    song = engine.get_track(track_id)
    if song is None:
        raise Http404("Unknown track")
    same_mood, mood = engine.recommend_same_mood(track_id, top_n=6)
    return render(request, "music/song_detail.html", {
        "song": song,
        "similar_songs": engine.recommend_similar(track_id, top_n=10),
        "same_mood": same_mood,
        "mood": mood,
        "more_from": engine.more_from_artists(track_id, top_n_per_artist=5),
    })
