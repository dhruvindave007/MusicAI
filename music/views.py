# views.py
import pandas as pd
from django.shortcuts import render
from . import utils

# Load CSV once (stick with your existing file)
df = pd.read_csv("music/data/spotify_tracks.csv")

def song_detail(request, track_id):
    # Validate the song exists
    if track_id not in df['track_id'].values:
        return render(request, "not_found.html", {"track_id": track_id})

    # Current song
    row = df[df['track_id'] == track_id].iloc[0]

    # Similar Songs (across dataset) — hybrid similarity
    similar_songs = utils.get_similar_songs(df, track_id, top_n=10)

    # More from the Artist(s)
    more_from = utils.get_more_from_artists(df, track_id, top_n_per_artist=5)

    return render(request, "music/song_detail.html", {
        "song": row.to_dict(),
        "similar_songs": similar_songs,
        "more_from": more_from,     # dict: { artist -> [tracks...] }
    })


def search(request):
    q = request.GET.get('q', '').strip()
    results = []
    if q:
        mask = (
            df['track_name'].fillna('').str.contains(q, case=False, na=False) |
            df['artist_name'].fillna('').str.contains(q, case=False, na=False) |
            df.get('album_name', '').fillna('').str.contains(q, case=False, na=False)
        )
        results = df[mask].to_dict(orient='records')

    return render(request, 'music/search.html', {'query': q, 'results': results})


def index(request):
    return render(request, 'music/search.html', {'query': '', 'results': []})
