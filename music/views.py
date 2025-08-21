# views.py
from django.shortcuts import render
from . import utils
import pandas as pd

# Load your dataset once globally
df = pd.read_csv("music/data/spotify_tracks.csv") # adjust path if needed


def song_detail(request, track_id):
    if track_id not in df['track_id'].values:
        return render(request, "not_found.html", {"track_id": track_id})

    # Get current song row
    row = df[df['track_id'] == track_id].iloc[0]

    # Get recommendations
    similar_songs = utils.get_similar_songs(df, track_id, top_n=10)
    for s in similar_songs:
        if "match_pct" in s:
            s["match_pct"] = round(s["match_pct"], 1)

    # Get more songs from same artist(s)
    more_from = utils.get_more_from_artists(df, track_id, top_n_per_artist=5)

    return render(request, "music/song_detail.html", {
        "song": row.to_dict(),
        "similar_songs": similar_songs,
        "more_from": more_from
    })
