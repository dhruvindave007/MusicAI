from django.urls import path
from . import views

urlpatterns = [
    path("search/", views.search, name="search"),
    path("song/<str:track_id>/", views.song_detail, name="song_detail"),
]