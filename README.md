# 🎵 MusicAI - Intelligent Music Recommendation System

A Django-based music recommendation web application that finds similar songs using advanced audio feature analysis and machine learning.

## 🌟 Features

- **Smart Song Search**: Search for tracks by song name, artist, or album
- **Similar Song Discovery**: Find songs with similar audio characteristics using hybrid similarity (cosine + euclidean)
- **Mood-based Recommendations**: Discover songs with similar emotional qualities
- **Artist Deep Dive**: Explore more tracks from the same artist
- **Audio Feature Analysis**: Leverages 10 audio features including danceability, energy, valence, tempo, and more
- **Machine Learning Models**: KNN + K-Means clustering for intelligent recommendations

## 📊 How It Works

### Audio Features Used
- **Danceability**: How suitable the track is for dancing
- **Energy**: Intensity and activity measure
- **Valence**: Musical positiveness (happy/sad)
- **Tempo**: Speed in BPM
- **Acousticness**: Confidence of acoustic sound
- **Instrumentalness**: Predicts vocal content
- **Speechiness**: Presence of spoken words
- **Liveness**: Audience presence probability
- **Loudness**: Overall volume in dB
- **Duration**: Track length

### Recommendation Algorithm
1. **Feature Weighting**: Prioritizes danceability, energy, and valence (1.4x weight)
2. **Scaling**: StandardScaler for audio features, MinMaxScaler for tempo/duration
3. **Hybrid Similarity**: 75% cosine similarity + 25% euclidean distance
4. **Artist Boost**: 5% bonus for same-artist tracks
5. **Popularity Bias**: 10% influence from track popularity
6. **Match Score**: 0-100% similarity percentage

### Expected Accuracy
- **Subjective Match Quality**: 60-75% (typical for feature-based recommenders)
- **Same-mood Clustering**: ~6 mood clusters (Happy/High Energy, Warm/Relaxed, etc.)
- No deep learning required - fast and efficient

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dhruvindave007/MusicApp.git
   cd MusicApp
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database**
   ```bash
   python manage.py migrate
   ```

5. **Train the recommendation models**
   ```bash
   python manage.py train_recommender
   ```
   This will:
   - Load `music/data/spotify_tracks.csv` (114,000+ tracks)
   - Extract and scale audio features
   - Train KNN model (15 neighbors)
   - Train K-Means clustering (6 mood clusters)
   - Save models to `music/artifacts/`

6. **Run the development server**
   ```bash
   python manage.py runserver
   ```

7. **Open your browser**
   Navigate to `http://127.0.0.1:8000/`

## 📁 Project Structure

```
MusicApp/
│
├── manage.py                          # Django management script
├── requirements.txt                   # Python dependencies
├── db.sqlite3                         # SQLite database
│
├── MusicAI/                          # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── music/                             # Main application
│   ├── views.py                       # Request handlers
│   ├── urls.py                        # URL routing
│   ├── models.py                      # Database models
│   ├── utils.py                       # Similarity calculations
│   ├── recommender.py                 # ML recommendation engine
│   │
│   ├── data/
│   │   └── spotify_tracks.csv         # Dataset (114k+ tracks)
│   │
│   ├── artifacts/                     # Trained models
│   │   ├── knn.joblib                 # KNN model
│   │   ├── kmeans.joblib              # K-Means clustering
│   │   ├── scalers.joblib             # Feature scalers
│   │   ├── cluster_names.joblib       # Mood cluster labels
│   │   └── tracks.parquet             # Processed data
│   │
│   ├── management/commands/
│   │   └── train_recommender.py       # Model training command
│   │
│   └── templates/music/               # HTML templates
│       ├── search.html
│       ├── song_detail.html
│       └── results.html
│
└── static/                            # CSS, JS, images
    └── style.css
```

## 🎯 Usage

### Search for Songs
1. Go to the home page
2. Enter a song name, artist, or album
3. Click search or press Enter

### View Similar Songs
1. Click on any song from search results
2. See the top 10 most similar songs with match percentages
3. Explore "More from this Artist" section

### Understanding Match Scores
- **90-100%**: Extremely similar (same mood, tempo, energy)
- **75-89%**: Very similar (comparable vibe)
- **60-74%**: Moderately similar (some shared characteristics)
- **Below 60%**: Loosely related

## 🛠️ Advanced Usage

### Custom Training Parameters

```bash
# Train with custom settings
python manage.py train_recommender \
  --csv music/data/spotify_tracks.csv \
  --out music/artifacts \
  --neighbors 20 \
  --clusters 8
```

### Using Your Own Dataset

1. Prepare a CSV with these columns:
   - Required: `track_name`, `artist_name`, `id` (Spotify track ID)
   - Audio features: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`
   - Optional: `album_name`, `popularity`, `year`

2. Place it in `music/data/`

3. Train the model:
   ```bash
   python manage.py train_recommender --csv music/data/your_dataset.csv
   ```

## 🧪 Testing

Run accuracy evaluation tests:
```bash
python manage.py test music
```

This validates:
- CSV loading and cleaning
- Feature extraction and scaling
- Model training and prediction
- Similarity calculation accuracy
- Recommendation diversity

## 🔧 Configuration

Edit `MusicAI/settings.py` to customize:
- Database settings
- Static files location
- Allowed hosts
- Debug mode

## 📦 Dependencies

Core packages:
- **Django 5.2.4**: Web framework
- **pandas 2.3.1**: Data manipulation
- **scikit-learn 1.7.1**: Machine learning models
- **numpy 2.3.2**: Numerical computing
- **joblib 1.5.1**: Model persistence

See `requirements.txt` for complete list.

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

### "No such file or directory: spotify_tracks.csv"
- Verify `music/data/spotify_tracks.csv` exists
- Check file path in training command

### "No artifacts found" error on startup
- Run `python manage.py train_recommender` first
- Models must be trained before running the server

### Search returns no results
- Dataset might not contain the artist/song
- Try partial names or alternative spellings
- Check if CSV loaded correctly

## 🚀 Deployment

For production deployment:

1. Set `DEBUG = False` in settings.py
2. Configure `ALLOWED_HOSTS`
3. Use proper database (PostgreSQL/MySQL)
4. Set up static file serving
5. Use production WSGI server (Gunicorn/uWSGI)
6. Configure SECRET_KEY via environment variable

## 📝 License

This project is for educational purposes.

## 👤 Author

**Dhruvin Dave**
- GitHub: [@dhruvindave007](https://github.com/dhruvindave007)

## 🙏 Acknowledgments

- Spotify for audio feature data
- scikit-learn community
- Django framework

## 📈 Future Enhancements

- [ ] Add user accounts and favorite tracking
- [ ] Playlist generation
- [ ] Real-time Spotify API integration
- [ ] Deep learning models (neural collaborative filtering)
- [ ] Genre-based filtering
- [ ] Export recommendations to Spotify
- [ ] Audio preview integration
- [ ] A/B testing framework for algorithm improvements

---

**Note**: This system uses audio feature similarity, not collaborative filtering. It finds songs that *sound* similar, not songs that similar users liked.
