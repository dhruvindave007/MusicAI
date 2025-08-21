#  MusicAI - Hybrid Music Recommendation System

A Django-based Music Recommendation System that suggests **similar songs** and shows **more from the same artist(s)** using a **hybrid AI approach** (content-based features + artist similarity).  

---

## Features
-  Search songs by title or artist  
-  Song details page with embedded Spotify player  
-  AI-powered recommendations:  
  - Section 1 → **Similar Songs (across dataset)**  
  - Section 2 → **More from the Same Artist(s)**  
-  Uses `spotify_tracks.csv` dataset for feature-based similarity  

---

## Tech Stack
- **Backend** → Django (Python)  
- **Frontend** → HTML, CSS, Bootstrap (inside templates)  
- **AI/ML** → scikit-learn (cosine similarity)  
- **Database** → SQLite (default Django)  
- **Dataset** → `spotify_tracks.csv`  

---

##  Project Structure
```bash
MusicAI/
│── music/ # Main Django app
│ ├── views.py # Search, song detail, recommendations
│ ├── utils.py # AI recommendation logic
│ ├── templates/music/ # HTML templates
│ └── static/ # CSS, JS, images
│── spotify_tracks.csv # Dataset
│── manage.py
│── requirements.txt
│── README.md
```

---

##  Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/MusicAI.git
cd MusicAI
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Screenshots
### 1. Search Bar
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/af641477-6ef3-4dfe-a0b2-16c0b2a08557" />


### 2. search results
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/d419b5ba-b424-41ec-9ed2-1ab64a809e99" />

### 3. Song Details and recomendation with Artist section
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/531b2f77-b64c-441a-b627-fa662e181948" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/1e710c77-5c10-45f0-8aaf-2685acfb3d7c" />

## Author
### Dhruvin Krutarthkumar Dave
- [Linkedin](https://www.linkedin.com/in/mrdhruvindave/)
- [Email ME](mailto:davedhruvin307@gmail.com) 

