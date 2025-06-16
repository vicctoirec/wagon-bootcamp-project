import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Local Data
LOCAL_DATA_PATH = os.path.join(ROOT_DIR, "raw_data")
DATA_CSV_INIT = os.path.join(LOCAL_DATA_PATH, "database_20250606_1630.csv")
DATA_CSV_17k = os.path.join(LOCAL_DATA_PATH, "data_17k_lyrics.csv")
DATA_CSV_17k_EMBED = os.path.join(LOCAL_DATA_PATH, "miniLM_17Klyrics.csv")
DATA_9K = os.path.join(LOCAL_DATA_PATH, "processed_df.pkl")

# Local Models
MODEL_TARGET = os.environ.get('MODEL_TARGET')
LOCAL_REGISTRY_PATH =  os.path.join(ROOT_DIR, "models")

# Pipeline and KNN model for SimilarSongs
LOCAL_PREPARE_PATH = os.path.join(ROOT_DIR, "ai_spotify_lyrics/prepare")
SIMILAR_SONGS_PIPELINE = os.path.join(LOCAL_REGISTRY_PATH, "pipe.joblib")
SIMILAR_SONGS_MODEL = os.path.join(LOCAL_REGISTRY_PATH, "knn_model.joblib")

# LyricsMatching
LYRICS_MATCHING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LYRICS_MATCHING_MODEL_PATH = os.path.join(LOCAL_REGISTRY_PATH, 'all-MiniLM-L6-v2')

# ZeroShot
ZEROSHOT_MODEL_NAME = "facebook/bart-large-mnli"
ZEROSHOT_MODEL_PATH = os.path.join(LOCAL_REGISTRY_PATH, 'bart-large-mnli')
