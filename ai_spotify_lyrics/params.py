import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Local Data
LOCAL_DATA_PATH = os.path.join(ROOT_DIR, "raw_data")
DATA_CSV_INIT = os.path.join(LOCAL_DATA_PATH, "database_20250606_1630.csv")
DATA_CSV_17k = os.path.join(LOCAL_DATA_PATH, "data_17k_lyrics.csv")
DATA_CSV_17k_EMBED = os.path.join(LOCAL_DATA_PATH, "embedded_17Klyrics.csv")
DATA_9K = os.path.join(LOCAL_DATA_PATH, "processed_df.pkl")

# Pipeline and KNN model for feature 3
LOCAL_PREPARE_PATH = os.path.join(ROOT_DIR, "ai_spotify_lyrics/prepare")
PIPELINE = os.path.join(LOCAL_PREPARE_PATH, "pipe.joblib")
KNN_MODEL = os.path.join(LOCAL_PREPARE_PATH, "knn_model.joblib")



# Local Models
MODEL_TARGET = os.environ.get('MODEL_TARGET')
LOCAL_REGISTRY_PATH =  os.path.join(ROOT_DIR, "models")
