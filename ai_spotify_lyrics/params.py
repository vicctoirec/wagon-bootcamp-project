import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Local Data
LOCAL_DATA_PATH = os.path.join(ROOT_DIR, "raw_data")
DATA_CSV_INIT = os.path.join(LOCAL_DATA_PATH, "database_20250606_1630.csv")

# Local Models
MODEL_TARGET = os.environ['MODEL_TARGET']
LOCAL_REGISTRY_PATH =  os.path.join(ROOT_DIR, "models")
