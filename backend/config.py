import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_STORAGE = os.path.join(BASE_DIR, "storage", "audio")
os.makedirs(AUDIO_STORAGE, exist_ok=True)

DB_URL = "sqlite:///./speechtherapy.db"
WHISPER_MODEL_SIZE = "small"   # or "tiny"
