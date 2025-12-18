# utils/file_utils.py
import os
from config import AUDIO_STORAGE
from datetime import datetime
import mimetypes

def save_audio_file(file_bytes, filename_prefix="audio", filename=None):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if filename:
        ext = os.path.splitext(filename)[1] or ".wav"
    else:
        ext = ".wav"
    file_path = os.path.join(AUDIO_STORAGE, f"{filename_prefix}_{timestamp}{ext}")

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    return file_path
