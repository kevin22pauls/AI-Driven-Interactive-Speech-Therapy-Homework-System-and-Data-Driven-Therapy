# services/audio.py
import os

def preprocess_audio(file_path):
    """
    For browser-recorded audio (webm/mp3),
    do NOT try to read with soundfile.
    Let ffmpeg/faster-whisper handle decoding.
    """
    return file_path
