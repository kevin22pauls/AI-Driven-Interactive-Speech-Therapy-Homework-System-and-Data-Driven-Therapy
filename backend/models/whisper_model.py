from faster_whisper import WhisperModel
from config import WHISPER_MODEL_SIZE

_whisper_model = None

def load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )
    return _whisper_model

def get_whisper_model():
    return _whisper_model
