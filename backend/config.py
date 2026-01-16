import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_STORAGE = os.path.join(BASE_DIR, "storage", "audio")
os.makedirs(AUDIO_STORAGE, exist_ok=True)

DB_URL = "sqlite:///./speechtherapy.db"
WHISPER_MODEL_SIZE = "small"   # or "tiny"

# ============================================================================
# ML Model Configuration
# ============================================================================

ML_CONFIG = {
    # Model cache directory
    'model_cache_dir': os.path.join(BASE_DIR, "models", "cache"),

    # Enable/disable ML components (for testing/fallback)
    'use_ml_vad': True,              # Silero VAD for pause detection
    'use_ml_stutter': True,          # ML stuttering detection
    'use_ml_phoneme': True,          # WavLM/Wav2Vec2 for phoneme analysis
    'use_ml_semantic': True,         # Cross-encoder for semantic scoring
    'use_word_timestamps': True,     # Whisper word-level timestamps

    # Inference settings
    'device': 'cpu',                 # 'cpu' or 'cuda'

    # ============================================================================
    # SONIVA Whisper Model (Fine-tuned for Aphasia Speech)
    # ============================================================================
    # The SONIVA model is a fine-tuned Whisper Medium model trained on post-stroke
    # aphasia speech, achieving significantly better WER on aphasic speech.
    # Download from: https://github.com/Clinical-Language-Cognition-Lab/SONIVA_paper4
    #
    # To use: Download the model and set the path below, or use HuggingFace ID
    # Example paths:
    #   - Local: os.path.join(BASE_DIR, "models", "soniva-whisper-medium")
    #   - HuggingFace: "Clinical-Language-Cognition-Lab/soniva-whisper-medium" (if available)
    'use_soniva_whisper': False,     # Disabled - SONIVA model not available (requires Google Drive access)
    'soniva_model_path': os.path.join(BASE_DIR, "models", "soniva-whisper-medium"),  # Path to SONIVA weights

    # Model variants (use smaller for CPU)
    'wavlm_model': 'microsoft/wavlm-base',  # Not wavlm-large for CPU
    'wav2vec2_phoneme_model': 'facebook/wav2vec2-lv-60-espeak-cv-ft',
    'sbert_model': 'sentence-transformers/all-mpnet-base-v2',
    'cross_encoder_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',

    # VAD settings
    'vad_threshold': 0.5,            # Speech detection threshold (0-1)
    'vad_min_speech_ms': 250,        # Minimum speech segment duration
    'vad_min_silence_ms': 100,       # Minimum silence to split segments
}

# Create model cache directory
os.makedirs(ML_CONFIG['model_cache_dir'], exist_ok=True)
