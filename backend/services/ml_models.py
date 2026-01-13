"""
ML Model Registry for Speech Therapy System

Centralized model management with:
- Lazy loading on first use
- Thread-safe singleton pattern
- Memory management for CPU systems
- Graceful fallback flags
"""

import os
# Set espeak-ng library path for phonemizer (must be set before importing transformers phoneme tokenizer)
if os.path.exists(r'C:\Program Files\eSpeak NG\libespeak-ng.dll'):
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = r'C:\Program Files\eSpeak NG\libespeak-ng.dll'
    os.environ['PHONEMIZER_ESPEAK_PATH'] = r'C:\Program Files\eSpeak NG'

import logging
import threading
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Global model registry instance
_registry = None
_registry_lock = threading.Lock()


class ModelRegistry:
    """
    Singleton registry for ML models.

    Models are loaded lazily on first use and cached for reuse.
    Provides fallback flags when models fail to load.
    """

    # Model configurations
    MODEL_CONFIGS = {
        'silero_vad': {
            'description': 'Silero VAD for voice activity detection',
            'max_memory_mb': 50,
            'load_func': '_load_silero_vad'
        },
        'wav2vec2_phoneme': {
            'description': 'Wav2Vec2 for phoneme recognition',
            'repo': 'facebook/wav2vec2-lv-60-espeak-cv-ft',
            'max_memory_mb': 400,
            'load_func': '_load_wav2vec2_phoneme'
        },
        'wavlm_base': {
            'description': 'WavLM for acoustic feature extraction',
            'repo': 'microsoft/wavlm-base',
            'max_memory_mb': 400,
            'load_func': '_load_wavlm'
        },
        'sentence_transformer': {
            'description': 'Sentence transformer for semantic similarity',
            'repo': 'sentence-transformers/all-mpnet-base-v2',
            'max_memory_mb': 450,
            'load_func': '_load_sentence_transformer'
        },
        'cross_encoder': {
            'description': 'Cross-encoder for accurate similarity scoring',
            'repo': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'max_memory_mb': 100,
            'load_func': '_load_cross_encoder'
        }
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the model registry.

        Args:
            cache_dir: Directory for caching downloaded models.
                      Defaults to backend/models/cache
        """
        self._models: Dict[str, Any] = {}
        self._fallback_flags: Dict[str, bool] = {}
        self._last_used: Dict[str, float] = {}
        self._lock = threading.Lock()

        # Set cache directory
        if cache_dir is None:
            base_dir = Path(__file__).parent.parent
            cache_dir = base_dir / "models" / "cache"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variable for transformers cache
        os.environ['TRANSFORMERS_CACHE'] = str(self._cache_dir)
        os.environ['HF_HOME'] = str(self._cache_dir)

        logger.info(f"ModelRegistry initialized with cache at {self._cache_dir}")

    def get_model(self, model_name: str) -> Optional[Any]:
        """
        Get a model by name, loading it if necessary.

        Args:
            model_name: Name of the model to get

        Returns:
            The loaded model, or None if loading failed
        """
        import time

        with self._lock:
            # Check if already loaded
            if model_name in self._models:
                self._last_used[model_name] = time.time()
                return self._models[model_name]

            # Check if previously failed
            if self._fallback_flags.get(model_name, False):
                logger.debug(f"Model {model_name} previously failed, using fallback")
                return None

            # Try to load
            if model_name not in self.MODEL_CONFIGS:
                logger.error(f"Unknown model: {model_name}")
                return None

            config = self.MODEL_CONFIGS[model_name]
            load_func = getattr(self, config['load_func'], None)

            if load_func is None:
                logger.error(f"No load function for model: {model_name}")
                return None

            try:
                logger.info(f"Loading model: {model_name}")
                model = load_func()
                self._models[model_name] = model
                self._last_used[model_name] = time.time()
                logger.info(f"Successfully loaded model: {model_name}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                self._fallback_flags[model_name] = True
                return None

    def is_available(self, model_name: str) -> bool:
        """Check if a model is available (loaded or loadable)."""
        if model_name in self._models:
            return True
        if self._fallback_flags.get(model_name, False):
            return False
        return model_name in self.MODEL_CONFIGS

    def use_fallback(self, model_name: str) -> bool:
        """Check if we should use fallback for a model."""
        return self._fallback_flags.get(model_name, False)

    def unload_model(self, model_name: str):
        """Unload a model to free memory."""
        with self._lock:
            if model_name in self._models:
                del self._models[model_name]
                logger.info(f"Unloaded model: {model_name}")
                import gc
                gc.collect()

    def _load_silero_vad(self):
        """Load Silero VAD model."""
        import torch

        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False  # Use PyTorch for CPU
        )

        # Store utils alongside model
        return {
            'model': model,
            'get_speech_timestamps': utils[0],
            'save_audio': utils[1],
            'read_audio': utils[2],
            'VADIterator': utils[3],
            'collect_chunks': utils[4]
        }

    def _load_wav2vec2_phoneme(self):
        """Load Wav2Vec2 model for phoneme recognition."""
        from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
        from transformers import Wav2Vec2PhonemeCTCTokenizer

        model_id = self.MODEL_CONFIGS['wav2vec2_phoneme']['repo']

        # Load tokenizer and feature extractor separately
        # Use Wav2Vec2PhonemeCTCTokenizer for phoneme models
        tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
            model_id,
            cache_dir=str(self._cache_dir)
        )
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_id,
            cache_dir=str(self._cache_dir)
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            model_id,
            cache_dir=str(self._cache_dir)
        )
        model.eval()

        return {
            'model': model,
            'tokenizer': tokenizer,
            'feature_extractor': feature_extractor
        }

    def _load_wavlm(self):
        """Load WavLM model for acoustic features."""
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor

        model_id = self.MODEL_CONFIGS['wavlm_base']['repo']
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_id,
            cache_dir=str(self._cache_dir)
        )
        model = WavLMModel.from_pretrained(
            model_id,
            cache_dir=str(self._cache_dir)
        )
        model.eval()

        return {
            'model': model,
            'feature_extractor': feature_extractor
        }

    def _load_sentence_transformer(self):
        """Load sentence transformer for semantic similarity."""
        from sentence_transformers import SentenceTransformer

        model_id = self.MODEL_CONFIGS['sentence_transformer']['repo']
        model = SentenceTransformer(
            model_id,
            cache_folder=str(self._cache_dir)
        )

        return model

    def _load_cross_encoder(self):
        """Load cross-encoder for accurate similarity scoring."""
        from sentence_transformers import CrossEncoder

        model_id = self.MODEL_CONFIGS['cross_encoder']['repo']
        model = CrossEncoder(
            model_id,
            max_length=512
        )

        return model


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry

    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()

    return _registry


def get_model(model_name: str) -> Optional[Any]:
    """Convenience function to get a model from the global registry."""
    return get_registry().get_model(model_name)


def is_model_available(model_name: str) -> bool:
    """Check if a model is available."""
    return get_registry().is_available(model_name)


def use_fallback(model_name: str) -> bool:
    """Check if we should use fallback for a model."""
    return get_registry().use_fallback(model_name)


# Fallback decorator
def with_fallback(fallback_func):
    """
    Decorator that falls back to a rule-based function if ML fails.

    Usage:
        @with_fallback(rule_based_function)
        def ml_function(*args, **kwargs):
            # ML implementation
    """
    import functools

    def decorator(ml_func):
        @functools.wraps(ml_func)
        def wrapper(*args, **kwargs):
            try:
                result = ml_func(*args, **kwargs)
                if isinstance(result, dict):
                    result['analysis_method'] = 'ml'
                return result
            except Exception as e:
                logger.warning(f"ML function {ml_func.__name__} failed: {e}")
                logger.info(f"Falling back to {fallback_func.__name__}")
                result = fallback_func(*args, **kwargs)
                if isinstance(result, dict):
                    result['analysis_method'] = 'rule_based'
                return result
        return wrapper
    return decorator


def preload_models(model_names: Optional[list] = None):
    """
    Preload specified models at startup.

    Args:
        model_names: List of model names to preload.
                    If None, loads all models.
    """
    registry = get_registry()

    if model_names is None:
        model_names = list(ModelRegistry.MODEL_CONFIGS.keys())

    for name in model_names:
        logger.info(f"Preloading model: {name}")
        registry.get_model(name)
