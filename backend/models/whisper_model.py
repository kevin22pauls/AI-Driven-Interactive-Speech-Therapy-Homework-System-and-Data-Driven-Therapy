"""
Whisper ASR Model Loader

Supports:
1. SONIVA fine-tuned Whisper (HuggingFace Transformers) - optimized for aphasia speech
2. Faster-Whisper (CTranslate2) - fallback option

The SONIVA model is a fine-tuned version of OpenAI's Whisper Medium model
specifically adapted for post-stroke aphasia speech recognition.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Iterator, Tuple, Any

logger = logging.getLogger(__name__)

# Global model instance
_whisper_model = None
_model_type = None  # 'soniva' or 'faster_whisper'


@dataclass
class WordTiming:
    """Word-level timing information"""
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class TranscriptionSegment:
    """Segment of transcribed audio"""
    start: float
    end: float
    text: str
    words: Optional[List[WordTiming]] = None


@dataclass
class TranscriptionInfo:
    """Metadata about the transcription"""
    language: str
    language_probability: float
    duration: float


class SonivaWhisperModel:
    """
    SONIVA Fine-Tuned Whisper Model wrapper

    This model is specifically trained on post-stroke aphasia speech from the SONIVA dataset,
    achieving significantly better WER on aphasic speech compared to standard Whisper.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the SONIVA Whisper model.

        Args:
            model_path: Path to the fine-tuned model weights (local directory or HuggingFace ID)
            device: 'cpu' or 'cuda'
        """
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        self.device = device
        logger.info(f"Loading SONIVA Whisper model from {model_path}...")

        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        # Get the forced decoder IDs for English transcription
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="english",
            task="transcribe"
        )

        logger.info("SONIVA Whisper model loaded successfully")

    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        language: str = "en",
        vad_filter: bool = True,
        vad_parameters: dict = None,
        word_timestamps: bool = False,
        **kwargs
    ) -> Tuple[Iterator[TranscriptionSegment], TranscriptionInfo]:
        """
        Transcribe audio file using SONIVA model.

        Args:
            audio_path: Path to audio file
            beam_size: Beam size for decoding
            language: Language code (default: 'en')
            vad_filter: Whether to filter silence (handled by preprocessing)
            vad_parameters: VAD parameters (for compatibility)
            word_timestamps: Whether to return word-level timestamps

        Returns:
            Tuple of (segments iterator, transcription info)
        """
        import torch
        import librosa
        import numpy as np

        # Load and preprocess audio
        try:
            # Try loading with librosa (supports many formats)
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            logger.warning(f"librosa failed to load {audio_path}: {e}, trying soundfile")
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

        duration = len(audio) / sr

        # Process audio through Whisper processor
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)

        # Generate transcription
        with torch.no_grad():
            # Use generate with beam search
            generated_ids = self.model.generate(
                input_features,
                forced_decoder_ids=self.forced_decoder_ids,
                max_length=448,
                num_beams=beam_size,
                return_timestamps=word_timestamps,
            )

        # Decode the transcription
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # Create segment (SONIVA model returns full transcription, not segments)
        # For word timestamps, we'd need additional processing
        segments = []

        if word_timestamps and hasattr(self.processor, 'decode'):
            # Try to get word-level timestamps if available
            try:
                # Decode with timestamps
                result = self.processor.decode(
                    generated_ids[0],
                    output_offsets=True,
                    skip_special_tokens=True
                )

                if hasattr(result, 'offsets') and result.offsets:
                    words = []
                    for offset in result.offsets:
                        words.append(WordTiming(
                            word=offset.get('text', ''),
                            start=offset.get('start_offset', 0) / 16000,  # Convert samples to seconds
                            end=offset.get('end_offset', 0) / 16000,
                            probability=1.0
                        ))

                    segment = TranscriptionSegment(
                        start=0.0,
                        end=duration,
                        text=transcription.strip(),
                        words=words
                    )
                else:
                    segment = TranscriptionSegment(
                        start=0.0,
                        end=duration,
                        text=transcription.strip(),
                        words=None
                    )
            except Exception as e:
                logger.warning(f"Could not extract word timestamps: {e}")
                segment = TranscriptionSegment(
                    start=0.0,
                    end=duration,
                    text=transcription.strip(),
                    words=None
                )
        else:
            segment = TranscriptionSegment(
                start=0.0,
                end=duration,
                text=transcription.strip(),
                words=None
            )

        segments.append(segment)

        info = TranscriptionInfo(
            language=language,
            language_probability=1.0,
            duration=duration
        )

        return iter(segments), info


class FasterWhisperWrapper:
    """
    Wrapper for faster-whisper to provide consistent interface
    """

    def __init__(self, model_size: str, device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        logger.info(f"Loading Faster-Whisper model ({model_size})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Faster-Whisper model loaded successfully")

    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        language: str = "en",
        vad_filter: bool = True,
        vad_parameters: dict = None,
        word_timestamps: bool = False,
        **kwargs
    ) -> Tuple[Iterator[Any], Any]:
        """
        Transcribe using faster-whisper (returns native types)
        """
        return self.model.transcribe(
            audio_path,
            beam_size=beam_size,
            language=language,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters or dict(min_silence_duration_ms=500),
            word_timestamps=word_timestamps
        )


def load_whisper_model(force_type: str = None):
    """
    Load the Whisper ASR model.

    Priority:
    1. SONIVA fine-tuned model (if available) - best for aphasia
    2. Faster-Whisper (fallback) - general purpose

    Args:
        force_type: Force a specific model type ('soniva' or 'faster_whisper')

    Returns:
        Loaded model instance
    """
    global _whisper_model, _model_type

    if _whisper_model is not None:
        return _whisper_model

    from config import WHISPER_MODEL_SIZE, ML_CONFIG

    # Check for SONIVA model configuration
    soniva_model_path = ML_CONFIG.get('soniva_model_path', None)
    use_soniva = ML_CONFIG.get('use_soniva_whisper', True)
    device = ML_CONFIG.get('device', 'cpu')

    # Determine which model to use
    if force_type == 'faster_whisper':
        use_soniva = False
    elif force_type == 'soniva':
        use_soniva = True

    # Try SONIVA model first
    if use_soniva and soniva_model_path:
        try:
            _whisper_model = SonivaWhisperModel(soniva_model_path, device=device)
            _model_type = 'soniva'
            logger.info("Using SONIVA fine-tuned Whisper model (optimized for aphasia)")
            return _whisper_model
        except Exception as e:
            logger.warning(f"Failed to load SONIVA model: {e}")
            logger.info("Falling back to Faster-Whisper")

    # Fallback to faster-whisper
    try:
        _whisper_model = FasterWhisperWrapper(
            WHISPER_MODEL_SIZE,
            device=device,
            compute_type="int8"
        )
        _model_type = 'faster_whisper'
        logger.info(f"Using Faster-Whisper model ({WHISPER_MODEL_SIZE})")
        return _whisper_model
    except Exception as e:
        logger.error(f"Failed to load Faster-Whisper: {e}")
        raise


def get_whisper_model():
    """Get the loaded Whisper model instance."""
    global _whisper_model
    if _whisper_model is None:
        return load_whisper_model()
    return _whisper_model


def get_model_type() -> str:
    """Get the type of currently loaded model."""
    global _model_type
    return _model_type or 'unknown'
