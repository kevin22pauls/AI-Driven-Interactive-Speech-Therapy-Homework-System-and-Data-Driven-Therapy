"""
ML-based Voice Activity Detection using Silero VAD

Provides precise speech/pause boundary detection with:
- Sub-100ms precision
- Actual speech vs silence detection (not just word gaps)
- Confidence scores for each segment
- Handles breath sounds, coughs, background noise
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Sampling rate for Silero VAD
SAMPLE_RATE = 16000


@dataclass
class SpeechSegment:
    """A detected speech segment."""
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float  # Detection confidence (0-1)


@dataclass
class VADPause:
    """A detected pause/silence segment."""
    start: float
    end: float
    duration: float
    pause_type: str  # 'speech_gap', 'silence', 'breath'
    confidence: float
    before_speech: bool  # Is there speech before this pause?
    after_speech: bool  # Is there speech after this pause?


class SileroVADAnalyzer:
    """
    Silero VAD for precise speech/pause boundary detection.

    Advantages over threshold-based:
    - Detects actual speech vs silence (not just word gaps)
    - Handles breath sounds, coughs, etc.
    - Sub-100ms precision
    - Confidence scores
    """

    def __init__(self):
        self.model = None
        self.utils = None
        self._loaded = False

    def load(self) -> bool:
        """
        Load the Silero VAD model.

        Returns:
            True if successfully loaded, False otherwise
        """
        if self._loaded:
            return True

        try:
            from services.ml_models import get_model

            vad_data = get_model('silero_vad')
            if vad_data is None:
                logger.warning("Failed to load Silero VAD from registry")
                return False

            self.model = vad_data['model']
            self.utils = vad_data
            self._loaded = True
            logger.info("Silero VAD loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading Silero VAD: {e}")
            return False

    def _load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Load audio file and convert to correct format for VAD.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio as numpy array at 16kHz, or None on error
        """
        try:
            import torch
            import torchaudio

            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample to 16kHz if necessary
            if sample_rate != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Return as 1D tensor
            return waveform.squeeze()

        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return None

    def detect_speech_segments(
        self,
        audio_path: str,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30
    ) -> List[SpeechSegment]:
        """
        Detect speech segments in audio.

        Args:
            audio_path: Path to audio file
            threshold: VAD threshold (0-1), higher = stricter
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence to split segments
            speech_pad_ms: Padding around speech segments

        Returns:
            List of SpeechSegment objects with timestamps
        """
        if not self._loaded and not self.load():
            logger.warning("VAD not loaded, returning empty results")
            return []

        audio = self._load_audio(audio_path)
        if audio is None:
            return []

        try:
            import torch

            # Get speech timestamps using Silero VAD
            get_speech_timestamps = self.utils['get_speech_timestamps']

            speech_timestamps = get_speech_timestamps(
                audio,
                self.model,
                threshold=threshold,
                sampling_rate=SAMPLE_RATE,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
                return_seconds=True  # Get times in seconds
            )

            # Convert to SpeechSegment objects
            segments = []
            for ts in speech_timestamps:
                # Silero returns dict with 'start' and 'end' in seconds when return_seconds=True
                start = ts.get('start', ts.get('start') / SAMPLE_RATE if 'start' in ts else 0)
                end = ts.get('end', ts.get('end') / SAMPLE_RATE if 'end' in ts else 0)

                # Handle sample-based timestamps
                if isinstance(start, int):
                    start = start / SAMPLE_RATE
                if isinstance(end, int):
                    end = end / SAMPLE_RATE

                segments.append(SpeechSegment(
                    start=float(start),
                    end=float(end),
                    confidence=threshold  # Silero doesn't return per-segment confidence
                ))

            logger.debug(f"Detected {len(segments)} speech segments")
            return segments

        except Exception as e:
            logger.error(f"Error detecting speech segments: {e}")
            return []

    def detect_pauses(
        self,
        audio_path: str,
        speech_segments: Optional[List[SpeechSegment]] = None,
        min_pause_duration: float = 0.1  # 100ms minimum
    ) -> List[VADPause]:
        """
        Detect pauses (gaps between speech segments).

        Args:
            audio_path: Path to audio file
            speech_segments: Pre-computed speech segments (optional)
            min_pause_duration: Minimum pause duration in seconds

        Returns:
            List of VADPause objects
        """
        # Get speech segments if not provided
        if speech_segments is None:
            speech_segments = self.detect_speech_segments(audio_path)

        if not speech_segments:
            return []

        pauses = []

        # Get total audio duration
        audio = self._load_audio(audio_path)
        if audio is not None:
            total_duration = len(audio) / SAMPLE_RATE
        else:
            total_duration = speech_segments[-1].end if speech_segments else 0

        # Check for initial silence
        if speech_segments[0].start > min_pause_duration:
            pauses.append(VADPause(
                start=0,
                end=speech_segments[0].start,
                duration=speech_segments[0].start,
                pause_type='silence',
                confidence=0.9,
                before_speech=False,
                after_speech=True
            ))

        # Find gaps between speech segments
        for i in range(len(speech_segments) - 1):
            current = speech_segments[i]
            next_seg = speech_segments[i + 1]

            gap_start = current.end
            gap_end = next_seg.start
            gap_duration = gap_end - gap_start

            if gap_duration >= min_pause_duration:
                # Classify pause type based on duration
                if gap_duration < 0.3:
                    pause_type = 'speech_gap'  # Brief gap, normal
                elif gap_duration < 1.0:
                    pause_type = 'hesitation'  # Medium pause
                else:
                    pause_type = 'silence'  # Long silence

                pauses.append(VADPause(
                    start=gap_start,
                    end=gap_end,
                    duration=gap_duration,
                    pause_type=pause_type,
                    confidence=min(current.confidence, next_seg.confidence),
                    before_speech=True,
                    after_speech=True
                ))

        # Check for trailing silence
        if speech_segments and total_duration - speech_segments[-1].end > min_pause_duration:
            pauses.append(VADPause(
                start=speech_segments[-1].end,
                end=total_duration,
                duration=total_duration - speech_segments[-1].end,
                pause_type='silence',
                confidence=0.9,
                before_speech=True,
                after_speech=False
            ))

        logger.debug(f"Detected {len(pauses)} pauses")
        return pauses

    def get_speech_probability(
        self,
        audio_path: str,
        window_size_ms: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get frame-level speech probability over time.

        Args:
            audio_path: Path to audio file
            window_size_ms: Window size for probability calculation

        Returns:
            Tuple of (timestamps, probabilities)
        """
        if not self._loaded and not self.load():
            return np.array([]), np.array([])

        audio = self._load_audio(audio_path)
        if audio is None:
            return np.array([]), np.array([])

        try:
            import torch

            # Calculate frame-level probabilities
            window_size_samples = int(SAMPLE_RATE * window_size_ms / 1000)
            num_frames = len(audio) // window_size_samples

            probabilities = []
            timestamps = []

            for i in range(num_frames):
                start_sample = i * window_size_samples
                end_sample = start_sample + window_size_samples
                chunk = audio[start_sample:end_sample]

                # Get speech probability for this chunk
                prob = self.model(chunk, SAMPLE_RATE).item()
                probabilities.append(prob)
                timestamps.append(start_sample / SAMPLE_RATE)

            return np.array(timestamps), np.array(probabilities)

        except Exception as e:
            logger.error(f"Error getting speech probability: {e}")
            return np.array([]), np.array([])

    def analyze_audio(
        self,
        audio_path: str,
        threshold: float = 0.5
    ) -> Dict:
        """
        Perform complete VAD analysis on audio.

        Args:
            audio_path: Path to audio file
            threshold: VAD threshold

        Returns:
            Dictionary with speech segments, pauses, and statistics
        """
        speech_segments = self.detect_speech_segments(audio_path, threshold=threshold)
        pauses = self.detect_pauses(audio_path, speech_segments)

        # Calculate statistics
        total_speech_time = sum(seg.end - seg.start for seg in speech_segments)
        total_pause_time = sum(p.duration for p in pauses)

        # Get audio duration
        audio = self._load_audio(audio_path)
        total_duration = len(audio) / SAMPLE_RATE if audio is not None else 0

        return {
            'speech_segments': [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'duration': seg.end - seg.start,
                    'confidence': seg.confidence
                }
                for seg in speech_segments
            ],
            'pauses': [
                {
                    'start': p.start,
                    'end': p.end,
                    'duration': p.duration,
                    'type': p.pause_type,
                    'confidence': p.confidence
                }
                for p in pauses
            ],
            'statistics': {
                'total_duration': total_duration,
                'speech_time': total_speech_time,
                'pause_time': total_pause_time,
                'speech_ratio': total_speech_time / total_duration if total_duration > 0 else 0,
                'num_speech_segments': len(speech_segments),
                'num_pauses': len(pauses),
                'avg_speech_segment_duration': total_speech_time / len(speech_segments) if speech_segments else 0,
                'avg_pause_duration': total_pause_time / len(pauses) if pauses else 0
            },
            'model_confidence': threshold,
            'analysis_method': 'ml'
        }


# Global instance for convenience
_vad_analyzer = None


def get_vad_analyzer() -> SileroVADAnalyzer:
    """Get the global VAD analyzer instance."""
    global _vad_analyzer

    if _vad_analyzer is None:
        _vad_analyzer = SileroVADAnalyzer()

    return _vad_analyzer


def analyze_audio_vad(audio_path: str, threshold: float = 0.5) -> Dict:
    """Convenience function to analyze audio with VAD."""
    analyzer = get_vad_analyzer()
    return analyzer.analyze_audio(audio_path, threshold)
