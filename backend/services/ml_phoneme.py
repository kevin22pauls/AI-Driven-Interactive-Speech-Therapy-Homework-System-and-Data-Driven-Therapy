"""
ML-based Phoneme Analysis using Wav2Vec2/WavLM

Provides acoustic phoneme analysis with:
- Direct acoustic-to-phoneme mapping
- GOP (Goodness of Pronunciation) scores
- Frame-level phoneme probabilities
- Works even when transcription is incorrect

Uses facebook/wav2vec2-lv-60-espeak-cv-ft for phoneme recognition.
This model outputs IPA phonemes which are converted to ARPAbet for
consistent comparison with text-based analysis (CMUdict).
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from services.phoneme_mapping import (
    convert_ipa_sequence_to_arpabet,
    ipa_to_arpabet,
    normalize_phoneme_for_comparison
)

logger = logging.getLogger(__name__)


@dataclass
class PhonemeScore:
    """Score for a single phoneme."""
    phoneme: str           # Detected phoneme (ARPAbet normalized)
    phoneme_ipa: str       # Original IPA from model
    expected: str          # Expected phoneme (ARPAbet)
    gop_score: float       # Goodness of pronunciation (-inf to 0, higher is better)
    probability: float     # Raw probability from model
    duration: float        # Duration in seconds
    start_time: float
    end_time: float


@dataclass
class MLPhonemeResult:
    """Result from ML phoneme analysis."""
    detected_phonemes: List[str]      # ARPAbet normalized
    detected_phonemes_ipa: List[str]  # Original IPA from model
    expected_phonemes: List[str]      # ARPAbet from CMUdict
    phoneme_scores: List[PhonemeScore]
    overall_gop: float
    per_ml: float  # ML-based phoneme error rate
    alignment: List[Tuple[str, str]]  # (expected_arpabet, detected_arpabet) pairs
    frame_probabilities: Optional[np.ndarray]
    model_confidence: float


class MLPhonemeAnalyzer:
    """
    Acoustic phoneme analyzer using Wav2Vec2.

    Performs phoneme recognition directly from audio and calculates
    GOP (Goodness of Pronunciation) scores.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None
        self._loaded = False

    def load(self) -> bool:
        """Load the Wav2Vec2 model for phoneme recognition."""
        if self._loaded:
            return True

        try:
            from services.ml_models import get_model

            model_data = get_model('wav2vec2_phoneme')
            if model_data is None:
                logger.warning("Failed to load Wav2Vec2 phoneme model from registry")
                return False

            self.model = model_data['model']
            self.tokenizer = model_data['tokenizer']
            self.feature_extractor = model_data['feature_extractor']
            self._loaded = True
            logger.info("Wav2Vec2 phoneme model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading Wav2Vec2 phoneme model: {e}")
            return False

    def _load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Load and preprocess audio for the model."""
        try:
            import torch
            import torchaudio

            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample to 16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            return waveform.squeeze().numpy()

        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return None

    def _text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to phoneme sequence using CMU dictionary."""
        try:
            # Try CMU dictionary first
            try:
                import nltk
                from nltk.corpus import cmudict
                d = cmudict.dict()
            except:
                nltk.download('cmudict', quiet=True)
                from nltk.corpus import cmudict
                d = cmudict.dict()

            phonemes = []
            words = text.lower().split()

            for word in words:
                clean_word = word.strip('.,!?;:"\'')
                if clean_word in d:
                    # Get first pronunciation variant
                    word_phonemes = d[clean_word][0]
                    # Remove stress markers (numbers) and uppercase
                    word_phonemes = [p.rstrip('0123456789').upper() for p in word_phonemes]
                    phonemes.extend(word_phonemes)

            return phonemes

        except Exception as e:
            logger.warning(f"Phoneme conversion failed: {e}")
            return []

    def analyze_phonemes(
        self,
        audio_path: str,
        expected_text: str
    ) -> Optional[MLPhonemeResult]:
        """
        Analyze phoneme production from audio.

        Args:
            audio_path: Path to audio file
            expected_text: Expected text for comparison

        Returns:
            MLPhonemeResult with phoneme analysis
        """
        if not self._loaded and not self.load():
            logger.warning("Phoneme model not loaded, skipping ML analysis")
            return None

        audio = self._load_audio(audio_path)
        if audio is None:
            return None

        try:
            import torch

            # Process audio using feature extractor
            inputs = self.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            # Get model predictions
            with torch.no_grad():
                logits = self.model(**inputs).logits

            # Get probabilities
            probs = torch.softmax(logits, dim=-1)

            # Decode predicted phonemes using tokenizer
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_phonemes_str = self.tokenizer.batch_decode(predicted_ids)[0]

            # Parse phoneme string - these are IPA symbols
            detected_ipa = self._parse_phoneme_output(predicted_phonemes_str)

            # Convert IPA to ARPAbet for comparison
            detected_arpabet = convert_ipa_sequence_to_arpabet(detected_ipa)

            # Get expected phonemes (already in ARPAbet from CMUdict)
            expected_phonemes = self._text_to_phonemes(expected_text)

            logger.debug(f"Detected IPA: {detected_ipa}")
            logger.debug(f"Detected ARPAbet: {detected_arpabet}")
            logger.debug(f"Expected ARPAbet: {expected_phonemes}")

            # Calculate GOP scores using frame probabilities
            phoneme_scores, overall_gop = self._calculate_gop_scores(
                probs[0].numpy(),
                detected_arpabet,
                detected_ipa,
                expected_phonemes,
                len(audio) / 16000  # Duration in seconds
            )

            # Calculate ML-based PER using proper alignment
            alignment, per_ml = self._align_and_calculate_per(
                expected_phonemes,
                detected_arpabet
            )

            # Calculate overall confidence
            model_confidence = float(torch.mean(torch.max(probs, dim=-1).values).item())

            return MLPhonemeResult(
                detected_phonemes=detected_arpabet,
                detected_phonemes_ipa=detected_ipa,
                expected_phonemes=expected_phonemes,
                phoneme_scores=phoneme_scores,
                overall_gop=overall_gop,
                per_ml=per_ml,
                alignment=alignment,
                frame_probabilities=probs[0].numpy(),
                model_confidence=model_confidence
            )

        except Exception as e:
            logger.error(f"ML phoneme analysis error: {e}", exc_info=True)
            return None

    def _parse_phoneme_output(self, output: str) -> List[str]:
        """
        Parse the phoneme output string from the model.

        The wav2vec2-lv-60-espeak-cv-ft model outputs IPA symbols
        separated by spaces.
        """
        phonemes = []
        for token in output.split():
            # Clean up the token
            token = token.strip()
            if token and token not in ['<s>', '</s>', '<pad>', '|', ' ', '<unk>']:
                # Keep original case for IPA (some symbols are case-sensitive)
                phonemes.append(token)
        return phonemes

    def _calculate_gop_scores(
        self,
        frame_probs: np.ndarray,
        detected_arpabet: List[str],
        detected_ipa: List[str],
        expected: List[str],
        total_duration: float
    ) -> Tuple[List[PhonemeScore], float]:
        """
        Calculate GOP (Goodness of Pronunciation) scores.

        GOP = log P(phoneme | acoustic) - log P(phoneme)

        Higher scores indicate better pronunciation.
        """
        scores = []

        if not detected_arpabet:
            return scores, 0.0

        num_detected = len(detected_arpabet)
        duration_per_phoneme = total_duration / max(num_detected, 1)
        num_frames = len(frame_probs)

        gop_values = []

        # Create alignment for scoring
        aligned_expected = self._get_aligned_expected(expected, detected_arpabet)

        for i in range(num_detected):
            start_time = i * duration_per_phoneme
            end_time = (i + 1) * duration_per_phoneme

            # Get frame range for this phoneme (proportional to position)
            frame_start = int(i * num_frames / num_detected)
            frame_end = int((i + 1) * num_frames / num_detected)
            frame_end = min(frame_end, num_frames)

            if frame_start < frame_end:
                # Get max probability across frames for this segment
                segment_probs = frame_probs[frame_start:frame_end]
                # Average of max probabilities per frame
                avg_prob = float(np.mean(np.max(segment_probs, axis=-1)))

                # Also compute variance to indicate confidence
                prob_variance = float(np.var(np.max(segment_probs, axis=-1)))
            else:
                avg_prob = 0.5
                prob_variance = 0.0

            # GOP: log probability (0 is perfect, negative is worse)
            gop = float(np.log(avg_prob + 1e-10))

            # Penalize high variance (inconsistent recognition)
            if prob_variance > 0.1:
                gop -= 0.5 * prob_variance

            gop_values.append(gop)

            exp_phoneme = aligned_expected[i] if i < len(aligned_expected) else ''
            ipa_phoneme = detected_ipa[i] if i < len(detected_ipa) else ''

            scores.append(PhonemeScore(
                phoneme=detected_arpabet[i],
                phoneme_ipa=ipa_phoneme,
                expected=exp_phoneme,
                gop_score=round(gop, 3),
                probability=round(avg_prob, 3),
                duration=round(duration_per_phoneme, 3),
                start_time=round(start_time, 3),
                end_time=round(end_time, 3)
            ))

        # Overall GOP is average
        overall_gop = float(np.mean(gop_values)) if gop_values else 0.0

        return scores, overall_gop

    def _get_aligned_expected(
        self,
        expected: List[str],
        detected: List[str]
    ) -> List[str]:
        """
        Get expected phonemes aligned with detected sequence using DTW-like alignment.
        """
        if not expected or not detected:
            return [''] * len(detected)

        try:
            # Use dynamic programming for alignment
            m, n = len(expected), len(detected)

            # Cost matrix
            dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
            dp[0][0] = 0

            # Backtrack matrix
            bt = [[None] * (n + 1) for _ in range(m + 1)]

            for i in range(m + 1):
                for j in range(n + 1):
                    if i == 0 and j == 0:
                        continue

                    candidates = []

                    # Match/substitute
                    if i > 0 and j > 0:
                        cost = 0 if expected[i-1] == detected[j-1] else 1
                        candidates.append((dp[i-1][j-1] + cost, 'match'))

                    # Delete from expected (gap in detected)
                    if i > 0:
                        candidates.append((dp[i-1][j] + 1, 'del'))

                    # Insert into detected (gap in expected)
                    if j > 0:
                        candidates.append((dp[i][j-1] + 1, 'ins'))

                    if candidates:
                        best_cost, best_op = min(candidates, key=lambda x: x[0])
                        dp[i][j] = best_cost
                        bt[i][j] = best_op

            # Backtrack to get alignment
            aligned = []
            i, j = m, n

            while j > 0:
                if i == 0:
                    aligned.append('')
                    j -= 1
                elif bt[i][j] == 'match':
                    aligned.append(expected[i-1])
                    i -= 1
                    j -= 1
                elif bt[i][j] == 'del':
                    i -= 1
                else:  # ins
                    aligned.append('')
                    j -= 1

            aligned.reverse()
            return aligned

        except Exception as e:
            logger.warning(f"Alignment failed, using simple alignment: {e}")
            # Simple fallback: pad shorter sequence
            result = []
            for i in range(len(detected)):
                if i < len(expected):
                    result.append(expected[i])
                else:
                    result.append('')
            return result

    def _align_and_calculate_per(
        self,
        expected: List[str],
        detected: List[str]
    ) -> Tuple[List[Tuple[str, str]], float]:
        """
        Align phoneme sequences and calculate PER using edit distance.

        Both sequences should be in ARPAbet format.
        """
        if not expected:
            return [(('', d) for d in detected)], (1.0 if detected else 0.0)

        try:
            import editdistance
            distance = editdistance.eval(expected, detected)
            per = distance / len(expected)
        except ImportError:
            # Fallback: simple edit distance
            distance = self._simple_edit_distance(expected, detected)
            per = distance / len(expected)

        # Create alignment pairs using DP alignment
        alignment = self._create_alignment_pairs(expected, detected)

        return alignment, min(1.0, per)

    def _simple_edit_distance(self, s1: List[str], s2: List[str]) -> int:
        """Simple Levenshtein distance implementation."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n]

    def _create_alignment_pairs(
        self,
        expected: List[str],
        detected: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Create (expected, detected) pairs using DP alignment.
        """
        m, n = len(expected), len(detected)

        # DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if expected[i-1] == detected[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        # Backtrack
        alignment = []
        i, j = m, n

        while i > 0 or j > 0:
            if i > 0 and j > 0 and (expected[i-1] == detected[j-1] or
                                     dp[i][j] == dp[i-1][j-1] + 1):
                alignment.append((expected[i-1], detected[j-1]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                alignment.append((expected[i-1], '-'))  # Deletion
                i -= 1
            else:
                alignment.append(('-', detected[j-1]))  # Insertion
                j -= 1

        alignment.reverse()
        return alignment


def format_ml_phoneme_result_for_api(result: MLPhonemeResult) -> Dict:
    """Format ML phoneme result for API response."""
    return {
        'detected_phonemes': result.detected_phonemes,
        'detected_phonemes_ipa': result.detected_phonemes_ipa,
        'expected_phonemes': result.expected_phonemes,
        'per_ml': round(result.per_ml, 3),
        'overall_gop': round(result.overall_gop, 3),
        'model_confidence': round(result.model_confidence, 3),
        'phoneme_scores': [
            {
                'phoneme': s.phoneme,
                'phoneme_ipa': s.phoneme_ipa,
                'expected': s.expected,
                'gop_score': s.gop_score,
                'probability': s.probability,
                'duration': s.duration,
                'start_time': s.start_time,
                'end_time': s.end_time
            }
            for s in result.phoneme_scores
        ],
        'alignment': result.alignment,
        'analysis_method': 'ml'
    }


# Global instance
_phoneme_analyzer = None


def get_phoneme_analyzer() -> MLPhonemeAnalyzer:
    """Get the global phoneme analyzer instance."""
    global _phoneme_analyzer

    if _phoneme_analyzer is None:
        _phoneme_analyzer = MLPhonemeAnalyzer()

    return _phoneme_analyzer


def analyze_phonemes_ml(
    audio_path: str,
    expected_text: str
) -> Optional[MLPhonemeResult]:
    """Convenience function for ML phoneme analysis."""
    analyzer = get_phoneme_analyzer()
    return analyzer.analyze_phonemes(audio_path, expected_text)
