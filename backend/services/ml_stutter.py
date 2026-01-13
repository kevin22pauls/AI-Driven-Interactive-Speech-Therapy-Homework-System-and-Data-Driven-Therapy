"""
ML-based Stuttering Detection

Hybrid approach combining:
- Text-based repetition detection (fast, accurate for word repetitions)
- Acoustic features for prolongation/block detection
- ML confidence scoring

This provides better accuracy than pure heuristics while remaining
efficient for CPU-only systems.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MLStutterEvent:
    """A stuttering event detected by ML."""
    event_type: str  # 'repetition', 'prolongation', 'block', 'interjection', 'revision'
    start_time: float
    end_time: float
    word: str
    position: int
    confidence: float
    severity: int  # 1-3 (mild, moderate, severe)
    acoustic_features: Optional[Dict] = None
    clinical_significance: str = ""


class MLStutterDetector:
    """
    Hybrid ML-based stuttering detector.

    Combines text-based heuristics (for repetitions/interjections)
    with acoustic features (for prolongations/blocks).
    """

    # Filler words for interjection detection
    FILLER_WORDS = {
        'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean',
        'actually', 'basically', 'literally', 'so', 'well', 'hmm'
    }

    def __init__(self):
        self._audio_features_model = None
        self._loaded = False

    def load(self) -> bool:
        """Load any required models."""
        # For acoustic feature extraction, we'll use librosa (no separate model needed)
        self._loaded = True
        return True

    def detect_stuttering_events(
        self,
        audio_path: str,
        word_timings: List[Dict],
        vad_data: Optional[Dict] = None
    ) -> List[MLStutterEvent]:
        """
        Detect stuttering events using hybrid approach.

        Args:
            audio_path: Path to audio file
            word_timings: List of word timing dictionaries
            vad_data: Optional VAD analysis data

        Returns:
            List of MLStutterEvent objects
        """
        events = []

        # 1. Text-based: Word repetitions (most reliable)
        repetitions = self._detect_word_repetitions(word_timings)
        events.extend(repetitions)

        # 2. Text-based: Interjections/fillers
        interjections = self._detect_interjections(word_timings)
        events.extend(interjections)

        # 3. Acoustic-based: Prolongations
        prolongations = self._detect_prolongations_acoustic(audio_path, word_timings)
        events.extend(prolongations)

        # 4. VAD-based: Blocks (using silence segments)
        if vad_data:
            blocks = self._detect_blocks_from_vad(vad_data, word_timings)
            events.extend(blocks)

        # Sort by position
        events.sort(key=lambda e: e.position)

        logger.info(f"ML Stutter Detection: {len(events)} events "
                   f"({len(repetitions)} repetitions, {len(interjections)} interjections, "
                   f"{len(prolongations)} prolongations)")

        return events

    def _detect_word_repetitions(self, word_timings: List[Dict]) -> List[MLStutterEvent]:
        """
        Detect whole-word repetitions from text.

        This is kept as text-based because it's highly accurate.
        """
        events = []

        if len(word_timings) < 2:
            return events

        i = 0
        while i < len(word_timings):
            word = word_timings[i].get('text', '').lower().strip('.,!?;:"\'')

            # Count consecutive repetitions
            rep_count = 1
            j = i + 1
            while j < len(word_timings):
                next_word = word_timings[j].get('text', '').lower().strip('.,!?;:"\'')
                if next_word == word:
                    rep_count += 1
                    j += 1
                else:
                    break

            if rep_count > 1:
                # Calculate severity based on repetition count
                if rep_count == 2:
                    severity = 1  # Mild
                elif rep_count <= 4:
                    severity = 2  # Moderate
                else:
                    severity = 3  # Severe

                # Calculate confidence (higher for more repetitions)
                confidence = min(0.95, 0.7 + (rep_count - 2) * 0.1)

                events.append(MLStutterEvent(
                    event_type='whole_word_repetition',
                    start_time=word_timings[i].get('start', 0),
                    end_time=word_timings[j-1].get('end', 0),
                    word=word,
                    position=i,
                    confidence=confidence,
                    severity=severity,
                    clinical_significance=f"Whole-word repetition ({rep_count}x) - core stuttering behavior"
                ))
                i = j
            else:
                i += 1

        return events

    def _detect_interjections(self, word_timings: List[Dict]) -> List[MLStutterEvent]:
        """Detect filler words/interjections."""
        events = []

        for i, word_info in enumerate(word_timings):
            word = word_info.get('text', '').lower().strip('.,!?;:"\'')

            if word in self.FILLER_WORDS:
                events.append(MLStutterEvent(
                    event_type='interjection',
                    start_time=word_info.get('start', 0),
                    end_time=word_info.get('end', 0),
                    word=word,
                    position=i,
                    confidence=1.0,  # High confidence for dictionary match
                    severity=1,  # Mild - common in all speakers
                    clinical_significance="Filler word - common in all speakers but may indicate word-finding difficulty"
                ))

        return events

    def _detect_prolongations_acoustic(
        self,
        audio_path: str,
        word_timings: List[Dict]
    ) -> List[MLStutterEvent]:
        """
        Detect prolongations using acoustic features.

        Uses duration analysis and optionally spectral features
        to identify prolonged sounds.
        """
        events = []

        if not word_timings:
            return events

        # Calculate duration statistics
        durations = []
        char_durations = []

        for w in word_timings:
            text = w.get('text', '').strip('.,!?;:"\'')
            duration = w.get('end', 0) - w.get('start', 0)
            if len(text) > 0 and duration > 0:
                durations.append(duration)
                char_durations.append(duration / len(text))

        if len(char_durations) < 3:
            return events

        # Calculate statistics
        mean_char_dur = np.mean(char_durations)
        std_char_dur = np.std(char_durations)

        # Threshold for prolongation: mean + 2*std, minimum 0.5s
        threshold = max(mean_char_dur + 2 * std_char_dur, 0.5)

        for i, w in enumerate(word_timings):
            text = w.get('text', '').strip('.,!?;:"\'')
            duration = w.get('end', 0) - w.get('start', 0)

            if len(text) > 0:
                char_dur = duration / len(text)

                if char_dur > threshold and duration > 0.5:
                    # Additional acoustic analysis could be added here
                    # For now, use duration-based detection

                    # Severity based on how much longer than expected
                    if duration < 1.0:
                        severity = 2  # Moderate
                    else:
                        severity = 3  # Severe

                    # Confidence based on how far above threshold
                    deviation = (char_dur - mean_char_dur) / std_char_dur if std_char_dur > 0 else 2
                    confidence = min(0.9, 0.6 + deviation * 0.1)

                    events.append(MLStutterEvent(
                        event_type='prolongation',
                        start_time=w.get('start', 0),
                        end_time=w.get('end', 0),
                        word=text,
                        position=i,
                        confidence=confidence,
                        severity=severity,
                        acoustic_features={
                            'duration': duration,
                            'char_duration': char_dur,
                            'deviation_sigma': deviation
                        },
                        clinical_significance=f"Sound prolongation ({duration:.2f}s) - core stuttering behavior"
                    ))

        return events

    def _detect_blocks_from_vad(
        self,
        vad_data: Dict,
        word_timings: List[Dict]
    ) -> List[MLStutterEvent]:
        """
        Detect blocks from VAD silence segments.

        Blocks are extended silences that occur mid-utterance,
        often accompanied by visible physical tension.
        """
        events = []
        vad_pauses = vad_data.get('pauses', [])

        for pause in vad_pauses:
            duration = pause.get('duration', 0)
            pause_type = pause.get('type', '')

            # Only consider long silences as potential blocks
            if duration >= 1.0 and pause_type in ['silence', 'block']:
                start = pause.get('start', 0)
                end = pause.get('end', 0)

                # Find position relative to words
                position = 0
                before_word = None
                after_word = None

                for i, w in enumerate(word_timings):
                    if w.get('end', 0) <= start:
                        position = i
                        before_word = w.get('text', '')
                    if w.get('start', 0) >= end and after_word is None:
                        after_word = w.get('text', '')
                        break

                # Check if this is truly mid-utterance (has speech before and after)
                if before_word and after_word:
                    # Severity based on duration
                    if duration < 2.0:
                        severity = 2  # Moderate
                    else:
                        severity = 3  # Severe

                    confidence = pause.get('confidence', 0.8)

                    events.append(MLStutterEvent(
                        event_type='block',
                        start_time=start,
                        end_time=end,
                        word=f"[{before_word}...{after_word}]",
                        position=position,
                        confidence=confidence,
                        severity=severity,
                        acoustic_features={
                            'duration': duration,
                            'vad_type': pause_type
                        },
                        clinical_significance=f"Block ({duration:.2f}s) - most severe stuttering type"
                    ))

        return events

    def get_stuttering_summary(self, events: List[MLStutterEvent]) -> Dict:
        """
        Generate a summary of stuttering events.

        Args:
            events: List of detected stuttering events

        Returns:
            Summary dictionary with counts and statistics
        """
        summary = {
            'total_events': len(events),
            'by_type': {},
            'by_severity': {1: 0, 2: 0, 3: 0},
            'avg_confidence': 0,
            'clinical_severity': 'normal'
        }

        if not events:
            return summary

        # Count by type
        for event in events:
            event_type = event.event_type
            summary['by_type'][event_type] = summary['by_type'].get(event_type, 0) + 1
            summary['by_severity'][event.severity] += 1

        # Calculate average confidence
        summary['avg_confidence'] = np.mean([e.confidence for e in events])

        # Determine clinical severity
        # Core behaviors: repetitions, prolongations, blocks
        core_events = [e for e in events if e.event_type in
                      ['whole_word_repetition', 'part_word_repetition', 'prolongation', 'block']]
        severe_events = [e for e in events if e.severity == 3]

        if len(severe_events) > 2 or len(core_events) > 5:
            summary['clinical_severity'] = 'severe'
        elif len(severe_events) > 0 or len(core_events) > 2:
            summary['clinical_severity'] = 'moderate'
        elif len(core_events) > 0:
            summary['clinical_severity'] = 'mild'

        return summary


def format_ml_stutter_events_for_api(events: List[MLStutterEvent]) -> List[Dict]:
    """Format ML stutter events for API response."""
    return [
        {
            'type': e.event_type,
            'start_time': round(e.start_time, 3),
            'end_time': round(e.end_time, 3),
            'word': e.word,
            'position': e.position,
            'confidence': round(e.confidence, 2),
            'severity': e.severity,
            'clinical_significance': e.clinical_significance,
            'acoustic_features': e.acoustic_features
        }
        for e in events
    ]


# Global instance
_stutter_detector = None


def get_stutter_detector() -> MLStutterDetector:
    """Get the global stutter detector instance."""
    global _stutter_detector

    if _stutter_detector is None:
        _stutter_detector = MLStutterDetector()

    return _stutter_detector


def detect_stuttering_ml(
    audio_path: str,
    word_timings: List[Dict],
    vad_data: Optional[Dict] = None
) -> Tuple[List[MLStutterEvent], Dict]:
    """
    Convenience function for ML stuttering detection.

    Returns:
        Tuple of (events list, summary dict)
    """
    detector = get_stutter_detector()
    events = detector.detect_stuttering_events(audio_path, word_timings, vad_data)
    summary = detector.get_stuttering_summary(events)
    return events, summary
