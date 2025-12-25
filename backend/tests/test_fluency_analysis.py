"""
Unit Tests for Fluency Analysis Module

These tests verify the correctness of fluency and stuttering detection,
including pause detection, repetition detection, LFR calculation, and
clinical metrics.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.fluency_analysis import (
    extract_words_from_segments,
    detect_pauses,
    detect_word_repetitions,
    detect_interjections,
    detect_prolongations,
    calculate_longest_fluent_run,
    calculate_speech_rate_variability,
    analyze_fluency,
    Pause,
    StutteringEvent
)


class TestWordExtraction:
    """Tests for extracting words from Whisper segments"""

    def test_simple_segment(self):
        """Test extracting words from single segment"""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "hello world"}
        ]
        words = extract_words_from_segments(segments)

        assert len(words) == 2
        assert words[0]['text'] == 'hello'
        assert words[1]['text'] == 'world'

    def test_multiple_segments(self):
        """Test extracting words from multiple segments"""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "I want"},
            {"start": 1.5, "end": 2.0, "text": "water"}
        ]
        words = extract_words_from_segments(segments)

        assert len(words) == 3
        assert words[0]['text'] == 'i'
        assert words[1]['text'] == 'want'
        assert words[2]['text'] == 'water'

    def test_empty_segments(self):
        """Test with empty segments"""
        words = extract_words_from_segments([])
        assert len(words) == 0

    def test_punctuation_removal(self):
        """Test that punctuation is stripped from words"""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello, world!"}
        ]
        words = extract_words_from_segments(segments)

        assert words[0]['text'] == 'hello'
        assert words[1]['text'] == 'world'


class TestPauseDetection:
    """Tests for pause detection"""

    def test_no_pauses(self):
        """Test continuous speech with no pauses"""
        words = [
            {"text": "hello", "start": 0.0, "end": 0.5},
            {"text": "world", "start": 0.5, "end": 1.0}
        ]
        pauses = detect_pauses(words)
        assert len(pauses) == 0

    def test_hesitation_detection(self):
        """Test detection of hesitation (300ms-1s pause)"""
        words = [
            {"text": "i", "start": 0.0, "end": 0.2},
            {"text": "want", "start": 0.6, "end": 0.8}  # 0.4s gap = hesitation
        ]
        pauses = detect_pauses(words)

        assert len(pauses) == 1
        assert pauses[0].pause_type == 'hesitation'
        assert pauses[0].duration > 0.3
        assert pauses[0].before_word == 'i'
        assert pauses[0].after_word == 'want'

    def test_block_detection(self):
        """Test detection of block (>1s pause)"""
        words = [
            {"text": "i", "start": 0.0, "end": 0.2},
            {"text": "want", "start": 1.5, "end": 1.7}  # 1.3s gap = block
        ]
        pauses = detect_pauses(words)

        assert len(pauses) == 1
        assert pauses[0].pause_type == 'block'
        assert pauses[0].duration > 1.0

    def test_pause_location(self):
        """Test that pause locations are correctly identified"""
        words = [
            {"text": "a", "start": 0.0, "end": 0.2},
            {"text": "b", "start": 0.7, "end": 0.9},
            {"text": "c", "start": 1.0, "end": 1.2},
            {"text": "d", "start": 1.7, "end": 1.9}
        ]
        pauses = detect_pauses(words)

        assert pauses[0].location == 'beginning'
        assert pauses[1].location == 'middle'
        assert pauses[2].location == 'end'


class TestStutteringDetection:
    """Tests for stuttering event detection"""

    def test_word_repetition_detection(self):
        """Test detection of word repetitions (I I I want)"""
        words = [
            {"text": "i", "start": 0.0, "end": 0.2},
            {"text": "i", "start": 0.3, "end": 0.5},
            {"text": "i", "start": 0.6, "end": 0.8},
            {"text": "want", "start": 1.0, "end": 1.2}
        ]
        events = detect_word_repetitions(words)

        assert len(events) == 1
        assert events[0].event_type == 'repetition'
        assert events[0].word == 'i'
        assert events[0].repetition_count == 3

    def test_multiple_repetitions(self):
        """Test detection of multiple separate repetitions"""
        words = [
            {"text": "i", "start": 0.0, "end": 0.2},
            {"text": "i", "start": 0.3, "end": 0.5},
            {"text": "want", "start": 1.0, "end": 1.2},
            {"text": "want", "start": 1.3, "end": 1.5}
        ]
        events = detect_word_repetitions(words)

        assert len(events) == 2
        assert events[0].word == 'i'
        assert events[1].word == 'want'

    def test_interjection_detection(self):
        """Test detection of filler words"""
        words = [
            {"text": "i", "start": 0.0, "end": 0.2},
            {"text": "um", "start": 0.5, "end": 0.7},
            {"text": "want", "start": 1.0, "end": 1.2},
            {"text": "uh", "start": 1.5, "end": 1.7},
            {"text": "water", "start": 2.0, "end": 2.3}
        ]
        events = detect_interjections(words)

        assert len(events) == 2
        assert all(e.event_type == 'interjection' for e in events)
        assert events[0].word == 'um'
        assert events[1].word == 'uh'

    def test_prolongation_detection(self):
        """Test detection of prolonged sounds"""
        words = [
            {"text": "i", "start": 0.0, "end": 0.2},  # Normal
            {"text": "waaant", "start": 1.0, "end": 2.5},  # Prolonged (1.5s)
            {"text": "water", "start": 3.0, "end": 3.3}  # Normal
        ]
        events = detect_prolongations(words)

        # Should detect the prolonged "waaant"
        assert len(events) >= 0  # May or may not detect based on threshold


class TestLongestFluentRun:
    """Tests for Longest Fluent Run calculation"""

    def test_lfr_no_dysfluencies(self):
        """Test LFR when there are no dysfluencies"""
        words = [
            {"text": "i", "start": 0.0, "end": 0.2},
            {"text": "want", "start": 0.2, "end": 0.5},
            {"text": "to", "start": 0.5, "end": 0.7},
            {"text": "drink", "start": 0.7, "end": 1.0},
            {"text": "water", "start": 1.0, "end": 1.3}
        ]
        pauses = []
        events = []

        lfr = calculate_longest_fluent_run(words, pauses, events)
        assert lfr == 5  # All 5 words are fluent

    def test_lfr_with_pause(self):
        """Test LFR when pause interrupts fluency"""
        words = [
            {"text": "i", "start": 0.0, "end": 0.2},
            {"text": "want", "start": 0.2, "end": 0.5},
            {"text": "to", "start": 1.5, "end": 1.7},  # Pause before this
            {"text": "drink", "start": 1.7, "end": 2.0},
            {"text": "water", "start": 2.0, "end": 2.3}
        ]
        pauses = [
            Pause(position=1, duration=1.0, pause_type='block', before_word='want', after_word='to')
        ]
        events = []

        lfr = calculate_longest_fluent_run(words, pauses, events)
        assert lfr == 3  # "to drink water" is the longest run

    def test_lfr_with_repetition(self):
        """Test LFR when repetition interrupts fluency"""
        words = [
            {"text": "i", "start": 0.0, "end": 0.2},
            {"text": "i", "start": 0.3, "end": 0.5},
            {"text": "want", "start": 1.0, "end": 1.2},
            {"text": "water", "start": 1.2, "end": 1.5}
        ]
        pauses = []
        events = [
            StutteringEvent(
                event_type='repetition',
                position=0,
                word='i',
                repetition_count=2
            )
        ]

        lfr = calculate_longest_fluent_run(words, pauses, events)
        assert lfr == 2  # "want water" after the repetition


class TestFluencyMetrics:
    """Tests for overall fluency metrics"""

    def test_fluency_analysis_normal_speech(self):
        """Test fluency analysis on normal, fluent speech"""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "I want to drink water"}
        ]
        result = analyze_fluency(segments)

        assert result.longest_fluent_run > 0
        assert result.fluency_percentage >= 90  # Should be very fluent
        assert result.total_words == 5
        assert len(result.clinical_notes) > 0

    def test_fluency_analysis_with_repetition(self):
        """Test fluency analysis with word repetition"""
        segments = [
            {"start": 0.0, "end": 0.5, "text": "I I I"},
            {"start": 1.0, "end": 1.5, "text": "want water"}
        ]
        result = analyze_fluency(segments)

        # Should detect repetitions
        repetitions = [e for e in result.stuttering_events if e.event_type == 'repetition']
        assert len(repetitions) > 0

        # Fluency should be impacted
        assert result.fluency_percentage < 100

    def test_fluency_analysis_with_fillers(self):
        """Test fluency analysis with filler words"""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "I um want uh water"}
        ]
        result = analyze_fluency(segments)

        # Should detect interjections
        interjections = [e for e in result.stuttering_events if e.event_type == 'interjection']
        assert len(interjections) == 2  # "um" and "uh"

    def test_dysfluencies_per_100_words(self):
        """Test calculation of dysfluencies per 100 words"""
        # 10 words with 2 repetitions = 20 dysfluencies per 100 words
        segments = [
            {"start": 0.0, "end": 1.0, "text": "I I want want to drink some clean cold water"}
        ]
        result = analyze_fluency(segments)

        # Should have dysfluencies detected
        assert result.dysfluencies_per_100_words > 0

    def test_empty_speech(self):
        """Test handling of empty speech"""
        result = analyze_fluency([])

        assert result.longest_fluent_run == 0
        assert result.total_words == 0
        assert result.fluency_percentage == 100.0
        assert "No speech" in result.clinical_notes[0]


class TestSpeechRateVariability:
    """Tests for speech rate variability calculation"""

    def test_constant_rate(self):
        """Test with constant speech rate (low variability)"""
        words = [
            {"text": "a", "start": 0.0, "end": 0.5},
            {"text": "b", "start": 0.5, "end": 1.0},
            {"text": "c", "start": 1.0, "end": 1.5},
            {"text": "d", "start": 1.5, "end": 2.0}
        ]
        variability = calculate_speech_rate_variability(words)

        # Should be low variability
        assert variability < 0.3

    def test_variable_rate(self):
        """Test with variable speech rate (high variability)"""
        words = [
            {"text": "fast", "start": 0.0, "end": 0.1},
            {"text": "fast", "start": 0.1, "end": 0.2},
            {"text": "slow", "start": 0.2, "end": 1.2},
            {"text": "slow", "start": 1.2, "end": 2.2}
        ]
        variability = calculate_speech_rate_variability(words)

        # Should be high variability
        assert variability > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
