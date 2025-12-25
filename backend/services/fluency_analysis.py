"""
Fluency and Stuttering Detection Module

This module provides comprehensive fluency analysis for speech therapy,
including stuttering detection, pause analysis, and fluency metrics.

Clinical Features:
- Longest Fluent Run (LFR) - key progress indicator
- Pause detection and classification (hesitations vs blocks)
- Stuttering event detection (repetitions, prolongations, interjections)
- Dysfluency frequency calculations
- Speech rate variability analysis

Critical for patients with:
- Stuttering disorders
- Cluttering
- Neurogenic fluency disorders
- Aphasia with fluency deficits
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)

# Thresholds for pause classification (in seconds)
HESITATION_THRESHOLD = 0.3  # 300ms - brief pause
BLOCK_THRESHOLD = 1.0  # 1000ms - significant pause/block

# Common filler words indicating dysfluency
FILLER_WORDS = {
    'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean',
    'actually', 'basically', 'literally', 'so', 'well'
}


@dataclass
class Pause:
    """Represents a detected pause in speech."""
    position: int  # Word position after which pause occurs
    duration: float  # Pause duration in seconds
    pause_type: str  # 'hesitation' or 'block'
    before_word: Optional[str] = None  # Word before pause
    after_word: Optional[str] = None  # Word after pause
    location: str = 'middle'  # 'beginning', 'middle', 'end'


@dataclass
class StutteringEvent:
    """Represents a detected stuttering event."""
    event_type: str  # 'repetition', 'prolongation', 'block', 'interjection'
    position: int  # Word position
    word: str  # The affected word
    repeated_word: Optional[str] = None  # For repetitions
    repetition_count: int = 0  # Number of repetitions
    duration: Optional[float] = None  # Duration if applicable
    confidence: float = 1.0


@dataclass
class FluencyAnalysisResult:
    """Complete fluency analysis result."""
    longest_fluent_run: int  # Maximum words without dysfluency
    total_pauses: int
    hesitation_count: int
    block_count: int
    pauses: List[Pause]
    stuttering_events: List[StutteringEvent]
    dysfluencies_per_100_words: float
    dysfluencies_per_minute: float
    fluency_percentage: float  # % of fluent speech
    speech_rate_variability: float  # Coefficient of variation
    total_words: int
    total_duration: float  # Total speech duration in seconds
    clinical_notes: List[str]


def extract_words_from_segments(segments: List[Dict]) -> List[Dict]:
    """
    Extract individual words with timestamps from Whisper segments.

    Whisper returns phrase-level segments. This function splits them into words.

    Args:
        segments: List of segment dictionaries with start, end, text

    Returns:
        List of word dictionaries with start, end, text
    """
    words = []

    for seg in segments:
        text = seg.get('text', '').strip()
        if not text:
            continue

        # Split segment text into words
        segment_words = text.split()
        if not segment_words:
            continue

        # Estimate word-level timestamps
        # (Whisper doesn't provide word-level, so we approximate)
        seg_start = seg['start']
        seg_end = seg['end']
        seg_duration = seg_end - seg_start

        # Distribute time evenly across words (simple approximation)
        time_per_word = seg_duration / len(segment_words) if len(segment_words) > 0 else 0

        for i, word in enumerate(segment_words):
            word_start = seg_start + (i * time_per_word)
            word_end = word_start + time_per_word

            words.append({
                'text': word.lower().strip('.,!?;:"\''),
                'start': word_start,
                'end': word_end,
                'segment_id': len(words)
            })

    return words


def detect_pauses(words: List[Dict]) -> List[Pause]:
    """
    Detect pauses between words.

    Args:
        words: List of word dictionaries with start/end times

    Returns:
        List of Pause objects
    """
    pauses = []

    if len(words) < 2:
        return pauses

    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i + 1]

        # Calculate gap between words
        gap = next_word['start'] - current_word['end']

        if gap > HESITATION_THRESHOLD:
            # Determine pause type
            if gap >= BLOCK_THRESHOLD:
                pause_type = 'block'
            else:
                pause_type = 'hesitation'

            # Determine location
            if i == 0:
                location = 'beginning'
            elif i == len(words) - 2:
                location = 'end'
            else:
                location = 'middle'

            pause = Pause(
                position=i,
                duration=gap,
                pause_type=pause_type,
                before_word=current_word['text'],
                after_word=next_word['text'],
                location=location
            )
            pauses.append(pause)

    return pauses


def detect_word_repetitions(words: List[Dict]) -> List[StutteringEvent]:
    """
    Detect word-level repetitions (e.g., "I I I want").

    Args:
        words: List of word dictionaries

    Returns:
        List of StutteringEvent objects for repetitions
    """
    events = []

    if len(words) < 2:
        return events

    i = 0
    while i < len(words) - 1:
        current_word = words[i]['text']

        # Count consecutive repetitions
        repetition_count = 1
        j = i + 1

        while j < len(words) and words[j]['text'] == current_word:
            repetition_count += 1
            j += 1

        # If word repeated at least once
        if repetition_count > 1:
            event = StutteringEvent(
                event_type='repetition',
                position=i,
                word=current_word,
                repeated_word=current_word,
                repetition_count=repetition_count,
                confidence=0.95
            )
            events.append(event)

            # Skip past all repetitions
            i = j
        else:
            i += 1

    return events


def detect_interjections(words: List[Dict]) -> List[StutteringEvent]:
    """
    Detect filler words and interjections (um, uh, like, etc.).

    Args:
        words: List of word dictionaries

    Returns:
        List of StutteringEvent objects for interjections
    """
    events = []

    for i, word_info in enumerate(words):
        word = word_info['text']

        if word in FILLER_WORDS:
            event = StutteringEvent(
                event_type='interjection',
                position=i,
                word=word,
                confidence=1.0
            )
            events.append(event)

    return events


def detect_prolongations(words: List[Dict]) -> List[StutteringEvent]:
    """
    Detect prolonged sounds based on word duration.

    A word is considered prolonged if its duration is significantly
    longer than expected for its length.

    Args:
        words: List of word dictionaries

    Returns:
        List of StutteringEvent objects for prolongations
    """
    events = []

    if not words:
        return events

    # Calculate average duration per character
    durations_per_char = []
    for word_info in words:
        word = word_info['text']
        duration = word_info['end'] - word_info['start']
        if len(word) > 0:
            durations_per_char.append(duration / len(word))

    if not durations_per_char:
        return events

    avg_duration_per_char = statistics.mean(durations_per_char)
    std_duration_per_char = statistics.stdev(durations_per_char) if len(durations_per_char) > 1 else 0

    # Threshold: 2 standard deviations above mean
    prolongation_threshold = avg_duration_per_char + (2 * std_duration_per_char)

    for i, word_info in enumerate(words):
        word = word_info['text']
        duration = word_info['end'] - word_info['start']

        if len(word) > 0:
            duration_per_char = duration / len(word)

            if duration_per_char > prolongation_threshold and duration > 0.5:  # At least 500ms
                event = StutteringEvent(
                    event_type='prolongation',
                    position=i,
                    word=word,
                    duration=duration,
                    confidence=0.7  # Lower confidence as this is an approximation
                )
                events.append(event)

    return events


def calculate_longest_fluent_run(
    words: List[Dict],
    pauses: List[Pause],
    stuttering_events: List[StutteringEvent]
) -> int:
    """
    Calculate the longest run of fluent speech without dysfluencies.

    Args:
        words: List of word dictionaries
        pauses: Detected pauses
        stuttering_events: Detected stuttering events

    Returns:
        Maximum number of consecutive fluent words
    """
    if not words:
        return 0

    # Create set of positions with dysfluencies
    dysfluent_positions = set()

    # Add pause positions (after the pause)
    for pause in pauses:
        dysfluent_positions.add(pause.position + 1)

    # Add stuttering event positions
    for event in stuttering_events:
        dysfluent_positions.add(event.position)
        # For repetitions, mark all repeated positions
        if event.event_type == 'repetition':
            for j in range(event.position, event.position + event.repetition_count):
                dysfluent_positions.add(j)

    # Calculate longest run
    max_run = 0
    current_run = 0

    for i in range(len(words)):
        if i in dysfluent_positions:
            max_run = max(max_run, current_run)
            current_run = 0
        else:
            current_run += 1

    # Check final run
    max_run = max(max_run, current_run)

    return max_run


def calculate_speech_rate_variability(words: List[Dict]) -> float:
    """
    Calculate speech rate variability (coefficient of variation).

    Measures consistency of speaking pace. Higher values indicate
    more variable rate (common in dysfluent speech).

    Args:
        words: List of word dictionaries

    Returns:
        Coefficient of variation (std / mean)
    """
    if len(words) < 2:
        return 0.0

    # Calculate rate for sliding windows
    window_size = 3  # 3-word windows
    rates = []

    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        duration = window[-1]['end'] - window[0]['start']
        if duration > 0:
            rate = len(window) / duration  # words per second
            rates.append(rate)

    if len(rates) < 2:
        return 0.0

    mean_rate = statistics.mean(rates)
    std_rate = statistics.stdev(rates)

    # Coefficient of variation
    cv = (std_rate / mean_rate) if mean_rate > 0 else 0.0

    return cv


def generate_fluency_clinical_notes(
    result: FluencyAnalysisResult,
    stuttering_events: List[StutteringEvent],
    pauses: List[Pause]
) -> List[str]:
    """
    Generate clinical notes based on fluency analysis.

    Args:
        result: Fluency analysis result
        stuttering_events: Detected stuttering events
        pauses: Detected pauses

    Returns:
        List of clinical note strings
    """
    notes = []

    # Overall fluency assessment
    if result.fluency_percentage >= 95:
        notes.append("Excellent fluency with minimal dysfluencies")
    elif result.fluency_percentage >= 85:
        notes.append("Good fluency with occasional dysfluencies")
    elif result.fluency_percentage >= 70:
        notes.append("Moderate fluency challenges detected")
    else:
        notes.append("Significant fluency difficulties observed")

    # Longest fluent run
    if result.longest_fluent_run >= 10:
        notes.append(f"Strong fluent run of {result.longest_fluent_run} words")
    elif result.longest_fluent_run >= 5:
        notes.append(f"Moderate fluent run of {result.longest_fluent_run} words")
    else:
        notes.append(f"Short fluent runs (max: {result.longest_fluent_run} words) - focus on fluency building")

    # Pause patterns
    if result.block_count > 0:
        notes.append(f"Detected {result.block_count} significant block(s) - consider desensitization therapy")

    if result.hesitation_count > result.total_words * 0.3:
        notes.append("Frequent hesitations - may indicate word-finding difficulties")

    # Stuttering patterns
    repetitions = [e for e in stuttering_events if e.event_type == 'repetition']
    if repetitions:
        avg_reps = statistics.mean([e.repetition_count for e in repetitions])
        notes.append(f"{len(repetitions)} word repetition(s) detected (avg: {avg_reps:.1f} repetitions)")

    prolongations = [e for e in stuttering_events if e.event_type == 'prolongation']
    if prolongations:
        notes.append(f"{len(prolongations)} prolongation(s) detected - practice easy onset techniques")

    interjections = [e for e in stuttering_events if e.event_type == 'interjection']
    if len(interjections) > result.total_words * 0.15:
        notes.append("Frequent filler words - recommend speech monitoring exercises")

    # Speech rate variability
    if result.speech_rate_variability > 0.5:
        notes.append("High speech rate variability - focus on consistent pacing")

    # Pause locations
    beginning_pauses = [p for p in pauses if p.location == 'beginning']
    if len(beginning_pauses) > 0:
        notes.append("Initial blocks detected - practice smooth speech initiation")

    return notes


def analyze_fluency(segments: List[Dict]) -> FluencyAnalysisResult:
    """
    Perform comprehensive fluency analysis on speech segments.

    Args:
        segments: List of Whisper segment dictionaries with start, end, text

    Returns:
        FluencyAnalysisResult with complete fluency metrics
    """
    # Extract words from segments
    words = extract_words_from_segments(segments)

    if not words:
        # Return empty result for no speech
        return FluencyAnalysisResult(
            longest_fluent_run=0,
            total_pauses=0,
            hesitation_count=0,
            block_count=0,
            pauses=[],
            stuttering_events=[],
            dysfluencies_per_100_words=0.0,
            dysfluencies_per_minute=0.0,
            fluency_percentage=100.0,
            speech_rate_variability=0.0,
            total_words=0,
            total_duration=0.0,
            clinical_notes=["No speech detected"]
        )

    # Calculate total duration
    total_duration = words[-1]['end'] - words[0]['start']
    total_words = len(words)

    # Detect pauses
    pauses = detect_pauses(words)
    hesitation_count = sum(1 for p in pauses if p.pause_type == 'hesitation')
    block_count = sum(1 for p in pauses if p.pause_type == 'block')

    # Detect stuttering events
    repetitions = detect_word_repetitions(words)
    interjections = detect_interjections(words)
    prolongations = detect_prolongations(words)

    stuttering_events = repetitions + interjections + prolongations

    # Calculate longest fluent run
    lfr = calculate_longest_fluent_run(words, pauses, stuttering_events)

    # Calculate dysfluency metrics
    total_dysfluencies = len(pauses) + len(stuttering_events)

    if total_words > 0:
        dysfluencies_per_100_words = (total_dysfluencies / total_words) * 100
        fluency_percentage = max(0, 100 - dysfluencies_per_100_words)
    else:
        dysfluencies_per_100_words = 0.0
        fluency_percentage = 100.0

    if total_duration > 0:
        dysfluencies_per_minute = (total_dysfluencies / total_duration) * 60
    else:
        dysfluencies_per_minute = 0.0

    # Calculate speech rate variability
    speech_rate_var = calculate_speech_rate_variability(words)

    # Create preliminary result
    result = FluencyAnalysisResult(
        longest_fluent_run=lfr,
        total_pauses=len(pauses),
        hesitation_count=hesitation_count,
        block_count=block_count,
        pauses=pauses,
        stuttering_events=stuttering_events,
        dysfluencies_per_100_words=dysfluencies_per_100_words,
        dysfluencies_per_minute=dysfluencies_per_minute,
        fluency_percentage=fluency_percentage,
        speech_rate_variability=speech_rate_var,
        total_words=total_words,
        total_duration=total_duration,
        clinical_notes=[]
    )

    # Generate clinical notes
    result.clinical_notes = generate_fluency_clinical_notes(result, stuttering_events, pauses)

    logger.info(f"Fluency analysis - LFR: {lfr}, Pauses: {len(pauses)}, "
               f"Stuttering events: {len(stuttering_events)}, "
               f"Fluency: {fluency_percentage:.1f}%")

    return result


def format_fluency_result_for_api(result: FluencyAnalysisResult) -> Dict:
    """
    Format FluencyAnalysisResult for API response.

    Args:
        result: FluencyAnalysisResult object

    Returns:
        Dictionary suitable for JSON serialization
    """
    return {
        "longest_fluent_run": result.longest_fluent_run,
        "total_pauses": result.total_pauses,
        "hesitation_count": result.hesitation_count,
        "block_count": result.block_count,
        "pauses": [
            {
                "position": p.position,
                "duration": round(p.duration, 3),
                "type": p.pause_type,
                "before_word": p.before_word,
                "after_word": p.after_word,
                "location": p.location
            }
            for p in result.pauses
        ],
        "stuttering_events": [
            {
                "type": e.event_type,
                "position": e.position,
                "word": e.word,
                "repeated_word": e.repeated_word,
                "repetition_count": e.repetition_count,
                "duration": round(e.duration, 3) if e.duration else None,
                "confidence": e.confidence
            }
            for e in result.stuttering_events
        ],
        "dysfluencies_per_100_words": round(result.dysfluencies_per_100_words, 2),
        "dysfluencies_per_minute": round(result.dysfluencies_per_minute, 2),
        "fluency_percentage": round(result.fluency_percentage, 1),
        "speech_rate_variability": round(result.speech_rate_variability, 3),
        "total_words": result.total_words,
        "total_duration": round(result.total_duration, 2),
        "clinical_notes": result.clinical_notes
    }
