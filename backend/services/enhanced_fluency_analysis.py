"""
Enhanced Fluency and Stuttering Detection for Aphasia Speech Therapy

This module provides comprehensive fluency analysis specifically designed
for disordered speech, implementing:
- Articulation Rate and Pause-Adjusted Speaking Rate
- Enhanced pause classification (anomic, apraxic, block)
- Adaptive Longest Fluent Run (LFR) with error tolerance
- Comprehensive stuttering detection with SSI-4 approximation
- Weighted Fluency Score (WFS)
- Multi-scale speech rate variability (local CV, global CV, PVI)

Based on clinical research in fluency disorders and aphasia.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import statistics
import numpy as np
from scipy.stats import linregress

logger = logging.getLogger(__name__)


# Syllable counting (simple approximation)
def count_syllables(word: str) -> int:
    """
    Count syllables in a word using vowel-based heuristic.
    """
    word = word.lower().strip()
    if not word:
        return 0

    vowels = 'aeiouy'
    count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    # Handle silent e
    if word.endswith('e') and count > 1:
        count -= 1

    # Handle -le endings
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1

    return max(1, count)


# Adaptive thresholds by disorder type
DISORDER_THRESHOLDS = {
    'normal': {'pause_thresh': 300, 'rate_thresh': 0.15, 'block_thresh': 500},
    'aphasia': {'pause_thresh': 500, 'rate_thresh': 0.25, 'block_thresh': 1000},
    'apraxia': {'pause_thresh': 400, 'rate_thresh': 0.30, 'block_thresh': 800},
    'stuttering': {'pause_thresh': 350, 'rate_thresh': 0.20, 'block_thresh': 500},
    'dysarthria': {'pause_thresh': 400, 'rate_thresh': 0.35, 'block_thresh': 800}
}

# Normative data for speech rates (syllables per minute)
NORMATIVE_RATES = {
    'normal': {'articulation_rate': (210, 260), 'speaking_rate': (150, 190)},
    'broca_aphasia': {'articulation_rate': (80, 150), 'speaking_rate': (40, 100)},
    'wernicke_aphasia': {'articulation_rate': (180, 240), 'speaking_rate': (140, 200)},
    'apraxia': {'articulation_rate': (60, 120), 'speaking_rate': (30, 80)},
    'dysarthria': {'articulation_rate': (100, 160), 'speaking_rate': (70, 120)}
}

# Filler words indicating dysfluency
FILLER_WORDS = {
    'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean',
    'actually', 'basically', 'literally', 'so', 'well', 'hmm'
}

# Content word POS tags (for anomic pause detection)
CONTENT_WORD_PREFIXES = ['what', 'who', 'where', 'when', 'how', 'why', 'which']

# Dysfluency severity weights for weighted fluency score
DYSFLUENCY_WEIGHTS = {
    'interjection': 0.3,  # Normal in all speakers
    'revision': 0.5,  # Indicates self-monitoring
    'phrase_repetition': 0.5,  # Often normal
    'whole_word_repetition': 0.8,  # More significant
    'part_word_repetition': 1.0,  # Core stuttering
    'prolongation': 1.0,  # Core stuttering
    'block': 1.2,  # Most severe
    'hesitation': 0.4,  # Mild
    'anomic_pause': 0.6,  # Word-finding
    'apraxic_pause': 0.7,  # Motor planning
}


@dataclass
class EnhancedPause:
    """Enhanced pause with clinical classification."""
    position: int
    duration: float
    pause_type: str  # 'normal', 'hesitation', 'anomic', 'apraxic', 'block', 'respiratory'
    before_word: Optional[str]
    after_word: Optional[str]
    location: str  # 'beginning', 'middle', 'end'
    clinical_significance: str
    content_word_follows: bool = False
    at_phrase_boundary: bool = False


@dataclass
class EnhancedStutteringEvent:
    """Enhanced stuttering event with severity scoring."""
    event_type: str  # 'whole_word_repetition', 'part_word_repetition', 'prolongation', 'block', 'interjection', 'revision'
    position: int
    word: str
    severity: int  # 1-3 (mild, moderate, severe)
    duration: Optional[float]
    repetition_count: int = 0
    weight: float = 1.0
    clinical_significance: str = ""


@dataclass
class SpeechRateMetrics:
    """Comprehensive speech rate metrics."""
    articulation_rate: float  # Syllables per minute (excludes pauses)
    speaking_rate: float  # Syllables per minute (includes everything)
    pause_adjusted_rate: float  # Rate excluding pathological pauses
    phonation_time: float  # Total speech time in seconds
    total_duration: float  # Total time including pauses
    phonation_ratio: float  # Proportion of time spent speaking
    pathological_pause_ratio: float  # Ratio of pathological pauses
    normative_comparison: str  # Clinical interpretation
    normative_level: str  # 'below', 'normal', 'above' for quick interpretation


@dataclass
class SpeechRateVariability:
    """Multi-scale speech rate variability metrics."""
    # Local variability (within-phrase)
    local_cv: float  # Coefficient of variation
    local_mean_rate: float
    local_std_rate: float

    # Global variability (across utterances)
    global_cv: float
    global_trend: str  # 'slowing', 'speeding', 'stable'
    global_trend_slope: float

    # Pairwise Variability Index
    raw_pvi: float
    normalized_pvi: float
    rhythm_interpretation: str


@dataclass
class StutteringSeverityIndex:
    """SSI-4 approximation for stuttering severity."""
    frequency_pct: float
    frequency_score: int
    avg_duration: float
    duration_score: int
    total_score: int
    severity: str  # 'very_mild', 'mild', 'moderate', 'severe', 'very_severe'
    percentile_estimate: int


@dataclass
class WeightedFluencyScore:
    """Weighted fluency score with breakdown."""
    standard_fluency_pct: float  # Unweighted
    weighted_fluency_pct: float  # Severity-weighted
    clinical_fluency_pct: float  # Excludes normal dysfluencies
    dysfluency_profile: Dict[str, int]
    normative_level: str  # 'below', 'normal', 'above' for quick interpretation


@dataclass
class EnhancedFluencyResult:
    """Complete enhanced fluency analysis result."""
    # Core metrics
    longest_fluent_run: int
    lfr_with_tolerance: int  # LFR allowing minor dysfluencies
    lfr_ratio: float

    # Pause analysis
    pauses: List[EnhancedPause]
    pause_metrics: Dict[str, float]

    # Stuttering analysis
    stuttering_events: List[EnhancedStutteringEvent]
    ssi_approximation: StutteringSeverityIndex

    # Rate metrics
    speech_rate_metrics: SpeechRateMetrics
    rate_variability: SpeechRateVariability

    # Fluency scores
    fluency_scores: WeightedFluencyScore

    # Summary
    total_words: int
    total_syllables: int
    total_duration: float
    clinical_notes: List[str]


def is_phrase_boundary(prev_word: str, next_word: str) -> bool:
    """
    Heuristically determine if position is a phrase boundary.
    """
    # Check for sentence-ending punctuation
    if prev_word and prev_word[-1] in '.!?;:':
        return True

    # Check for common phrase-final patterns
    phrase_endings = ['and', 'but', 'or', 'so', 'then', 'because']
    if next_word and next_word.lower() in phrase_endings:
        return True

    return False


def is_content_word(word: Optional[str]) -> bool:
    """
    Heuristically determine if word is a content word (noun, verb, adj, adv).
    """
    if not word:
        return False

    word_lower = word.lower().strip('.,!?;:"\'')

    # Function words (not content)
    function_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up',
        'about', 'into', 'over', 'after', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
        'its', 'our', 'their', 'this', 'that', 'these', 'those', 'and', 'or',
        'but', 'if', 'because', 'as', 'until', 'while', 'although', 'though'
    }

    return word_lower not in function_words and len(word_lower) > 2


def classify_pause_enhanced(
    duration: float,
    word_timings: List[Dict],
    position: int,
    thresholds: Dict
) -> Tuple[str, str]:
    """
    Classify pause with clinical categorization.

    Args:
        duration: Pause duration in seconds
        word_timings: List of word timing dictionaries
        position: Position in word sequence
        thresholds: Threshold dictionary for classification

    Returns:
        Tuple of (pause_type, clinical_significance)
    """
    pause_thresh = thresholds.get('pause_thresh', 300) / 1000  # Convert to seconds
    block_thresh = thresholds.get('block_thresh', 1000) / 1000

    prev_word = word_timings[position]['text'] if position < len(word_timings) else None
    next_word = word_timings[position + 1]['text'] if position + 1 < len(word_timings) else None

    at_boundary = is_phrase_boundary(prev_word, next_word)
    content_follows = is_content_word(next_word)

    # Normal juncture pause
    if duration < 0.25 or (duration < pause_thresh and at_boundary):
        return 'normal', 'Normal prosodic pause'

    # Hesitation (brief mid-phrase pause)
    if duration < 0.5 and not at_boundary:
        return 'hesitation', 'Brief hesitation - mild word-finding pause'

    # Anomic pause (before content word)
    if 0.5 <= duration < 2.0 and content_follows:
        return 'anomic', 'Word-finding pause before content word - anomia indicator'

    # Apraxic pause (word-initial struggle)
    # Detected by pause followed by phonetic groping or false starts
    if 0.3 <= duration < 1.0 and position > 0:
        # Check for repeated initial sounds (would need phoneme data for full detection)
        if prev_word and next_word and prev_word[0].lower() == next_word[0].lower():
            return 'apraxic', 'Motor planning pause with possible phonetic groping'

    # Block (extended silence)
    if duration >= block_thresh:
        return 'block', 'Significant block/extended pause - possible severe dysfluency'

    # Default to word-finding for longer pauses
    if duration >= 0.5:
        return 'anomic', 'Extended pause - word-finding difficulty'

    return 'hesitation', 'Mid-phrase hesitation'


def detect_pauses_enhanced(
    words: List[Dict],
    thresholds: Dict
) -> List[EnhancedPause]:
    """
    Detect and classify pauses with clinical categorization.

    Args:
        words: List of word dictionaries with timing
        thresholds: Threshold dictionary

    Returns:
        List of EnhancedPause objects
    """
    pauses = []

    if len(words) < 2:
        return pauses

    for i in range(len(words) - 1):
        current = words[i]
        next_word = words[i + 1]

        gap = next_word['start'] - current['end']

        if gap > 0.1:  # 100ms minimum
            pause_type, clinical_sig = classify_pause_enhanced(gap, words, i, thresholds)

            # Determine location
            relative_pos = i / len(words)
            if relative_pos < 0.1:
                location = 'beginning'
            elif relative_pos > 0.9:
                location = 'end'
            else:
                location = 'middle'

            pauses.append(EnhancedPause(
                position=i,
                duration=gap,
                pause_type=pause_type,
                before_word=current['text'],
                after_word=next_word['text'],
                location=location,
                clinical_significance=clinical_sig,
                content_word_follows=is_content_word(next_word['text']),
                at_phrase_boundary=is_phrase_boundary(current['text'], next_word['text'])
            ))

    return pauses


def detect_stuttering_events_enhanced(words: List[Dict]) -> List[EnhancedStutteringEvent]:
    """
    Comprehensive stuttering event detection.

    Args:
        words: List of word dictionaries

    Returns:
        List of EnhancedStutteringEvent objects
    """
    events = []

    if len(words) < 2:
        return events

    # 1. Whole-word repetitions
    i = 0
    while i < len(words):
        word = words[i]['text'].lower().strip('.,!?;:"\'')

        # Count consecutive repetitions
        rep_count = 1
        j = i + 1
        while j < len(words) and words[j]['text'].lower().strip('.,!?;:"\'') == word:
            rep_count += 1
            j += 1

        if rep_count > 1:
            severity = 1 if rep_count == 2 else (2 if rep_count <= 4 else 3)
            events.append(EnhancedStutteringEvent(
                event_type='whole_word_repetition',
                position=i,
                word=word,
                severity=severity,
                duration=words[j-1]['end'] - words[i]['start'] if j > i else None,
                repetition_count=rep_count,
                weight=DYSFLUENCY_WEIGHTS['whole_word_repetition'],
                clinical_significance=f"Whole-word repetition ({rep_count}x) - core stuttering behavior"
            ))
            i = j
        else:
            i += 1

    # 2. Interjections (filler words)
    for i, word_info in enumerate(words):
        word = word_info['text'].lower().strip('.,!?;:"\'')
        if word in FILLER_WORDS:
            events.append(EnhancedStutteringEvent(
                event_type='interjection',
                position=i,
                word=word,
                severity=1,
                weight=DYSFLUENCY_WEIGHTS['interjection'],
                clinical_significance="Filler word - common in all speakers but may indicate word-finding pause"
            ))

    # 3. Prolongations (based on duration)
    if words:
        # Calculate average duration per character
        char_durations = []
        for w in words:
            word = w['text'].strip('.,!?;:"\'')
            duration = w['end'] - w['start']
            if len(word) > 0:
                char_durations.append(duration / len(word))

        if char_durations:
            mean_dur = statistics.mean(char_durations)
            std_dur = statistics.stdev(char_durations) if len(char_durations) > 1 else 0

            for i, w in enumerate(words):
                word = w['text'].strip('.,!?;:"\'')
                duration = w['end'] - w['start']
                if len(word) > 0:
                    dur_per_char = duration / len(word)
                    if dur_per_char > mean_dur + 2 * std_dur and duration > 0.5:
                        events.append(EnhancedStutteringEvent(
                            event_type='prolongation',
                            position=i,
                            word=word,
                            severity=2 if duration < 1.0 else 3,
                            duration=duration,
                            weight=DYSFLUENCY_WEIGHTS['prolongation'],
                            clinical_significance=f"Sound prolongation ({duration:.2f}s) - core stuttering behavior"
                        ))

    return events


def calculate_speech_rates(
    word_timings: List[Dict],
    pauses: List[EnhancedPause]
) -> SpeechRateMetrics:
    """
    Calculate comprehensive speech rate metrics.

    Args:
        word_timings: List of word dictionaries
        pauses: List of detected pauses

    Returns:
        SpeechRateMetrics object
    """
    if not word_timings:
        return SpeechRateMetrics(
            articulation_rate=0, speaking_rate=0, pause_adjusted_rate=0,
            phonation_time=0, total_duration=0, phonation_ratio=0,
            pathological_pause_ratio=0, normative_comparison="Insufficient data",
            normative_level="unknown"
        )

    total_duration = word_timings[-1]['end'] - word_timings[0]['start']

    # Calculate phonation time (speech without pauses)
    phonation_time = sum(w['end'] - w['start'] for w in word_timings)

    # Count syllables
    total_syllables = sum(count_syllables(w['text']) for w in word_timings)

    # Calculate pathological pause time
    pathological_pauses = [p for p in pauses if p.pause_type in ['anomic', 'apraxic', 'block']]
    pathological_time = sum(p.duration for p in pathological_pauses)

    # Calculate rates (convert to per minute)
    articulation_rate = (total_syllables / phonation_time * 60) if phonation_time > 0 else 0
    speaking_rate = (total_syllables / total_duration * 60) if total_duration > 0 else 0

    # Pause-adjusted rate (excludes pathological pauses)
    adjusted_duration = total_duration - pathological_time
    pause_adjusted_rate = (total_syllables / adjusted_duration * 60) if adjusted_duration > 0 else 0

    phonation_ratio = phonation_time / total_duration if total_duration > 0 else 0
    pathological_ratio = pathological_time / total_duration if total_duration > 0 else 0

    # Normative comparison with level classification
    ar = articulation_rate
    if ar >= 210:
        comparison = "Articulation rate within normal limits"
        level = "normal"
    elif ar >= 150:
        comparison = "Mildly reduced articulation rate"
        level = "below"
    elif ar >= 100:
        comparison = "Moderately reduced articulation rate - consistent with aphasia or apraxia"
        level = "below"
    elif ar >= 60:
        comparison = "Severely reduced articulation rate - consistent with significant motor speech disorder"
        level = "below"
    else:
        comparison = "Profoundly reduced articulation rate"
        level = "below"

    return SpeechRateMetrics(
        articulation_rate=articulation_rate,
        speaking_rate=speaking_rate,
        pause_adjusted_rate=pause_adjusted_rate,
        phonation_time=phonation_time,
        total_duration=total_duration,
        phonation_ratio=phonation_ratio,
        pathological_pause_ratio=pathological_ratio,
        normative_comparison=comparison,
        normative_level=level
    )


def calculate_rate_variability(
    word_timings: List[Dict],
    window_size: int = 5
) -> SpeechRateVariability:
    """
    Calculate multi-scale speech rate variability.

    Args:
        word_timings: List of word dictionaries
        window_size: Window size for local variability

    Returns:
        SpeechRateVariability object
    """
    if len(word_timings) < window_size:
        return SpeechRateVariability(
            local_cv=0, local_mean_rate=0, local_std_rate=0,
            global_cv=0, global_trend='stable', global_trend_slope=0,
            raw_pvi=0, normalized_pvi=0, rhythm_interpretation='insufficient_data'
        )

    # Local variability (within-phrase)
    local_rates = []
    for i in range(len(word_timings) - window_size + 1):
        window = word_timings[i:i + window_size]
        duration = window[-1]['end'] - window[0]['start']
        syllables = sum(count_syllables(w['text']) for w in window)
        if duration > 0:
            local_rates.append(syllables / duration * 60)

    local_mean = statistics.mean(local_rates) if local_rates else 0
    local_std = statistics.stdev(local_rates) if len(local_rates) > 1 else 0
    local_cv = (local_std / local_mean) if local_mean > 0 else 0

    # Global variability and trend
    global_rates = local_rates  # Use same data for simplicity
    global_mean = local_mean
    global_std = local_std
    global_cv = local_cv

    # Trend analysis
    if len(global_rates) > 3:
        x = np.arange(len(global_rates))
        slope, _, r_value, _, _ = linregress(x, global_rates)
        if slope < -0.5:
            trend = 'slowing'
        elif slope > 0.5:
            trend = 'speeding'
        else:
            trend = 'stable'
    else:
        slope = 0
        trend = 'stable'

    # Pairwise Variability Index (PVI)
    word_durations = [w['end'] - w['start'] for w in word_timings]

    if len(word_durations) >= 2:
        # Raw PVI
        differences = [abs(word_durations[i] - word_durations[i+1])
                      for i in range(len(word_durations)-1)]
        raw_pvi = np.mean(differences) if differences else 0

        # Normalized PVI
        normalized_diffs = []
        for i in range(len(word_durations) - 1):
            avg = (word_durations[i] + word_durations[i+1]) / 2
            if avg > 0:
                normalized_diffs.append(abs(word_durations[i] - word_durations[i+1]) / avg)
        normalized_pvi = np.mean(normalized_diffs) * 100 if normalized_diffs else 0
    else:
        raw_pvi = 0
        normalized_pvi = 0

    # Rhythm interpretation
    if normalized_pvi < 40:
        rhythm = 'reduced_variability'  # Common in dysarthria
    elif normalized_pvi < 70:
        rhythm = 'normal'
    else:
        rhythm = 'excessive_variability'  # Common in stuttering/apraxia

    return SpeechRateVariability(
        local_cv=local_cv,
        local_mean_rate=local_mean,
        local_std_rate=local_std,
        global_cv=global_cv,
        global_trend=trend,
        global_trend_slope=slope,
        raw_pvi=raw_pvi,
        normalized_pvi=normalized_pvi,
        rhythm_interpretation=rhythm
    )


def calculate_ssi_approximation(
    events: List[EnhancedStutteringEvent],
    total_syllables: int,
    total_duration: float
) -> StutteringSeverityIndex:
    """
    Calculate SSI-4 approximation for stuttering severity.

    Args:
        events: List of stuttering events
        total_syllables: Total syllables in sample
        total_duration: Total duration in seconds

    Returns:
        StutteringSeverityIndex object
    """
    if total_syllables == 0:
        return StutteringSeverityIndex(
            frequency_pct=0, frequency_score=0, avg_duration=0,
            duration_score=0, total_score=0, severity='normal',
            percentile_estimate=50
        )

    # Core stuttering events only
    core_events = [e for e in events if e.event_type in
                   ['whole_word_repetition', 'part_word_repetition', 'prolongation', 'block']]

    stuttered_syllables = len(set(e.position for e in core_events))
    frequency_pct = (stuttered_syllables / total_syllables) * 100

    # Frequency score (SSI-4 scale)
    freq_thresholds = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 7), (9, 8),
        (12, 9), (14, 10), (17, 11), (21, 12), (25, 14), (28, 15),
        (31, 16), (34, 17)
    ]
    frequency_score = 2  # Default minimum
    for thresh, score in freq_thresholds:
        if frequency_pct >= thresh:
            frequency_score = score
        else:
            break

    # Duration score (average of 3 longest)
    durations = sorted([e.duration for e in core_events if e.duration], reverse=True)[:3]
    avg_duration = np.mean(durations) if durations else 0

    dur_thresholds = [
        (0.5, 2), (0.75, 4), (1.0, 6), (1.5, 8), (3.0, 10),
        (5.0, 12), (10.0, 14), (15.0, 16)
    ]
    duration_score = 2
    for thresh, score in dur_thresholds:
        if avg_duration >= thresh:
            duration_score = score
        else:
            break

    total_score = frequency_score + duration_score

    # Severity classification
    if total_score <= 10:
        severity = 'very_mild'
        percentile = 10
    elif total_score <= 17:
        severity = 'mild'
        percentile = 25
    elif total_score <= 24:
        severity = 'moderate'
        percentile = 50
    elif total_score <= 30:
        severity = 'severe'
        percentile = 75
    else:
        severity = 'very_severe'
        percentile = 95

    return StutteringSeverityIndex(
        frequency_pct=frequency_pct,
        frequency_score=frequency_score,
        avg_duration=avg_duration,
        duration_score=duration_score,
        total_score=total_score,
        severity=severity,
        percentile_estimate=percentile
    )


def calculate_weighted_fluency(
    events: List[EnhancedStutteringEvent],
    pauses: List[EnhancedPause],
    total_syllables: int
) -> WeightedFluencyScore:
    """
    Calculate weighted fluency scores.

    Args:
        events: Stuttering events
        pauses: Pause events
        total_syllables: Total syllables

    Returns:
        WeightedFluencyScore object
    """
    if total_syllables == 0:
        return WeightedFluencyScore(
            standard_fluency_pct=100, weighted_fluency_pct=100,
            clinical_fluency_pct=100, dysfluency_profile={},
            normative_level="unknown"
        )

    # Count dysfluencies by type
    profile = {}
    for e in events:
        profile[e.event_type] = profile.get(e.event_type, 0) + 1

    # Add significant pauses
    for p in pauses:
        if p.pause_type in ['anomic', 'apraxic', 'block']:
            profile[p.pause_type] = profile.get(p.pause_type, 0) + 1

    total_dysfluencies = len(events) + len([p for p in pauses if p.pause_type != 'normal'])

    # Standard fluency (unweighted)
    standard_fluency = 100 * (1 - total_dysfluencies / total_syllables)

    # Weighted fluency
    weighted_count = sum(e.weight for e in events)
    weighted_count += sum(DYSFLUENCY_WEIGHTS.get(p.pause_type, 0.5)
                         for p in pauses if p.pause_type != 'normal')
    weighted_fluency = 100 * (1 - weighted_count / total_syllables)

    # Clinical fluency (excludes normal dysfluencies)
    clinical_events = [e for e in events if e.weight >= 0.8]
    clinical_pauses = [p for p in pauses if p.pause_type in ['block', 'apraxic']]
    clinical_count = len(clinical_events) + len(clinical_pauses)
    clinical_fluency = 100 * (1 - clinical_count / total_syllables)

    # Determine normative level based on weighted fluency percentage
    wf = max(0, weighted_fluency)
    if wf >= 95:
        fluency_level = "normal"
    elif wf >= 85:
        fluency_level = "normal"  # Still within functional range
    elif wf >= 70:
        fluency_level = "below"
    else:
        fluency_level = "below"

    return WeightedFluencyScore(
        standard_fluency_pct=max(0, standard_fluency),
        weighted_fluency_pct=max(0, weighted_fluency),
        clinical_fluency_pct=max(0, clinical_fluency),
        dysfluency_profile=profile,
        normative_level=fluency_level
    )


def calculate_adaptive_lfr(
    words: List[Dict],
    pauses: List[EnhancedPause],
    events: List[EnhancedStutteringEvent],
    error_tolerance: int = 1
) -> Tuple[int, int, float]:
    """
    Calculate adaptive Longest Fluent Run with error tolerance.

    Args:
        words: Word timings
        pauses: Pause events
        events: Stuttering events
        error_tolerance: Number of minor dysfluencies allowed

    Returns:
        Tuple of (strict_lfr, tolerant_lfr, lfr_ratio)
    """
    if not words:
        return 0, 0, 0.0

    # Mark dysfluent positions with severity
    dysfluency_map = {}  # position -> severity (1=minor, 2=major)

    for p in pauses:
        if p.pause_type == 'hesitation':
            dysfluency_map[p.position + 1] = 1
        elif p.pause_type in ['anomic', 'apraxic', 'block']:
            dysfluency_map[p.position + 1] = 2

    for e in events:
        severity = 1 if e.severity == 1 else 2
        dysfluency_map[e.position] = max(dysfluency_map.get(e.position, 0), severity)
        if e.event_type == 'whole_word_repetition':
            for j in range(e.position, e.position + e.repetition_count):
                dysfluency_map[j] = 2

    # Calculate strict LFR (no tolerance)
    strict_max = 0
    current = 0
    for i in range(len(words)):
        if dysfluency_map.get(i, 0) == 0:
            current += 1
        else:
            strict_max = max(strict_max, current)
            current = 0
    strict_max = max(strict_max, current)

    # Calculate tolerant LFR (allows minor dysfluencies)
    tolerant_max = 0
    current = 0
    errors_in_run = 0

    for i in range(len(words)):
        severity = dysfluency_map.get(i, 0)
        if severity == 0:
            current += 1
        elif severity == 1 and errors_in_run < error_tolerance:
            current += 1
            errors_in_run += 1
        else:
            tolerant_max = max(tolerant_max, current)
            current = 0
            errors_in_run = 0

    tolerant_max = max(tolerant_max, current)

    lfr_ratio = tolerant_max / len(words) if words else 0

    return strict_max, tolerant_max, lfr_ratio


def extract_words_with_timing(segments: List[Dict]) -> List[Dict]:
    """
    Extract words with timing from Whisper segments.

    Args:
        segments: Whisper segment dictionaries

    Returns:
        List of word dictionaries with timing
    """
    words = []

    for seg in segments:
        text = seg.get('text', '').strip()
        if not text:
            continue

        segment_words = text.split()
        if not segment_words:
            continue

        seg_start = seg['start']
        seg_end = seg['end']
        seg_duration = seg_end - seg_start

        time_per_word = seg_duration / len(segment_words) if segment_words else 0

        for i, word in enumerate(segment_words):
            word_start = seg_start + (i * time_per_word)
            word_end = word_start + time_per_word

            words.append({
                'text': word.lower().strip('.,!?;:"\''),
                'start': word_start,
                'end': word_end,
                'original': word
            })

    return words


def generate_enhanced_fluency_notes(
    result: 'EnhancedFluencyResult'
) -> List[str]:
    """
    Generate comprehensive clinical notes.

    Args:
        result: EnhancedFluencyResult (before notes are added)

    Returns:
        List of clinical note strings
    """
    notes = []

    # Overall fluency assessment
    wf = result.fluency_scores.weighted_fluency_pct
    if wf >= 95:
        notes.append("Excellent fluency - within normal limits")
    elif wf >= 85:
        notes.append("Good fluency with occasional dysfluencies")
    elif wf >= 70:
        notes.append("Moderate fluency difficulties")
    else:
        notes.append("Significant fluency impairment")

    # LFR insights
    if result.lfr_with_tolerance >= 10:
        notes.append(f"Strong fluent run ({result.lfr_with_tolerance} words) - good fluency capacity")
    elif result.lfr_with_tolerance >= 5:
        notes.append(f"Moderate fluent runs ({result.lfr_with_tolerance} words)")
    else:
        notes.append(f"Short fluent runs (max {result.lfr_with_tolerance} words) - fluency building needed")

    # Speech rate assessment
    sr = result.speech_rate_metrics
    notes.append(sr.normative_comparison)

    if sr.pathological_pause_ratio > 0.3:
        notes.append("High pathological pause ratio - significant word-finding or motor planning difficulty")

    # Stuttering severity
    ssi = result.ssi_approximation
    if ssi.severity != 'normal' and ssi.severity != 'very_mild':
        notes.append(f"Stuttering severity: {ssi.severity.replace('_', ' ')} (SSI score: {ssi.total_score})")

    # Rate variability
    rv = result.rate_variability
    if rv.rhythm_interpretation == 'reduced_variability':
        notes.append("Reduced rhythm variability - may indicate dysarthria")
    elif rv.rhythm_interpretation == 'excessive_variability':
        notes.append("Excessive rhythm variability - may indicate motor planning difficulty")

    if rv.global_trend == 'slowing':
        notes.append("Speech rate decreasing over time - possible fatigue effect")

    # Specific dysfluency patterns
    profile = result.fluency_scores.dysfluency_profile
    if profile.get('anomic', 0) > 2:
        notes.append("Multiple word-finding pauses detected - anomia treatment recommended")
    if profile.get('whole_word_repetition', 0) > 2:
        notes.append("Frequent word repetitions - fluency shaping techniques recommended")
    if profile.get('block', 0) > 0:
        notes.append("Blocks detected - consider desensitization and cancellation techniques")

    return notes


def _merge_vad_with_word_pauses(
    words: List[Dict],
    vad_data: Dict,
    thresholds: Dict
) -> List[EnhancedPause]:
    """
    Merge VAD-detected pauses with word-level analysis for clinical classification.

    This combines the precise pause boundaries from Silero VAD with the
    linguistic context from word timings for clinical interpretation.

    Args:
        words: Word timings list
        vad_data: VAD analysis result from ml_vad
        thresholds: Disorder-specific thresholds

    Returns:
        List of EnhancedPause objects with ML-enhanced detection
    """
    pauses = []
    vad_pauses = vad_data.get('pauses', [])

    for vad_pause in vad_pauses:
        start = vad_pause.get('start', 0)
        end = vad_pause.get('end', 0)
        duration = vad_pause.get('duration', end - start)
        vad_type = vad_pause.get('type', 'speech_gap')
        confidence = vad_pause.get('confidence', 0.5)

        # Skip very short pauses
        if duration < 0.1:
            continue

        # Find surrounding words for context
        before_word = None
        after_word = None
        position = 0

        for i, word in enumerate(words):
            if word['end'] <= start:
                before_word = word['text']
                position = i
            if word['start'] >= end and after_word is None:
                after_word = word['text']
                break

        # Determine location
        relative_pos = position / len(words) if words else 0.5
        if relative_pos < 0.1:
            location = 'beginning'
        elif relative_pos > 0.9:
            location = 'end'
        else:
            location = 'middle'

        # Clinical classification based on VAD type and context
        content_follows = is_content_word(after_word) if after_word else False
        at_boundary = is_phrase_boundary(before_word, after_word) if (before_word and after_word) else False

        # Map VAD type to clinical classification
        pause_thresh = thresholds.get('pause_thresh', 300) / 1000
        block_thresh = thresholds.get('block_thresh', 1000) / 1000

        if vad_type == 'silence' and duration >= block_thresh:
            pause_type = 'block'
            clinical_sig = f'Significant block detected by VAD ({duration:.2f}s) - severe dysfluency'
        elif vad_type == 'hesitation' or (duration >= 0.3 and duration < 0.5):
            pause_type = 'hesitation'
            clinical_sig = f'Brief hesitation detected ({duration:.2f}s)'
        elif content_follows and duration >= 0.5:
            pause_type = 'anomic'
            clinical_sig = f'Word-finding pause before content word ({duration:.2f}s) - anomia indicator'
        elif duration < 0.25 or (duration < pause_thresh and at_boundary):
            pause_type = 'normal'
            clinical_sig = 'Normal prosodic pause'
        else:
            pause_type = 'hesitation'
            clinical_sig = f'Mid-phrase hesitation ({duration:.2f}s)'

        pauses.append(EnhancedPause(
            position=position,
            duration=duration,
            pause_type=pause_type,
            before_word=before_word,
            after_word=after_word,
            location=location,
            clinical_significance=clinical_sig + f' (VAD confidence: {confidence:.2f})',
            content_word_follows=content_follows,
            at_phrase_boundary=at_boundary
        ))

    return pauses


def analyze_fluency_enhanced(
    segments: List[Dict],
    disorder_type: str = 'aphasia',
    audio_path: Optional[str] = None,
    vad_data: Optional[Dict] = None
) -> EnhancedFluencyResult:
    """
    Perform comprehensive enhanced fluency analysis.

    This is the main entry point for enhanced fluency analysis.

    Args:
        segments: Whisper segment dictionaries
        disorder_type: Type of disorder for threshold adjustment
        audio_path: Path to audio file (for ML-based analysis)
        vad_data: Pre-computed VAD analysis from Silero VAD (optional)

    Returns:
        EnhancedFluencyResult with complete analysis
    """
    # Extract words
    words = extract_words_with_timing(segments)

    if not words:
        # Return empty result with properly initialized pause_metrics
        empty_pause_metrics = {
            'total_pauses': 0,
            'hesitation_count': 0,
            'anomic_count': 0,
            'apraxic_count': 0,
            'block_count': 0,
            'mean_pause_duration': 0
        }
        return EnhancedFluencyResult(
            longest_fluent_run=0,
            lfr_with_tolerance=0,
            lfr_ratio=0,
            pauses=[],
            pause_metrics=empty_pause_metrics,
            stuttering_events=[],
            ssi_approximation=StutteringSeverityIndex(0, 0, 0, 0, 0, 'normal', 50),
            speech_rate_metrics=SpeechRateMetrics(0, 0, 0, 0, 0, 0, 0, "No speech", "unknown"),
            rate_variability=SpeechRateVariability(0, 0, 0, 0, 'stable', 0, 0, 0, 'insufficient_data'),
            fluency_scores=WeightedFluencyScore(100, 100, 100, {}, "unknown"),
            total_words=0,
            total_syllables=0,
            total_duration=0,
            clinical_notes=["No speech detected"]
        )

    # Get thresholds
    thresholds = DISORDER_THRESHOLDS.get(disorder_type, DISORDER_THRESHOLDS['aphasia'])

    total_duration = words[-1]['end'] - words[0]['start']
    total_syllables = sum(count_syllables(w['text']) for w in words)

    # Detect pauses - use VAD data if available for more accurate detection
    if vad_data and vad_data.get('pauses'):
        pauses = _merge_vad_with_word_pauses(words, vad_data, thresholds)
        logger.info(f"Using ML VAD-enhanced pause detection: {len(pauses)} pauses")
    else:
        pauses = detect_pauses_enhanced(words, thresholds)
        logger.debug(f"Using threshold-based pause detection: {len(pauses)} pauses")

    # Detect stuttering events
    events = detect_stuttering_events_enhanced(words)

    # Calculate metrics
    speech_rates = calculate_speech_rates(words, pauses)
    rate_var = calculate_rate_variability(words)
    ssi = calculate_ssi_approximation(events, total_syllables, total_duration)
    fluency_scores = calculate_weighted_fluency(events, pauses, total_syllables)
    strict_lfr, tolerant_lfr, lfr_ratio = calculate_adaptive_lfr(words, pauses, events)

    # Pause metrics summary
    pause_metrics = {
        'total_pauses': len(pauses),
        'hesitation_count': len([p for p in pauses if p.pause_type == 'hesitation']),
        'anomic_count': len([p for p in pauses if p.pause_type == 'anomic']),
        'apraxic_count': len([p for p in pauses if p.pause_type == 'apraxic']),
        'block_count': len([p for p in pauses if p.pause_type == 'block']),
        'mean_pause_duration': statistics.mean([p.duration for p in pauses]) if pauses else 0
    }

    # Create result (without notes initially)
    result = EnhancedFluencyResult(
        longest_fluent_run=strict_lfr,
        lfr_with_tolerance=tolerant_lfr,
        lfr_ratio=lfr_ratio,
        pauses=pauses,
        pause_metrics=pause_metrics,
        stuttering_events=events,
        ssi_approximation=ssi,
        speech_rate_metrics=speech_rates,
        rate_variability=rate_var,
        fluency_scores=fluency_scores,
        total_words=len(words),
        total_syllables=total_syllables,
        total_duration=total_duration,
        clinical_notes=[]
    )

    # Generate clinical notes
    result.clinical_notes = generate_enhanced_fluency_notes(result)

    return result


def format_enhanced_fluency_result_for_api(result: EnhancedFluencyResult) -> Dict:
    """
    Format EnhancedFluencyResult for API response.

    Args:
        result: EnhancedFluencyResult object

    Returns:
        Dictionary suitable for JSON serialization
    """
    return {
        # LFR metrics
        'longest_fluent_run': result.longest_fluent_run,
        'lfr_with_tolerance': result.lfr_with_tolerance,
        'lfr_ratio': round(result.lfr_ratio, 3),

        # Pause analysis
        'pause_metrics': {
            'total_pauses': result.pause_metrics.get('total_pauses', 0),
            'hesitation_count': result.pause_metrics.get('hesitation_count', 0),
            'anomic_count': result.pause_metrics.get('anomic_count', 0),
            'apraxic_count': result.pause_metrics.get('apraxic_count', 0),
            'block_count': result.pause_metrics.get('block_count', 0),
            'mean_pause_duration': round(result.pause_metrics.get('mean_pause_duration', 0), 3)
        },
        'pauses': [
            {
                'position': p.position,
                'duration': round(p.duration, 3),
                'type': p.pause_type,
                'before_word': p.before_word,
                'after_word': p.after_word,
                'location': p.location,
                'clinical_significance': p.clinical_significance,
                'content_word_follows': p.content_word_follows
            }
            for p in result.pauses
        ],

        # Stuttering analysis
        'stuttering_events': [
            {
                'type': e.event_type,
                'position': e.position,
                'word': e.word,
                'severity': e.severity,
                'duration': round(e.duration, 3) if e.duration else None,
                'repetition_count': e.repetition_count,
                'weight': round(e.weight, 2),
                'clinical_significance': e.clinical_significance
            }
            for e in result.stuttering_events
        ],
        'ssi_approximation': {
            'frequency_pct': round(result.ssi_approximation.frequency_pct, 2),
            'frequency_score': result.ssi_approximation.frequency_score,
            'avg_duration': round(result.ssi_approximation.avg_duration, 3),
            'duration_score': result.ssi_approximation.duration_score,
            'total_score': result.ssi_approximation.total_score,
            'severity': result.ssi_approximation.severity,
            'percentile_estimate': result.ssi_approximation.percentile_estimate
        },

        # Speech rate metrics
        'speech_rate_metrics': {
            'articulation_rate': round(result.speech_rate_metrics.articulation_rate, 1),
            'speaking_rate': round(result.speech_rate_metrics.speaking_rate, 1),
            'pause_adjusted_rate': round(result.speech_rate_metrics.pause_adjusted_rate, 1),
            'phonation_time': round(result.speech_rate_metrics.phonation_time, 2),
            'phonation_ratio': round(result.speech_rate_metrics.phonation_ratio, 3),
            'pathological_pause_ratio': round(result.speech_rate_metrics.pathological_pause_ratio, 3),
            'normative_comparison': result.speech_rate_metrics.normative_comparison,
            'normative_level': result.speech_rate_metrics.normative_level
        },

        # Rate variability
        'rate_variability': {
            'local_cv': round(result.rate_variability.local_cv, 3),
            'local_mean_rate': round(result.rate_variability.local_mean_rate, 1),
            'global_cv': round(result.rate_variability.global_cv, 3),
            'global_trend': result.rate_variability.global_trend,
            'global_trend_slope': round(result.rate_variability.global_trend_slope, 3),
            'normalized_pvi': round(result.rate_variability.normalized_pvi, 1),
            'rhythm_interpretation': result.rate_variability.rhythm_interpretation
        },

        # Fluency scores
        'fluency_scores': {
            'standard_fluency_pct': round(result.fluency_scores.standard_fluency_pct, 1),
            'weighted_fluency_pct': round(result.fluency_scores.weighted_fluency_pct, 1),
            'clinical_fluency_pct': round(result.fluency_scores.clinical_fluency_pct, 1),
            'dysfluency_profile': result.fluency_scores.dysfluency_profile,
            'normative_level': result.fluency_scores.normative_level
        },

        # Summary
        'total_words': result.total_words,
        'total_syllables': result.total_syllables,
        'total_duration': round(result.total_duration, 2),
        'clinical_notes': result.clinical_notes
    }
