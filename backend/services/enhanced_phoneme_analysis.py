"""
Enhanced Phoneme-Level Speech Analysis for Aphasia

This module provides advanced phoneme analysis specifically designed for disordered speech,
implementing:
- Weighted Phoneme Error Rate (WPER) with phonetic distance weighting
- Multi-attempt segmentation for conduite d'approche detection
- Error type classification with clinical significance
- Neologism and paraphasia detection

Based on clinical research in aphasia speech production.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import statistics

from services.phoneme_features import (
    calculate_phonetic_distance,
    is_phonetically_similar,
    get_phoneme_class,
    analyze_phonetic_substitution,
    strip_stress
)
from services.phoneme_lookup import get_phoneme_dict, lookup_text_phonemes

logger = logging.getLogger(__name__)


# Error type weights for aphasia-specific analysis
ERROR_TYPE_WEIGHTS = {
    'phonetically_similar_substitution': 0.3,  # d < 0.3 - minor error
    'phonetically_dissimilar_substitution': 0.7,  # d >= 0.3 - significant
    'phonemic_paraphasia': 0.5,  # Real word substitution
    'neologism': 1.0,  # Severe phonological disintegration
    'deletion_unstressed': 0.4,  # Common in apraxia
    'deletion_stressed': 0.8,  # More severe
    'insertion': 0.6,  # Perseverative or planning error
    'metathesis': 0.5,  # Sequencing error
}


@dataclass
class EnhancedPhonemeError:
    """Enhanced phoneme error with clinical weighting."""
    error_type: str
    position: int
    expected_phoneme: Optional[str]
    actual_phoneme: Optional[str]
    word: str
    phonetic_distance: float
    weight: float
    clinical_significance: str
    is_stressed: bool = False


@dataclass
class ProductionAttempt:
    """Represents a single production attempt for multi-attempt analysis."""
    phonemes: List[str]
    score: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class MultiAttemptResult:
    """Result of multi-attempt analysis for conduite d'approche."""
    best_attempt: List[str]
    best_score: float
    num_attempts: int
    conduite_d_approche: bool  # True if progressive improvement detected
    attempt_scores: List[float]
    clinical_notes: List[str]


@dataclass
class EnhancedPhonemeAnalysisResult:
    """Complete enhanced phoneme analysis result."""
    # Standard metrics
    per_rule: float  # Rule-based Phoneme Error Rate (from CMUdict comparison)
    wper: float  # Weighted Phoneme Error Rate
    total_phonemes: int

    # Error details
    errors: List[EnhancedPhonemeError]
    error_summary: Dict[str, int]
    problematic_phonemes: Dict[str, int]

    # Multi-attempt analysis
    multi_attempt_result: Optional[MultiAttemptResult]

    # Clinical insights
    error_pattern_analysis: Dict
    clinical_notes: List[str]

    # Phoneme class breakdown
    phoneme_class_errors: Dict[str, int]


def get_deletion_weight(phoneme: str, word: str = "", position: int = 0) -> float:
    """
    Get deletion weight based on stress and position.

    Args:
        phoneme: The deleted phoneme
        word: The word context
        position: Position in word

    Returns:
        Weight for this deletion
    """
    # Check if phoneme has stress marker (0, 1, 2 in ARPAbet)
    has_stress = any(c.isdigit() for c in phoneme)
    stress_level = 0

    for c in phoneme:
        if c.isdigit():
            stress_level = int(c)
            break

    # Primary stress deletion is more severe
    if stress_level == 1:
        return ERROR_TYPE_WEIGHTS['deletion_stressed']
    else:
        return ERROR_TYPE_WEIGHTS['deletion_unstressed']


def weighted_phoneme_distance(
    ref: List[str],
    hyp: List[str]
) -> Tuple[float, List[EnhancedPhonemeError]]:
    """
    Calculate weighted phoneme distance using modified Wagner-Fischer algorithm.

    This implementation uses phonetic distance for substitution costs,
    making phonetically similar substitutions less costly than dissimilar ones.

    Args:
        ref: Reference (expected) phoneme sequence
        hyp: Hypothesis (actual) phoneme sequence

    Returns:
        Tuple of (weighted_per, list of enhanced errors)
    """
    m, n = len(ref), len(hyp)

    if m == 0:
        return (1.0 if n > 0 else 0.0, [])

    # DP matrix for weighted distances
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    # Backtrack matrix to reconstruct alignment
    backtrack = [[None] * (n + 1) for _ in range(m + 1)]

    # Initialize with deletion weights
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + get_deletion_weight(ref[i-1])
        backtrack[i][0] = 'del'

    # Initialize with insertion weights
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + ERROR_TYPE_WEIGHTS['insertion']
        backtrack[0][j] = 'ins'

    # Fill DP matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if strip_stress(ref[i-1]) == strip_stress(hyp[j-1]):
                dp[i][j] = dp[i-1][j-1]
                backtrack[i][j] = 'match'
            else:
                # Calculate phonetic distance for substitution
                phon_dist = calculate_phonetic_distance(ref[i-1], hyp[j-1])

                # Determine substitution weight based on phonetic similarity
                if phon_dist < 0.3:
                    sub_cost = ERROR_TYPE_WEIGHTS['phonetically_similar_substitution']
                else:
                    sub_cost = ERROR_TYPE_WEIGHTS['phonetically_dissimilar_substitution']

                del_cost = get_deletion_weight(ref[i-1])
                ins_cost = ERROR_TYPE_WEIGHTS['insertion']

                costs = [
                    (dp[i-1][j-1] + sub_cost, 'sub'),
                    (dp[i-1][j] + del_cost, 'del'),
                    (dp[i][j-1] + ins_cost, 'ins')
                ]

                min_cost, op = min(costs, key=lambda x: x[0])
                dp[i][j] = min_cost
                backtrack[i][j] = op

    # Backtrack to get errors
    errors = []
    i, j = m, n

    while i > 0 or j > 0:
        if i == 0:
            errors.append(EnhancedPhonemeError(
                error_type='insertion',
                position=j-1,
                expected_phoneme=None,
                actual_phoneme=hyp[j-1],
                word='',
                phonetic_distance=1.0,
                weight=ERROR_TYPE_WEIGHTS['insertion'],
                clinical_significance='Extra phoneme inserted - possible perseveration'
            ))
            j -= 1
        elif j == 0:
            errors.append(EnhancedPhonemeError(
                error_type='deletion',
                position=i-1,
                expected_phoneme=ref[i-1],
                actual_phoneme=None,
                word='',
                phonetic_distance=1.0,
                weight=get_deletion_weight(ref[i-1]),
                clinical_significance='Phoneme deleted',
                is_stressed=any(c.isdigit() and int(c) == 1 for c in ref[i-1])
            ))
            i -= 1
        else:
            op = backtrack[i][j]

            if op == 'match':
                i -= 1
                j -= 1
            elif op == 'sub':
                phon_dist = calculate_phonetic_distance(ref[i-1], hyp[j-1])
                analysis = analyze_phonetic_substitution(ref[i-1], hyp[j-1])

                if phon_dist < 0.3:
                    error_type = 'phonetically_similar_substitution'
                    weight = ERROR_TYPE_WEIGHTS['phonetically_similar_substitution']
                else:
                    error_type = 'phonetically_dissimilar_substitution'
                    weight = ERROR_TYPE_WEIGHTS['phonetically_dissimilar_substitution']

                errors.append(EnhancedPhonemeError(
                    error_type=error_type,
                    position=i-1,
                    expected_phoneme=ref[i-1],
                    actual_phoneme=hyp[j-1],
                    word='',
                    phonetic_distance=phon_dist,
                    weight=weight,
                    clinical_significance=analysis.clinical_significance
                ))
                i -= 1
                j -= 1
            elif op == 'del':
                errors.append(EnhancedPhonemeError(
                    error_type='deletion',
                    position=i-1,
                    expected_phoneme=ref[i-1],
                    actual_phoneme=None,
                    word='',
                    phonetic_distance=1.0,
                    weight=get_deletion_weight(ref[i-1]),
                    clinical_significance='Phoneme omitted',
                    is_stressed=any(c.isdigit() and int(c) == 1 for c in ref[i-1])
                ))
                i -= 1
            else:  # ins
                errors.append(EnhancedPhonemeError(
                    error_type='insertion',
                    position=j-1,
                    expected_phoneme=None,
                    actual_phoneme=hyp[j-1],
                    word='',
                    phonetic_distance=1.0,
                    weight=ERROR_TYPE_WEIGHTS['insertion'],
                    clinical_significance='Extra phoneme produced'
                ))
                j -= 1

    errors.reverse()

    # Calculate WPER
    total_weight = sum(e.weight for e in errors)
    wper = total_weight / m if m > 0 else 0.0

    return min(1.0, wper), errors


def detect_metathesis(ref: List[str], hyp: List[str], errors: List[EnhancedPhonemeError]) -> List[EnhancedPhonemeError]:
    """
    Detect metathesis (transposition) patterns in errors.

    Metathesis is when two phonemes are swapped: "ask" → "aks"

    Args:
        ref: Reference phonemes
        hyp: Hypothesis phonemes
        errors: Current error list

    Returns:
        Updated error list with metathesis detection
    """
    # Look for adjacent deletion + insertion that could be metathesis
    updated_errors = []
    i = 0

    while i < len(errors):
        if i < len(errors) - 1:
            e1, e2 = errors[i], errors[i+1]

            # Check for deletion followed by insertion at adjacent positions
            if (e1.error_type == 'deletion' and e2.error_type == 'insertion' and
                e1.expected_phoneme and e2.actual_phoneme):
                # Check if swapped phonemes appear nearby in both sequences
                # This is a heuristic for metathesis detection
                if abs(e1.position - e2.position) <= 2:
                    # Mark as potential metathesis
                    updated_errors.append(EnhancedPhonemeError(
                        error_type='metathesis',
                        position=e1.position,
                        expected_phoneme=e1.expected_phoneme,
                        actual_phoneme=e2.actual_phoneme,
                        word=e1.word,
                        phonetic_distance=0.5,
                        weight=ERROR_TYPE_WEIGHTS['metathesis'],
                        clinical_significance='Phoneme sequencing error (metathesis)'
                    ))
                    i += 2
                    continue

        updated_errors.append(errors[i])
        i += 1

    return updated_errors


def segment_into_attempts(
    phoneme_sequence: List[Tuple[str, Dict]],
    pause_threshold: float = 0.3
) -> List[ProductionAttempt]:
    """
    Segment utterance into separate production attempts based on pauses
    and repetition patterns.

    This is critical for conduite d'approche detection in aphasia.

    Args:
        phoneme_sequence: List of (phoneme, timing_dict) tuples
        pause_threshold: Threshold in seconds for attempt segmentation

    Returns:
        List of ProductionAttempt objects
    """
    if not phoneme_sequence:
        return []

    attempts = []
    current_attempt_phonemes = []
    current_start = None

    for i, (phoneme, timing) in enumerate(phoneme_sequence):
        if current_start is None:
            current_start = timing.get('start', 0)

        if i > 0:
            prev_timing = phoneme_sequence[i-1][1]
            gap = timing.get('start', 0) - prev_timing.get('end', 0)

            # Check for new attempt indicators
            is_new_attempt = False

            # Long pause
            if gap > pause_threshold:
                is_new_attempt = True

            # Restart pattern (similar initial phonemes)
            if len(current_attempt_phonemes) >= 2:
                if strip_stress(phoneme) == strip_stress(current_attempt_phonemes[0]):
                    is_new_attempt = True

            if is_new_attempt and current_attempt_phonemes:
                attempts.append(ProductionAttempt(
                    phonemes=current_attempt_phonemes.copy(),
                    score=0.0,  # Will be calculated later
                    start_time=current_start,
                    end_time=prev_timing.get('end', 0)
                ))
                current_attempt_phonemes = []
                current_start = timing.get('start', 0)

        current_attempt_phonemes.append(phoneme)

    # Add final attempt
    if current_attempt_phonemes:
        final_timing = phoneme_sequence[-1][1] if phoneme_sequence else {}
        attempts.append(ProductionAttempt(
            phonemes=current_attempt_phonemes,
            score=0.0,
            start_time=current_start,
            end_time=final_timing.get('end', 0)
        ))

    return attempts


def score_attempt(attempt: List[str], target: List[str]) -> float:
    """
    Score a production attempt against target phonemes.

    Args:
        attempt: Produced phoneme sequence
        target: Target phoneme sequence

    Returns:
        Score between 0 (completely wrong) and 1 (perfect match)
    """
    if not attempt or not target:
        return 0.0

    wper, _ = weighted_phoneme_distance(target, attempt)
    return max(0.0, 1.0 - wper)


def analyze_multi_attempts(
    attempts: List[ProductionAttempt],
    target: List[str]
) -> MultiAttemptResult:
    """
    Analyze multiple production attempts for conduite d'approche pattern.

    Conduite d'approche is when a patient progressively improves their
    approximation to the target word through successive attempts.

    Args:
        attempts: List of production attempts
        target: Target phoneme sequence

    Returns:
        MultiAttemptResult with analysis
    """
    if not attempts:
        return MultiAttemptResult(
            best_attempt=[],
            best_score=0.0,
            num_attempts=0,
            conduite_d_approche=False,
            attempt_scores=[],
            clinical_notes=["No production attempts detected"]
        )

    # Score each attempt
    scores = []
    for attempt in attempts:
        score = score_attempt(attempt.phonemes, target)
        attempt.score = score
        scores.append(score)

    # Find best attempt
    best_idx = scores.index(max(scores))
    best_attempt = attempts[best_idx]

    # Detect conduite d'approche (progressive improvement)
    is_conduite = False
    if len(scores) > 1:
        # Check if scores are generally increasing
        improvements = sum(1 for i in range(len(scores)-1) if scores[i+1] >= scores[i])
        is_conduite = improvements >= (len(scores) - 1) * 0.7  # 70% improving

    # Generate clinical notes
    clinical_notes = []

    if len(attempts) == 1:
        clinical_notes.append("Single production attempt")
    else:
        clinical_notes.append(f"{len(attempts)} production attempts detected")

        if is_conduite:
            clinical_notes.append("Conduite d'approche pattern detected - progressive improvement toward target")
            clinical_notes.append("This indicates preserved phonological access with motor planning difficulties")
        else:
            clinical_notes.append("No clear improvement pattern across attempts")

        if best_idx == len(attempts) - 1:
            clinical_notes.append("Final attempt was best - good self-correction")
        elif best_idx == 0:
            clinical_notes.append("First attempt was best - subsequent attempts showed regression")

    return MultiAttemptResult(
        best_attempt=best_attempt.phonemes,
        best_score=best_attempt.score,
        num_attempts=len(attempts),
        conduite_d_approche=is_conduite,
        attempt_scores=scores,
        clinical_notes=clinical_notes
    )


def analyze_error_patterns(errors: List[EnhancedPhonemeError]) -> Dict:
    """
    Analyze error patterns for clinical insights.

    Args:
        errors: List of enhanced phoneme errors

    Returns:
        Dictionary with pattern analysis
    """
    if not errors:
        return {
            'dominant_error_type': None,
            'phonetically_similar_ratio': 0.0,
            'class_preservation_ratio': 0.0,
            'common_substitutions': [],
            'affected_phoneme_classes': {}
        }

    # Count error types
    error_type_counts = {}
    for e in errors:
        error_type_counts[e.error_type] = error_type_counts.get(e.error_type, 0) + 1

    dominant_type = max(error_type_counts.items(), key=lambda x: x[1])[0] if error_type_counts else None

    # Calculate phonetically similar ratio for substitutions
    substitutions = [e for e in errors if 'substitution' in e.error_type]
    if substitutions:
        similar_count = sum(1 for e in substitutions if e.phonetic_distance < 0.3)
        similar_ratio = similar_count / len(substitutions)
    else:
        similar_ratio = 0.0

    # Analyze common substitution patterns
    substitution_patterns = {}
    for e in substitutions:
        if e.expected_phoneme and e.actual_phoneme:
            key = f"{strip_stress(e.expected_phoneme)}→{strip_stress(e.actual_phoneme)}"
            substitution_patterns[key] = substitution_patterns.get(key, 0) + 1

    common_subs = sorted(substitution_patterns.items(), key=lambda x: x[1], reverse=True)[:5]

    # Analyze affected phoneme classes
    class_errors = {}
    for e in errors:
        if e.expected_phoneme:
            pclass = get_phoneme_class(e.expected_phoneme)
            class_errors[pclass] = class_errors.get(pclass, 0) + 1

    return {
        'dominant_error_type': dominant_type,
        'error_type_distribution': error_type_counts,
        'phonetically_similar_ratio': similar_ratio,
        'common_substitutions': common_subs,
        'affected_phoneme_classes': class_errors
    }


def generate_enhanced_clinical_notes(
    wper: float,
    per: float,
    errors: List[EnhancedPhonemeError],
    pattern_analysis: Dict,
    multi_attempt: Optional[MultiAttemptResult]
) -> List[str]:
    """
    Generate comprehensive clinical notes based on enhanced analysis.

    Args:
        wper: Weighted Phoneme Error Rate
        per: Standard Phoneme Error Rate
        errors: List of enhanced errors
        pattern_analysis: Error pattern analysis
        multi_attempt: Multi-attempt analysis result

    Returns:
        List of clinical insight strings
    """
    notes = []

    # Overall assessment using WPER
    if wper == 0:
        notes.append("Excellent phoneme production - no errors detected")
    elif wper < 0.15:
        notes.append("Mild phoneme errors - largely accurate production")
    elif wper < 0.30:
        notes.append("Moderate phoneme errors - some phonological difficulty")
    elif wper < 0.50:
        notes.append("Significant phoneme errors - phonological breakdown evident")
    else:
        notes.append("Severe phoneme errors - substantial phonological impairment")

    # Compare WPER to PER for insights (only valid for substitution errors)
    # Note: This comparison is only meaningful when errors are substitutions, not deletions
    similar_ratio = pattern_analysis.get('phonetically_similar_ratio', 0)
    dominant = pattern_analysis.get('dominant_error_type', '')

    # Only make phonetic similarity claims based on actual ratio, not WPER/PER comparison
    # The WPER/PER comparison can be misleading with deletions (all deletions have weight 0.4)
    if similar_ratio > 0.5 and 'substitution' in dominant:
        notes.append("Most errors are phonetically similar - phonological system partially preserved")
    elif similar_ratio < 0.3 and 'substitution' in dominant and len(errors) > 2:
        notes.append("Errors tend to be phonetically dissimilar - more severe phonological disruption")

    # Error type insights
    if pattern_analysis.get('dominant_error_type'):
        dominant = pattern_analysis['dominant_error_type']
        if 'substitution' in dominant:
            notes.append(f"Substitutions are the primary error type")
        elif dominant == 'deletion':
            notes.append("Deletions are the primary error type - may indicate apraxic component")
        elif dominant == 'insertion':
            notes.append("Insertions are common - possible perseveration or planning errors")
        elif dominant == 'metathesis':
            notes.append("Sequencing errors (metathesis) detected - motor programming difficulty")

    # Phonetic similarity insights (additional detail if substitutions are dominant)
    if similar_ratio > 0.7 and 'substitution' in dominant:
        notes.append("High proportion of phonetically similar errors - good phonological awareness")

    # Common substitution patterns
    common_subs = pattern_analysis.get('common_substitutions', [])
    if common_subs:
        top_sub = common_subs[0]
        notes.append(f"Most common substitution: {top_sub[0]} ({top_sub[1]}x)")

    # Phoneme class recommendations
    class_errors = pattern_analysis.get('affected_phoneme_classes', {})
    if class_errors:
        worst_class = max(class_errors.items(), key=lambda x: x[1])[0]
        recommendations = {
            'fricative': "Focus on fricative production exercises",
            'stop': "Practice stop consonant release timing",
            'nasal': "Work on nasal resonance control",
            'liquid': "Liquid consonant (L, R) practice recommended",
            'glide': "Glide production exercises suggested",
            'front_vowel': "Front vowel discrimination training recommended",
            'back_vowel': "Back vowel articulation practice needed",
            'central_vowel': "Central vowel (schwa) clarity exercises recommended"
        }
        if worst_class in recommendations:
            notes.append(recommendations[worst_class])

    # Multi-attempt notes
    if multi_attempt and multi_attempt.num_attempts > 1:
        notes.extend(multi_attempt.clinical_notes)

    return notes


def analyze_phonemes_enhanced(
    expected_text: str,
    actual_text: str,
    word_timings: Optional[List[Dict]] = None,
    ml_detected_phonemes: Optional[List[str]] = None
) -> EnhancedPhonemeAnalysisResult:
    """
    Perform enhanced phoneme-level analysis with clinical weighting.

    This is the main entry point for enhanced phoneme analysis,
    replacing the standard analyze_phonemes function.

    Args:
        expected_text: Expected text (prompt or answer)
        actual_text: Actual transcribed text
        word_timings: Optional word-level timing information
        ml_detected_phonemes: Optional ML-detected phonemes from Wav2Vec2 (ARPAbet).
                              If provided, these are used as the actual phonemes instead
                              of CMUdict lookup on transcript. This provides more accurate
                              phoneme analysis when acoustic detection is available.

    Returns:
        EnhancedPhonemeAnalysisResult with complete analysis
    """
    phoneme_dict = get_phoneme_dict()

    # Convert expected text to phonemes (always use CMUdict for expected)
    expected_words_phonemes = phoneme_dict.text_to_phonemes(expected_text)

    # Flatten to get full phoneme sequences
    expected_phonemes = []
    for word, phonemes in expected_words_phonemes:
        if phonemes:
            expected_phonemes.extend(phonemes)

    # For actual phonemes: use ML-detected if available, otherwise fall back to CMUdict
    if ml_detected_phonemes:
        # Use ML-detected phonemes (acoustic ground truth)
        actual_phonemes = ml_detected_phonemes
        # Create dummy word-phoneme mapping for error attribution
        actual_words_phonemes = [("(ml-detected)", ml_detected_phonemes)]
        logger.info(f"Using {len(ml_detected_phonemes)} ML-detected phonemes for analysis")
    else:
        # Fall back to CMUdict lookup (text-based)
        actual_words_phonemes = phoneme_dict.text_to_phonemes(actual_text)
        actual_phonemes = []
        for word, phonemes in actual_words_phonemes:
            if phonemes:
                actual_phonemes.extend(phonemes)

    # Calculate standard PER
    if expected_phonemes:
        import editdistance
        standard_distance = editdistance.eval(
            [strip_stress(p) for p in expected_phonemes],
            [strip_stress(p) for p in actual_phonemes]
        )
        per = min(1.0, standard_distance / len(expected_phonemes))
    else:
        per = 0.0 if not actual_phonemes else 1.0

    # Calculate WPER with enhanced errors
    wper, errors = weighted_phoneme_distance(expected_phonemes, actual_phonemes)

    # Detect metathesis
    errors = detect_metathesis(expected_phonemes, actual_phonemes, errors)

    # Map errors to words
    expected_word_boundaries = []
    pos = 0
    for word, phonemes in expected_words_phonemes:
        if phonemes:
            expected_word_boundaries.append((word, pos, pos + len(phonemes)))
            pos += len(phonemes)

    for error in errors:
        for word, start, end in expected_word_boundaries:
            if start <= error.position < end:
                error.word = word
                break

    # Multi-attempt analysis (if timing available)
    multi_attempt_result = None
    if word_timings:
        # Create phoneme sequence with timings
        phoneme_timings = []
        for word_info in word_timings:
            word = word_info.get('text', '')
            word_phonemes = phoneme_dict.text_to_phonemes(word)
            for _, phonemes in word_phonemes:
                if phonemes:
                    for p in phonemes:
                        phoneme_timings.append((p, {
                            'start': word_info.get('start', 0),
                            'end': word_info.get('end', 0)
                        }))

        if phoneme_timings:
            attempts = segment_into_attempts(phoneme_timings)
            if len(attempts) > 1:
                multi_attempt_result = analyze_multi_attempts(attempts, expected_phonemes)

    # Error summary
    error_summary = {}
    for e in errors:
        error_summary[e.error_type] = error_summary.get(e.error_type, 0) + 1

    # Problematic phonemes
    problematic_phonemes = {}
    for e in errors:
        if e.expected_phoneme:
            clean_phoneme = strip_stress(e.expected_phoneme)
            problematic_phonemes[clean_phoneme] = problematic_phonemes.get(clean_phoneme, 0) + 1

    # Phoneme class errors
    phoneme_class_errors = {}
    for e in errors:
        if e.expected_phoneme:
            pclass = get_phoneme_class(e.expected_phoneme)
            phoneme_class_errors[pclass] = phoneme_class_errors.get(pclass, 0) + 1

    # Pattern analysis
    pattern_analysis = analyze_error_patterns(errors)

    # Clinical notes
    clinical_notes = generate_enhanced_clinical_notes(
        wper, per, errors, pattern_analysis, multi_attempt_result
    )

    return EnhancedPhonemeAnalysisResult(
        per_rule=per,
        wper=wper,
        total_phonemes=len(expected_phonemes),
        errors=errors,
        error_summary=error_summary,
        problematic_phonemes=problematic_phonemes,
        multi_attempt_result=multi_attempt_result,
        error_pattern_analysis=pattern_analysis,
        clinical_notes=clinical_notes,
        phoneme_class_errors=phoneme_class_errors
    )


def format_enhanced_phoneme_result_for_api(result: EnhancedPhonemeAnalysisResult) -> Dict:
    """
    Format EnhancedPhonemeAnalysisResult for API response.

    Args:
        result: EnhancedPhonemeAnalysisResult object

    Returns:
        Dictionary suitable for JSON serialization
    """
    multi_attempt_data = None
    if result.multi_attempt_result:
        multi_attempt_data = {
            'best_score': round(result.multi_attempt_result.best_score, 3),
            'num_attempts': result.multi_attempt_result.num_attempts,
            'conduite_d_approche': result.multi_attempt_result.conduite_d_approche,
            'attempt_scores': [round(s, 3) for s in result.multi_attempt_result.attempt_scores],
            'clinical_notes': result.multi_attempt_result.clinical_notes
        }

    return {
        "per_rule": round(result.per_rule, 3),
        "wper": round(result.wper, 3),
        "total_phonemes": result.total_phonemes,
        "error_count": len(result.errors),
        "error_summary": result.error_summary,
        "problematic_phonemes": result.problematic_phonemes,
        "phoneme_class_errors": result.phoneme_class_errors,
        "errors": [
            {
                "type": e.error_type,
                "position": e.position,
                "expected": e.expected_phoneme,
                "actual": e.actual_phoneme,
                "word": e.word,
                "phonetic_distance": round(e.phonetic_distance, 3),
                "weight": round(e.weight, 3),
                "clinical_significance": e.clinical_significance,
                "is_stressed": e.is_stressed
            }
            for e in result.errors
        ],
        "error_pattern_analysis": {
            "dominant_error_type": result.error_pattern_analysis.get('dominant_error_type'),
            "phonetically_similar_ratio": round(result.error_pattern_analysis.get('phonetically_similar_ratio', 0), 3),
            "common_substitutions": result.error_pattern_analysis.get('common_substitutions', []),
            "error_type_distribution": result.error_pattern_analysis.get('error_type_distribution', {})
        },
        "multi_attempt_analysis": multi_attempt_data,
        "clinical_notes": result.clinical_notes
    }
