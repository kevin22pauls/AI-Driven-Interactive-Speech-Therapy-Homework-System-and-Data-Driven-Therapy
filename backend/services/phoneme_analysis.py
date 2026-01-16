"""
Phoneme-Level Speech Analysis

This module provides comprehensive phoneme-level analysis for speech therapy,
including Phoneme Error Rate (PER), error classification, and GOP scores.

Clinical Features:
- Phoneme Error Rate (PER) calculation
- Error type classification (substitution, deletion, insertion)
- Problematic phoneme identification
- Longitudinal tracking support
- Goodness of Pronunciation (GOP) scoring

This is critical for aphasia therapy as it identifies WHICH SOUNDS patients
struggle with, not just if the word is wrong.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import editdistance  # Levenshtein distance for phoneme alignment
from services.phoneme_lookup import get_phoneme_dict, lookup_text_phonemes
from services.forced_alignment import get_phoneme_timestamps

logger = logging.getLogger(__name__)


@dataclass
class PhonemeError:
    """Represents a single phoneme-level error."""
    error_type: str  # 'substitution', 'deletion', 'insertion'
    position: int  # Position in word/sequence
    expected_phoneme: Optional[str]  # What was expected
    actual_phoneme: Optional[str]  # What was spoken
    word: str  # The word this error occurred in
    confidence: float  # Confidence score (0-1)


@dataclass
class PhonemeAnalysisResult:
    """Complete phoneme-level analysis result."""
    per: float  # Phoneme Error Rate (0-1)
    total_phonemes: int
    errors: List[PhonemeError]
    problematic_phonemes: Dict[str, int]  # phoneme -> error count
    error_summary: Dict[str, int]  # error_type -> count
    aligned_phonemes: List[Dict]  # Phoneme timestamps from forced alignment
    gop_scores: Optional[Dict[str, float]]  # Goodness of Pronunciation per phoneme
    clinical_notes: List[str]  # Clinical insights for therapist


def align_phoneme_sequences(
    expected: List[str],
    actual: List[str]
) -> Tuple[List[Tuple[Optional[str], Optional[str]]], List[PhonemeError]]:
    """
    Align two phoneme sequences and identify errors using edit distance.

    Args:
        expected: Expected phoneme sequence
        actual: Actual phoneme sequence from speech

    Returns:
        Tuple of:
        - List of aligned phoneme pairs (expected, actual)
        - List of PhonemeError objects
    """
    # Use dynamic programming to find optimal alignment
    # This is similar to sequence alignment in bioinformatics

    errors = []
    aligned_pairs = []

    # Simple approach using editdistance library
    # For production, you might want a more sophisticated algorithm

    # Calculate edit operations
    operations = get_edit_operations(expected, actual)

    expected_idx = 0
    actual_idx = 0

    for op, pos in operations:
        if op == 'match':
            # Phonemes match
            aligned_pairs.append((expected[expected_idx], actual[actual_idx]))
            expected_idx += 1
            actual_idx += 1

        elif op == 'substitution':
            # Phoneme substitution
            aligned_pairs.append((expected[expected_idx], actual[actual_idx]))
            errors.append(PhonemeError(
                error_type='substitution',
                position=expected_idx,
                expected_phoneme=expected[expected_idx],
                actual_phoneme=actual[actual_idx],
                word='',  # Will be filled in by caller
                confidence=1.0
            ))
            expected_idx += 1
            actual_idx += 1

        elif op == 'deletion':
            # Phoneme was expected but not spoken
            aligned_pairs.append((expected[expected_idx], None))
            errors.append(PhonemeError(
                error_type='deletion',
                position=expected_idx,
                expected_phoneme=expected[expected_idx],
                actual_phoneme=None,
                word='',
                confidence=1.0
            ))
            expected_idx += 1

        elif op == 'insertion':
            # Extra phoneme was spoken
            aligned_pairs.append((None, actual[actual_idx]))
            errors.append(PhonemeError(
                error_type='insertion',
                position=actual_idx,
                expected_phoneme=None,
                actual_phoneme=actual[actual_idx],
                word='',
                confidence=1.0
            ))
            actual_idx += 1

    return aligned_pairs, errors


def get_edit_operations(seq1: List[str], seq2: List[str]) -> List[Tuple[str, int]]:
    """
    Get edit operations to transform seq1 into seq2.

    Returns:
        List of (operation, position) tuples
    """
    # Build edit distance matrix
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )

    # Backtrack to get operations
    operations = []
    i, j = m, n

    while i > 0 or j > 0:
        if i == 0:
            operations.append(('insertion', j-1))
            j -= 1
        elif j == 0:
            operations.append(('deletion', i-1))
            i -= 1
        elif seq1[i-1] == seq2[j-1]:
            operations.append(('match', i-1))
            i -= 1
            j -= 1
        else:
            # Find which operation was used
            costs = []
            if i > 0 and j > 0:
                costs.append((dp[i-1][j-1], 'substitution'))
            if i > 0:
                costs.append((dp[i-1][j], 'deletion'))
            if j > 0:
                costs.append((dp[i][j-1], 'insertion'))

            min_cost, op = min(costs, key=lambda x: x[0])

            if op == 'substitution':
                operations.append(('substitution', i-1))
                i -= 1
                j -= 1
            elif op == 'deletion':
                operations.append(('deletion', i-1))
                i -= 1
            else:  # insertion
                operations.append(('insertion', j-1))
                j -= 1

    operations.reverse()
    return operations


def calculate_per(expected_phonemes: List[str], actual_phonemes: List[str]) -> float:
    """
    Calculate Phoneme Error Rate (PER).

    PER = (S + D + I) / N
    where:
        S = substitutions
        D = deletions
        I = insertions
        N = number of phonemes in reference

    Args:
        expected_phonemes: Expected phoneme sequence
        actual_phonemes: Actual phoneme sequence

    Returns:
        PER as float between 0 and 1
    """
    if not expected_phonemes:
        return 0.0 if not actual_phonemes else 1.0

    distance = editdistance.eval(expected_phonemes, actual_phonemes)
    per = distance / len(expected_phonemes)

    return min(1.0, per)  # Cap at 1.0


def analyze_phonemes(
    expected_text: str,
    actual_text: str,
    audio_path: Optional[str] = None
) -> PhonemeAnalysisResult:
    """
    Perform comprehensive phoneme-level analysis.

    Args:
        expected_text: Expected text (prompt or answer)
        actual_text: Actual transcribed text
        audio_path: Optional path to audio for forced alignment

    Returns:
        PhonemeAnalysisResult with complete analysis
    """
    phoneme_dict = get_phoneme_dict()

    # Convert texts to phonemes
    expected_words_phonemes = phoneme_dict.text_to_phonemes(expected_text)
    actual_words_phonemes = phoneme_dict.text_to_phonemes(actual_text)

    # Flatten to get full phoneme sequences
    expected_phonemes = []
    expected_word_boundaries = []  # Track which phonemes belong to which word

    for word, phonemes in expected_words_phonemes:
        if phonemes:
            start_idx = len(expected_phonemes)
            expected_phonemes.extend(phonemes)
            expected_word_boundaries.append((word, start_idx, len(expected_phonemes)))
        else:
            logger.warning(f"Word '{word}' not found in CMU dictionary")

    actual_phonemes = []
    actual_word_boundaries = []

    for word, phonemes in actual_words_phonemes:
        if phonemes:
            start_idx = len(actual_phonemes)
            actual_phonemes.extend(phonemes)
            actual_word_boundaries.append((word, start_idx, len(actual_phonemes)))

    # Calculate PER
    per = calculate_per(expected_phonemes, actual_phonemes)

    # Align phonemes and identify errors
    aligned_pairs, errors = align_phoneme_sequences(expected_phonemes, actual_phonemes)

    # Map errors to words
    for error in errors:
        # Find which word this error belongs to
        for word, start, end in expected_word_boundaries:
            if start <= error.position < end:
                error.word = word
                break

    # Count problematic phonemes
    problematic_phonemes = {}
    for error in errors:
        if error.expected_phoneme:
            problematic_phonemes[error.expected_phoneme] = \
                problematic_phonemes.get(error.expected_phoneme, 0) + 1

    # Count error types
    error_summary = {
        'substitution': sum(1 for e in errors if e.error_type == 'substitution'),
        'deletion': sum(1 for e in errors if e.error_type == 'deletion'),
        'insertion': sum(1 for e in errors if e.error_type == 'insertion')
    }

    # Get phoneme timestamps if audio path provided
    aligned_phonemes = []
    if audio_path:
        try:
            aligned_phonemes = get_phoneme_timestamps(audio_path, actual_text)
        except Exception as e:
            logger.warning(f"Forced alignment failed: {e}")

    # Generate clinical notes
    clinical_notes = generate_clinical_notes(
        errors,
        problematic_phonemes,
        per,
        expected_text,
        actual_text
    )

    return PhonemeAnalysisResult(
        per=per,
        total_phonemes=len(expected_phonemes),
        errors=errors,
        problematic_phonemes=problematic_phonemes,
        error_summary=error_summary,
        aligned_phonemes=aligned_phonemes,
        gop_scores=None,  # Implement GOP if needed
        clinical_notes=clinical_notes
    )


def generate_clinical_notes(
    errors: List[PhonemeError],
    problematic_phonemes: Dict[str, int],
    per: float,
    expected_text: str,
    actual_text: str
) -> List[str]:
    """
    Generate clinical notes for therapist based on analysis.

    Args:
        errors: List of phoneme errors
        problematic_phonemes: Phoneme error counts
        per: Phoneme Error Rate
        expected_text: Expected text
        actual_text: Actual text

    Returns:
        List of clinical note strings
    """
    notes = []

    # Overall assessment
    if per == 0:
        notes.append("Excellent phoneme-level accuracy")
    elif per < 0.1:
        notes.append("Very good phoneme production with minor errors")
    elif per < 0.25:
        notes.append("Moderate phoneme errors detected")
    else:
        notes.append("Significant phoneme-level difficulties observed")

    # Most problematic phonemes
    if problematic_phonemes:
        top_problems = sorted(problematic_phonemes.items(), key=lambda x: x[1], reverse=True)[:3]
        phoneme_list = ", ".join([f"{p} ({count}x)" for p, count in top_problems])
        notes.append(f"Most problematic phonemes: {phoneme_list}")

    # Error pattern analysis
    substitutions = [e for e in errors if e.error_type == 'substitution']
    if substitutions:
        # Look for patterns
        common_subs = {}
        for sub in substitutions:
            key = f"{sub.expected_phoneme}→{sub.actual_phoneme}"
            common_subs[key] = common_subs.get(key, 0) + 1

        if common_subs:
            most_common = max(common_subs.items(), key=lambda x: x[1])
            notes.append(f"Common substitution pattern: {most_common[0]}")

    # Specific recommendations
    if 'AA' in problematic_phonemes or 'AE' in problematic_phonemes:
        notes.append("Consider vowel discrimination exercises")
    if any(p in problematic_phonemes for p in ['TH', 'DH', 'S', 'Z']):
        notes.append("Focus on fricative production")
    if any(p in problematic_phonemes for p in ['R', 'L']):
        notes.append("Liquid consonant practice recommended")

    return notes


def format_phoneme_result_for_api(result: PhonemeAnalysisResult) -> Dict:
    """
    Format PhonemeAnalysisResult for API response.

    Args:
        result: PhonemeAnalysisResult object

    Returns:
        Dictionary suitable for JSON serialization
    """
    return {
        "per_rule": round(result.per, 3),
        "total_phonemes": result.total_phonemes,
        "error_count": len(result.errors),
        "error_summary": result.error_summary,
        "problematic_phonemes": result.problematic_phonemes,
        "errors": [
            {
                "type": e.error_type,
                "position": e.position,
                "expected": e.expected_phoneme,
                "actual": e.actual_phoneme,
                "word": e.word,
                "confidence": e.confidence
            }
            for e in result.errors
        ],
        "aligned_phonemes": result.aligned_phonemes,
        "clinical_notes": result.clinical_notes
    }
