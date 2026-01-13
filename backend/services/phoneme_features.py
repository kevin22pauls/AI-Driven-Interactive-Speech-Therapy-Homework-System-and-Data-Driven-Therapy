"""
Phoneme Feature Matrix for Weighted Phonetic Distance Calculations

This module implements distinctive feature theory for calculating phonetic similarity
between phonemes, critical for weighted phoneme error rate (WPER) in aphasia analysis.

Features are based on articulatory phonetics:
- Voicing: ±voice
- Place of articulation: labial, dental, alveolar, palatal, velar, glottal
- Manner of articulation: stop, fricative, affricate, nasal, liquid, glide
- Height (vowels): high, mid, low
- Backness (vowels): front, central, back
- Tenseness (vowels): tense, lax
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Feature weights for distance calculation
FEATURE_WEIGHTS = {
    'voicing': 0.15,
    'place': 0.25,
    'manner': 0.30,
    'height': 0.15,
    'backness': 0.15
}

# Consonant features (ARPAbet)
CONSONANT_FEATURES = {
    # Stops
    'P': {'voicing': 'voiceless', 'place': 'labial', 'manner': 'stop'},
    'B': {'voicing': 'voiced', 'place': 'labial', 'manner': 'stop'},
    'T': {'voicing': 'voiceless', 'place': 'alveolar', 'manner': 'stop'},
    'D': {'voicing': 'voiced', 'place': 'alveolar', 'manner': 'stop'},
    'K': {'voicing': 'voiceless', 'place': 'velar', 'manner': 'stop'},
    'G': {'voicing': 'voiced', 'place': 'velar', 'manner': 'stop'},

    # Fricatives
    'F': {'voicing': 'voiceless', 'place': 'labiodental', 'manner': 'fricative'},
    'V': {'voicing': 'voiced', 'place': 'labiodental', 'manner': 'fricative'},
    'TH': {'voicing': 'voiceless', 'place': 'dental', 'manner': 'fricative'},
    'DH': {'voicing': 'voiced', 'place': 'dental', 'manner': 'fricative'},
    'S': {'voicing': 'voiceless', 'place': 'alveolar', 'manner': 'fricative'},
    'Z': {'voicing': 'voiced', 'place': 'alveolar', 'manner': 'fricative'},
    'SH': {'voicing': 'voiceless', 'place': 'palatal', 'manner': 'fricative'},
    'ZH': {'voicing': 'voiced', 'place': 'palatal', 'manner': 'fricative'},
    'HH': {'voicing': 'voiceless', 'place': 'glottal', 'manner': 'fricative'},

    # Affricates
    'CH': {'voicing': 'voiceless', 'place': 'palatal', 'manner': 'affricate'},
    'JH': {'voicing': 'voiced', 'place': 'palatal', 'manner': 'affricate'},

    # Nasals
    'M': {'voicing': 'voiced', 'place': 'labial', 'manner': 'nasal'},
    'N': {'voicing': 'voiced', 'place': 'alveolar', 'manner': 'nasal'},
    'NG': {'voicing': 'voiced', 'place': 'velar', 'manner': 'nasal'},

    # Liquids
    'L': {'voicing': 'voiced', 'place': 'alveolar', 'manner': 'liquid'},
    'R': {'voicing': 'voiced', 'place': 'alveolar', 'manner': 'liquid'},

    # Glides
    'W': {'voicing': 'voiced', 'place': 'labial', 'manner': 'glide'},
    'Y': {'voicing': 'voiced', 'place': 'palatal', 'manner': 'glide'},
}

# Vowel features (ARPAbet - stress markers removed)
VOWEL_FEATURES = {
    # Front vowels
    'IY': {'height': 'high', 'backness': 'front', 'tenseness': 'tense'},
    'IH': {'height': 'high', 'backness': 'front', 'tenseness': 'lax'},
    'EY': {'height': 'mid', 'backness': 'front', 'tenseness': 'tense'},
    'EH': {'height': 'mid', 'backness': 'front', 'tenseness': 'lax'},
    'AE': {'height': 'low', 'backness': 'front', 'tenseness': 'lax'},

    # Central vowels
    'AH': {'height': 'mid', 'backness': 'central', 'tenseness': 'lax'},
    'AX': {'height': 'mid', 'backness': 'central', 'tenseness': 'lax'},  # Schwa
    'ER': {'height': 'mid', 'backness': 'central', 'tenseness': 'lax'},  # R-colored

    # Back vowels
    'UW': {'height': 'high', 'backness': 'back', 'tenseness': 'tense'},
    'UH': {'height': 'high', 'backness': 'back', 'tenseness': 'lax'},
    'OW': {'height': 'mid', 'backness': 'back', 'tenseness': 'tense'},
    'AO': {'height': 'mid', 'backness': 'back', 'tenseness': 'lax'},
    'AA': {'height': 'low', 'backness': 'back', 'tenseness': 'lax'},

    # Diphthongs (represented by first element)
    'AY': {'height': 'low', 'backness': 'front', 'tenseness': 'tense'},
    'AW': {'height': 'low', 'backness': 'back', 'tenseness': 'tense'},
    'OY': {'height': 'mid', 'backness': 'back', 'tenseness': 'tense'},
}

# Place distance matrix (0-1 scale based on articulatory distance)
PLACE_DISTANCES = {
    ('labial', 'labial'): 0.0,
    ('labial', 'labiodental'): 0.2,
    ('labial', 'dental'): 0.4,
    ('labial', 'alveolar'): 0.5,
    ('labial', 'palatal'): 0.7,
    ('labial', 'velar'): 0.8,
    ('labial', 'glottal'): 1.0,

    ('labiodental', 'labiodental'): 0.0,
    ('labiodental', 'dental'): 0.2,
    ('labiodental', 'alveolar'): 0.3,
    ('labiodental', 'palatal'): 0.6,
    ('labiodental', 'velar'): 0.7,
    ('labiodental', 'glottal'): 0.9,

    ('dental', 'dental'): 0.0,
    ('dental', 'alveolar'): 0.2,
    ('dental', 'palatal'): 0.4,
    ('dental', 'velar'): 0.6,
    ('dental', 'glottal'): 0.8,

    ('alveolar', 'alveolar'): 0.0,
    ('alveolar', 'palatal'): 0.3,
    ('alveolar', 'velar'): 0.5,
    ('alveolar', 'glottal'): 0.7,

    ('palatal', 'palatal'): 0.0,
    ('palatal', 'velar'): 0.3,
    ('palatal', 'glottal'): 0.5,

    ('velar', 'velar'): 0.0,
    ('velar', 'glottal'): 0.3,

    ('glottal', 'glottal'): 0.0,
}

# Manner distance matrix
MANNER_DISTANCES = {
    ('stop', 'stop'): 0.0,
    ('stop', 'fricative'): 0.4,
    ('stop', 'affricate'): 0.3,
    ('stop', 'nasal'): 0.5,
    ('stop', 'liquid'): 0.6,
    ('stop', 'glide'): 0.7,

    ('fricative', 'fricative'): 0.0,
    ('fricative', 'affricate'): 0.2,
    ('fricative', 'nasal'): 0.5,
    ('fricative', 'liquid'): 0.5,
    ('fricative', 'glide'): 0.6,

    ('affricate', 'affricate'): 0.0,
    ('affricate', 'nasal'): 0.5,
    ('affricate', 'liquid'): 0.5,
    ('affricate', 'glide'): 0.6,

    ('nasal', 'nasal'): 0.0,
    ('nasal', 'liquid'): 0.3,
    ('nasal', 'glide'): 0.5,

    ('liquid', 'liquid'): 0.0,
    ('liquid', 'glide'): 0.3,

    ('glide', 'glide'): 0.0,
}

# Height distance matrix for vowels
HEIGHT_DISTANCES = {
    ('high', 'high'): 0.0,
    ('high', 'mid'): 0.5,
    ('high', 'low'): 1.0,
    ('mid', 'mid'): 0.0,
    ('mid', 'low'): 0.5,
    ('low', 'low'): 0.0,
}

# Backness distance matrix for vowels
BACKNESS_DISTANCES = {
    ('front', 'front'): 0.0,
    ('front', 'central'): 0.5,
    ('front', 'back'): 1.0,
    ('central', 'central'): 0.0,
    ('central', 'back'): 0.5,
    ('back', 'back'): 0.0,
}


def strip_stress(phoneme: str) -> str:
    """Remove stress markers (0, 1, 2) from phoneme."""
    return ''.join(c for c in phoneme if not c.isdigit())


def get_phoneme_features(phoneme: str) -> Optional[Dict]:
    """
    Get distinctive features for a phoneme.

    Args:
        phoneme: ARPAbet phoneme (may include stress markers)

    Returns:
        Dictionary of features or None if not found
    """
    clean_phoneme = strip_stress(phoneme)

    if clean_phoneme in CONSONANT_FEATURES:
        features = CONSONANT_FEATURES[clean_phoneme].copy()
        features['type'] = 'consonant'
        return features
    elif clean_phoneme in VOWEL_FEATURES:
        features = VOWEL_FEATURES[clean_phoneme].copy()
        features['type'] = 'vowel'
        return features

    logger.warning(f"Unknown phoneme: {phoneme}")
    return None


def get_distance_from_matrix(matrix: Dict, val1: str, val2: str) -> float:
    """Get distance from a symmetric matrix."""
    key1 = (val1, val2)
    key2 = (val2, val1)

    if key1 in matrix:
        return matrix[key1]
    elif key2 in matrix:
        return matrix[key2]
    else:
        # Default to maximum distance if not found
        return 1.0


def calculate_phonetic_distance(phoneme1: str, phoneme2: str) -> float:
    """
    Calculate phonetic distance between two phonemes based on distinctive features.

    Args:
        phoneme1: First phoneme (ARPAbet)
        phoneme2: Second phoneme (ARPAbet)

    Returns:
        Distance score between 0 (identical) and 1 (maximally different)
    """
    if strip_stress(phoneme1) == strip_stress(phoneme2):
        return 0.0

    features1 = get_phoneme_features(phoneme1)
    features2 = get_phoneme_features(phoneme2)

    if not features1 or not features2:
        # Unknown phoneme - return high distance
        return 0.8

    # Different phoneme types (consonant vs vowel) - maximum distance
    if features1['type'] != features2['type']:
        return 1.0

    total_distance = 0.0
    total_weight = 0.0

    if features1['type'] == 'consonant':
        # Consonant distance calculation
        # Voicing
        voicing_dist = 0.0 if features1['voicing'] == features2['voicing'] else 1.0
        total_distance += FEATURE_WEIGHTS['voicing'] * voicing_dist
        total_weight += FEATURE_WEIGHTS['voicing']

        # Place
        place_dist = get_distance_from_matrix(
            PLACE_DISTANCES,
            features1['place'],
            features2['place']
        )
        total_distance += FEATURE_WEIGHTS['place'] * place_dist
        total_weight += FEATURE_WEIGHTS['place']

        # Manner
        manner_dist = get_distance_from_matrix(
            MANNER_DISTANCES,
            features1['manner'],
            features2['manner']
        )
        total_distance += FEATURE_WEIGHTS['manner'] * manner_dist
        total_weight += FEATURE_WEIGHTS['manner']

    else:
        # Vowel distance calculation
        # Height
        height_dist = get_distance_from_matrix(
            HEIGHT_DISTANCES,
            features1['height'],
            features2['height']
        )
        total_distance += FEATURE_WEIGHTS['height'] * height_dist
        total_weight += FEATURE_WEIGHTS['height']

        # Backness
        backness_dist = get_distance_from_matrix(
            BACKNESS_DISTANCES,
            features1['backness'],
            features2['backness']
        )
        total_distance += FEATURE_WEIGHTS['backness'] * backness_dist
        total_weight += FEATURE_WEIGHTS['backness']

        # Tenseness
        tense_dist = 0.0 if features1.get('tenseness') == features2.get('tenseness') else 0.3
        total_distance += 0.10 * tense_dist
        total_weight += 0.10

    return total_distance / total_weight if total_weight > 0 else 0.5


def is_phonetically_similar(phoneme1: str, phoneme2: str, threshold: float = 0.3) -> bool:
    """
    Check if two phonemes are phonetically similar.

    Args:
        phoneme1: First phoneme
        phoneme2: Second phoneme
        threshold: Distance threshold for similarity (default 0.3)

    Returns:
        True if phonemes are similar
    """
    return calculate_phonetic_distance(phoneme1, phoneme2) < threshold


def get_phoneme_class(phoneme: str) -> str:
    """
    Get the phoneme class for clinical categorization.

    Returns one of: 'stop', 'fricative', 'affricate', 'nasal', 'liquid',
                    'glide', 'front_vowel', 'central_vowel', 'back_vowel', 'unknown'
    """
    features = get_phoneme_features(phoneme)

    if not features:
        return 'unknown'

    if features['type'] == 'consonant':
        return features['manner']
    else:
        backness = features.get('backness', 'central')
        return f"{backness}_vowel"


@dataclass
class PhoneticDistanceResult:
    """Result of phonetic distance calculation with clinical interpretation."""
    distance: float
    similarity_level: str  # 'identical', 'similar', 'moderately_different', 'very_different'
    phoneme1_class: str
    phoneme2_class: str
    clinical_significance: str


def analyze_phonetic_substitution(expected: str, actual: str) -> PhoneticDistanceResult:
    """
    Analyze a phoneme substitution with clinical interpretation.

    Args:
        expected: Expected phoneme
        actual: Actual (produced) phoneme

    Returns:
        PhoneticDistanceResult with clinical insights
    """
    distance = calculate_phonetic_distance(expected, actual)
    p1_class = get_phoneme_class(expected)
    p2_class = get_phoneme_class(actual)

    # Determine similarity level
    if distance == 0:
        similarity = 'identical'
    elif distance < 0.3:
        similarity = 'similar'
    elif distance < 0.6:
        similarity = 'moderately_different'
    else:
        similarity = 'very_different'

    # Generate clinical significance
    if similarity == 'identical':
        clinical = "Correct production"
    elif similarity == 'similar':
        clinical = "Minor phonological error - phonological access largely intact"
    elif p1_class == p2_class:
        clinical = f"Within-class substitution ({p1_class}) - manner preserved"
    elif p1_class.endswith('_vowel') and p2_class.endswith('_vowel'):
        clinical = "Vowel substitution - may indicate dialect variation or vowel discrimination difficulty"
    else:
        clinical = f"Cross-class substitution ({p1_class} → {p2_class}) - significant phonological breakdown"

    return PhoneticDistanceResult(
        distance=distance,
        similarity_level=similarity,
        phoneme1_class=p1_class,
        phoneme2_class=p2_class,
        clinical_significance=clinical
    )
