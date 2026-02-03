"""
Phoneme Mapping Between ARPAbet and IPA

This module provides bidirectional mapping between ARPAbet (CMU dict format)
and IPA (Wav2Vec2 model output format) for consistent phoneme comparison.

The Wav2Vec2 model (facebook/wav2vec2-lv-60-espeak-cv-ft) outputs IPA symbols,
while CMU dict and our text-based analysis use ARPAbet.
"""

import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

# ARPAbet to IPA mapping (comprehensive)
ARPABET_TO_IPA = {
    # Vowels
    'AA': 'ɑ',      # father
    'AE': 'æ',      # cat
    'AH': 'ʌ',      # but
    'AO': 'ɔ',      # dog
    'AW': 'aʊ',     # now
    'AY': 'aɪ',     # my
    'EH': 'ɛ',      # bed
    'ER': 'ɝ',      # bird (r-colored)
    'EY': 'eɪ',     # say
    'IH': 'ɪ',      # bit
    'IY': 'iː',     # beat
    'OW': 'oʊ',     # go
    'OY': 'ɔɪ',     # boy
    'UH': 'ʊ',      # book
    'UW': 'uː',     # boot

    # Consonants
    'B': 'b',
    'CH': 'tʃ',
    'D': 'd',
    'DH': 'ð',      # this
    'F': 'f',
    'G': 'ɡ',
    'HH': 'h',
    'JH': 'dʒ',     # judge
    'K': 'k',
    'L': 'l',
    'M': 'm',
    'N': 'n',
    'NG': 'ŋ',      # sing
    'P': 'p',
    'R': 'ɹ',       # red (technically ɹ in IPA)
    'S': 's',
    'SH': 'ʃ',      # she
    'T': 't',
    'TH': 'θ',      # think
    'V': 'v',
    'W': 'w',
    'Y': 'j',       # yes
    'Z': 'z',
    'ZH': 'ʒ',      # measure
}

# Build reverse mapping (IPA to ARPAbet)
# Note: This includes common variations found in wav2vec2 output
IPA_TO_ARPABET = {v: k for k, v in ARPABET_TO_IPA.items()}

# Add additional IPA variants that the model might produce
IPA_VARIANTS = {
    # Schwa variations
    'ə': 'AH',
    'ɐ': 'AH',
    'ᵻ': 'IH',
    'ƏL': 'AH',     # Syllabic L with schwa
    'Ə': 'AH',      # Capital schwa

    # R-colored vowels
    'ɚ': 'ER',
    'ɜ': 'ER',
    'ɜː': 'ER',     # Long version of ɜ
    'ɜːɹ': 'ER',    # ɜ with length marker and r
    'ɝː': 'ER',     # Long r-colored schwa

    # Vowel variations
    'i': 'IY',
    'u': 'UW',
    'e': 'EY',
    'o': 'OW',
    'a': 'AA',
    'ɑː': 'AA',
    'ɔː': 'AO',
    'ɔːɹ': 'AO',    # AO with r
    'Ɔːɹ': 'AO',    # Capital variant
    'iː': 'IY',
    'uː': 'UW',
    'eː': 'EY',
    'oː': 'OW',

    # Consonant variations
    'r': 'R',
    'ɾ': 'D',       # Flap often realized as D
    'ʁ': 'R',       # French R -> English R
    'β': 'V',       # Approximant -> V
    'ɣ': 'G',       # Velar fricative -> G
    'ç': 'HH',      # Palatal fricative
    'x': 'K',       # Velar fricative

    # Affricates
    'ts': 'T',      # TS cluster
    'dz': 'D',      # DZ cluster

    # Long vowels (remove length marker)
    'ɑ̃': 'AA',     # Nasalized
    'ɔ̃': 'AO',
    'ɛ̃': 'EH',

    # Special
    'ʔ': 'T',       # Glottal stop -> T approximation
}

# Merge variants into main IPA mapping
IPA_TO_ARPABET.update(IPA_VARIANTS)


def ipa_to_arpabet(ipa_phoneme: str) -> str:
    """
    Convert an IPA phoneme to ARPAbet.

    Args:
        ipa_phoneme: IPA phoneme symbol

    Returns:
        ARPAbet equivalent, or original if not found
    """
    # Normalize: lowercase for lookup
    ipa_lower = ipa_phoneme.lower()

    # Direct lookup
    if ipa_lower in IPA_TO_ARPABET:
        return IPA_TO_ARPABET[ipa_lower]

    # Try without combining characters (diacritics)
    base_char = ipa_lower.rstrip('ːˑ̃̈̊')
    if base_char in IPA_TO_ARPABET:
        return IPA_TO_ARPABET[base_char]

    # Return uppercase version as fallback
    logger.debug(f"Unknown IPA phoneme: {ipa_phoneme}")
    return ipa_phoneme.upper()


def arpabet_to_ipa(arpabet_phoneme: str) -> str:
    """
    Convert an ARPAbet phoneme to IPA.

    Args:
        arpabet_phoneme: ARPAbet phoneme (may include stress markers)

    Returns:
        IPA equivalent
    """
    # Remove stress markers
    clean = ''.join(c for c in arpabet_phoneme if not c.isdigit()).upper()

    return ARPABET_TO_IPA.get(clean, arpabet_phoneme.lower())


def convert_ipa_sequence_to_arpabet(ipa_phonemes: List[str]) -> List[str]:
    """
    Convert a sequence of IPA phonemes to ARPAbet.

    Args:
        ipa_phonemes: List of IPA phoneme symbols

    Returns:
        List of ARPAbet phonemes
    """
    return [ipa_to_arpabet(p) for p in ipa_phonemes]


def convert_arpabet_sequence_to_ipa(arpabet_phonemes: List[str]) -> List[str]:
    """
    Convert a sequence of ARPAbet phonemes to IPA.

    Args:
        arpabet_phonemes: List of ARPAbet phonemes

    Returns:
        List of IPA phoneme symbols
    """
    return [arpabet_to_ipa(p) for p in arpabet_phonemes]


def normalize_phoneme_for_comparison(phoneme: str, source: str = 'unknown') -> str:
    """
    Normalize a phoneme to ARPAbet for consistent comparison.

    Args:
        phoneme: Phoneme symbol (IPA or ARPAbet)
        source: 'ipa', 'arpabet', or 'unknown'

    Returns:
        Normalized ARPAbet phoneme
    """
    if source == 'ipa':
        return ipa_to_arpabet(phoneme)
    elif source == 'arpabet':
        # Just clean stress markers
        return ''.join(c for c in phoneme if not c.isdigit()).upper()
    else:
        # Try to detect - IPA typically has special unicode chars
        has_ipa_chars = any(ord(c) > 127 for c in phoneme)
        if has_ipa_chars:
            return ipa_to_arpabet(phoneme)
        else:
            return ''.join(c for c in phoneme if not c.isdigit()).upper()
