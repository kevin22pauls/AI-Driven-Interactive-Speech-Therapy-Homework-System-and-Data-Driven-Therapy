"""
Phoneme Lookup Service using CMU Pronouncing Dictionary

This module provides phoneme transcription for English words using the CMU
Pronouncing Dictionary. It's essential for phoneme-level speech analysis in
the therapy system.

Features:
- Fast dictionary-based phoneme lookup
- Multiple pronunciation variants support
- Normalization of input text
- Fallback handling for unknown words
"""

import os
import logging
from typing import List, Optional, Dict, Tuple
import re

logger = logging.getLogger(__name__)

# ARPAbet to IPA mapping for clinical clarity (optional, for display)
ARPABET_TO_IPA = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ',
    'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
    'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ',
    'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
    'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
    'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ',
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}


class PhonemeDict:
    """
    CMU Pronouncing Dictionary wrapper for phoneme lookup.

    Loads the dictionary once at initialization for fast lookups.
    """

    def __init__(self, dict_path: Optional[str] = None):
        """
        Initialize the phoneme dictionary.

        Args:
            dict_path: Path to CMU dict file. If None, uses default location.
        """
        if dict_path is None:
            # Default path relative to backend directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dict_path = os.path.join(base_dir, "data", "cmudict-0.7b.txt")

        self.dict_path = dict_path
        self.phoneme_dict: Dict[str, List[List[str]]] = {}
        self._load_dictionary()

    def _load_dictionary(self):
        """Load the CMU Pronouncing Dictionary into memory."""
        if not os.path.exists(self.dict_path):
            logger.error(f"CMU Dictionary not found at {self.dict_path}")
            return

        try:
            with open(self.dict_path, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(';;;'):
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    word = parts[0].lower()
                    # Handle variant pronunciations like "word(2)"
                    word = re.sub(r'\(\d+\)', '', word)

                    # Remove stress markers (0, 1, 2) from phonemes
                    phonemes = [re.sub(r'\d', '', p) for p in parts[1:]]

                    if word not in self.phoneme_dict:
                        self.phoneme_dict[word] = []

                    self.phoneme_dict[word].append(phonemes)

            logger.info(f"Loaded {len(self.phoneme_dict)} words from CMU Dictionary")

        except Exception as e:
            logger.error(f"Error loading CMU Dictionary: {e}")

    def get_phonemes(self, word: str, variant: int = 0) -> Optional[List[str]]:
        """
        Get phonemes for a word.

        Args:
            word: Word to look up (will be normalized)
            variant: Which pronunciation variant to return (default: 0 = first)

        Returns:
            List of phonemes (ARPAbet format without stress), or None if not found
        """
        word_normalized = word.lower().strip()

        # Remove punctuation
        word_normalized = re.sub(r'[^\w\s-]', '', word_normalized)

        if word_normalized in self.phoneme_dict:
            pronunciations = self.phoneme_dict[word_normalized]
            if variant < len(pronunciations):
                return pronunciations[variant]
            else:
                return pronunciations[0]  # Return first if variant doesn't exist

        return None

    def get_all_pronunciations(self, word: str) -> List[List[str]]:
        """
        Get all pronunciation variants for a word.

        Args:
            word: Word to look up

        Returns:
            List of pronunciation variants (each is a list of phonemes)
        """
        word_normalized = word.lower().strip()
        word_normalized = re.sub(r'[^\w\s-]', '', word_normalized)

        return self.phoneme_dict.get(word_normalized, [])

    def text_to_phonemes(self, text: str) -> List[Tuple[str, Optional[List[str]]]]:
        """
        Convert text to phonemes word by word.

        Args:
            text: Input text (sentence or phrase)

        Returns:
            List of tuples: (word, phonemes_list)
            phonemes_list is None if word not found in dictionary
        """
        words = text.lower().split()
        result = []

        for word in words:
            # Clean word
            word_clean = re.sub(r'[^\w\s-]', '', word)
            if not word_clean:
                continue

            phonemes = self.get_phonemes(word_clean)
            result.append((word_clean, phonemes))

        return result

    def phonemes_to_ipa(self, phonemes: List[str]) -> str:
        """
        Convert ARPAbet phonemes to IPA for display.

        Args:
            phonemes: List of ARPAbet phonemes

        Returns:
            IPA transcription string
        """
        ipa_symbols = []
        for phoneme in phonemes:
            # Remove any remaining stress markers
            phoneme_clean = re.sub(r'\d', '', phoneme)
            ipa_symbols.append(ARPABET_TO_IPA.get(phoneme_clean, phoneme_clean))

        return ''.join(ipa_symbols)


# Global instance for efficient reuse
_phoneme_dict_instance: Optional[PhonemeDict] = None


def get_phoneme_dict() -> PhonemeDict:
    """
    Get the global PhonemeDict instance (singleton pattern).

    Returns:
        PhonemeDict instance
    """
    global _phoneme_dict_instance

    if _phoneme_dict_instance is None:
        _phoneme_dict_instance = PhonemeDict()

    return _phoneme_dict_instance


def lookup_word_phonemes(word: str) -> Optional[List[str]]:
    """
    Convenience function to look up phonemes for a single word.

    Args:
        word: Word to look up

    Returns:
        List of phonemes or None if not found
    """
    phoneme_dict = get_phoneme_dict()
    return phoneme_dict.get_phonemes(word)


def lookup_text_phonemes(text: str) -> List[Tuple[str, Optional[List[str]]]]:
    """
    Convenience function to get phonemes for text.

    Args:
        text: Input text

    Returns:
        List of (word, phonemes) tuples
    """
    phoneme_dict = get_phoneme_dict()
    return phoneme_dict.text_to_phonemes(text)
