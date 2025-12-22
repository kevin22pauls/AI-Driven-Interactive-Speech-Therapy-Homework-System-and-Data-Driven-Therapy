"""
Unit Tests for Phoneme Analysis Module

These tests verify the correctness of phoneme-level speech analysis,
including PER calculation, error detection, and clinical insights generation.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.phoneme_analysis import (
    calculate_per,
    align_phoneme_sequences,
    analyze_phonemes,
    generate_clinical_notes,
    PhonemeError
)
from services.phoneme_lookup import PhonemeDict


class TestPhonemeErrorRate:
    """Tests for PER calculation"""

    def test_per_perfect_match(self):
        """Test PER when phonemes match perfectly"""
        expected = ['B', 'AH', 'T', 'AH', 'L']
        actual = ['B', 'AH', 'T', 'AH', 'L']
        per = calculate_per(expected, actual)
        assert per == 0.0, "Perfect match should have PER of 0"

    def test_per_complete_mismatch(self):
        """Test PER when all phonemes are wrong"""
        expected = ['B', 'AH', 'T', 'AH', 'L']
        actual = ['K', 'AE', 'P']
        per = calculate_per(expected, actual)
        assert per > 0.0, "Mismatch should have PER > 0"

    def test_per_with_substitution(self):
        """Test PER with phoneme substitution (bootle vs bottle)"""
        # bottle: B AH T AH L
        # bootle: B UW T AH L (AA -> UW substitution)
        expected = ['B', 'AH', 'T', 'AH', 'L']
        actual = ['B', 'UW', 'T', 'AH', 'L']
        per = calculate_per(expected, actual)
        # 1 error out of 5 phonemes = 0.2
        assert abs(per - 0.2) < 0.01, f"Expected PER ~0.2, got {per}"

    def test_per_with_deletion(self):
        """Test PER with phoneme deletion"""
        expected = ['B', 'AH', 'T', 'AH', 'L']
        actual = ['B', 'T', 'AH', 'L']  # Missing AH
        per = calculate_per(expected, actual)
        assert per > 0.0, "Deletion should result in PER > 0"

    def test_per_with_insertion(self):
        """Test PER with phoneme insertion"""
        expected = ['K', 'AH', 'P']
        actual = ['K', 'AH', 'P', 'S']  # Extra S
        per = calculate_per(expected, actual)
        assert per > 0.0, "Insertion should result in PER > 0"

    def test_per_empty_sequences(self):
        """Test PER with empty sequences"""
        assert calculate_per([], []) == 0.0
        assert calculate_per(['A'], []) == 1.0
        assert calculate_per([], ['A']) == 1.0


class TestPhonemeAlignment:
    """Tests for phoneme sequence alignment"""

    def test_alignment_perfect_match(self):
        """Test alignment with perfect match"""
        expected = ['B', 'AH', 'T']
        actual = ['B', 'AH', 'T']
        aligned, errors = align_phoneme_sequences(expected, actual)

        assert len(errors) == 0, "Perfect match should have no errors"
        assert len(aligned) == 3, "Should have 3 aligned pairs"

    def test_alignment_substitution(self):
        """Test alignment with substitution"""
        expected = ['B', 'AH', 'T']
        actual = ['B', 'UW', 'T']  # AH -> UW
        aligned, errors = align_phoneme_sequences(expected, actual)

        assert len(errors) == 1, "Should detect 1 error"
        assert errors[0].error_type == 'substitution'
        assert errors[0].expected_phoneme == 'AH'
        assert errors[0].actual_phoneme == 'UW'

    def test_alignment_deletion(self):
        """Test alignment with deletion"""
        expected = ['B', 'AH', 'T', 'AH', 'L']
        actual = ['B', 'T', 'AH', 'L']  # Missing middle AH
        aligned, errors = align_phoneme_sequences(expected, actual)

        deletions = [e for e in errors if e.error_type == 'deletion']
        assert len(deletions) > 0, "Should detect deletion"

    def test_alignment_insertion(self):
        """Test alignment with insertion"""
        expected = ['K', 'AH', 'P']
        actual = ['K', 'AH', 'P', 'S']  # Extra S
        aligned, errors = align_phoneme_sequences(expected, actual)

        insertions = [e for e in errors if e.error_type == 'insertion']
        assert len(insertions) > 0, "Should detect insertion"


class TestPhonemeAnalysis:
    """Integration tests for full phoneme analysis"""

    def test_analyze_simple_match(self):
        """Test analysis with matching text"""
        expected = "cup"
        actual = "cup"
        result = analyze_phonemes(expected, actual)

        assert result.per == 0.0, "Matching words should have PER of 0"
        assert result.total_phonemes > 0
        assert len(result.errors) == 0

    def test_analyze_bottle_bootle(self):
        """Test classic bottle/bootle example"""
        expected = "bottle"
        actual = "bootle"
        result = analyze_phonemes(expected, actual)

        assert result.per > 0.0, "Different words should have PER > 0"
        assert len(result.errors) > 0, "Should detect errors"
        assert len(result.clinical_notes) > 0, "Should generate clinical notes"

    def test_analyze_unknown_word(self):
        """Test analysis with word not in CMU dict"""
        # Even if words aren't in dict, should handle gracefully
        expected = "xyzabc"
        actual = "xyzabc"
        result = analyze_phonemes(expected, actual)

        # Should handle gracefully (no phonemes found)
        assert isinstance(result.per, float)

    def test_problematic_phonemes_tracking(self):
        """Test that problematic phonemes are tracked"""
        expected = "bottle"
        actual = "bootle"
        result = analyze_phonemes(expected, actual)

        # Should identify which phoneme was problematic
        assert len(result.problematic_phonemes) > 0


class TestClinicalNotes:
    """Tests for clinical note generation"""

    def test_clinical_notes_perfect(self):
        """Test notes for perfect pronunciation"""
        notes = generate_clinical_notes(
            errors=[],
            problematic_phonemes={},
            per=0.0,
            expected_text="cup",
            actual_text="cup"
        )

        assert len(notes) > 0
        assert any("excellent" in note.lower() for note in notes)

    def test_clinical_notes_with_errors(self):
        """Test notes when errors are present"""
        errors = [
            PhonemeError(
                error_type='substitution',
                position=1,
                expected_phoneme='AH',
                actual_phoneme='UW',
                word='bottle',
                confidence=1.0
            )
        ]
        problematic_phonemes = {'AH': 1}

        notes = generate_clinical_notes(
            errors=errors,
            problematic_phonemes=problematic_phonemes,
            per=0.2,
            expected_text="bottle",
            actual_text="bootle"
        )

        assert len(notes) > 0
        # Should mention the problematic phoneme
        notes_text = ' '.join(notes)
        assert 'AH' in notes_text or 'problematic' in notes_text.lower()

    def test_clinical_notes_vowel_issues(self):
        """Test that vowel issues generate appropriate recommendations"""
        problematic_phonemes = {'AA': 2, 'AE': 1}

        notes = generate_clinical_notes(
            errors=[],
            problematic_phonemes=problematic_phonemes,
            per=0.15,
            expected_text="cat",
            actual_text="cot"
        )

        notes_text = ' '.join(notes).lower()
        assert 'vowel' in notes_text


class TestPhonemeDict:
    """Tests for CMU Dictionary lookups"""

    def test_phoneme_dict_load(self):
        """Test that CMU dictionary loads"""
        phoneme_dict = PhonemeDict()
        assert len(phoneme_dict.phoneme_dict) > 0, "Dictionary should load words"

    def test_lookup_common_word(self):
        """Test lookup of common word"""
        phoneme_dict = PhonemeDict()
        phonemes = phoneme_dict.get_phonemes("bottle")

        assert phonemes is not None, "Should find 'bottle' in dictionary"
        assert len(phonemes) > 0, "Should return phoneme list"
        # bottle = B AH T AH L or similar
        assert 'B' in phonemes, "Should start with B"

    def test_lookup_case_insensitive(self):
        """Test that lookup is case insensitive"""
        phoneme_dict = PhonemeDict()
        lower = phoneme_dict.get_phonemes("bottle")
        upper = phoneme_dict.get_phonemes("BOTTLE")
        mixed = phoneme_dict.get_phonemes("Bottle")

        assert lower == upper == mixed, "Lookup should be case insensitive"

    def test_text_to_phonemes(self):
        """Test converting text to phonemes"""
        phoneme_dict = PhonemeDict()
        result = phoneme_dict.text_to_phonemes("I have a bottle")

        assert len(result) > 0, "Should return word-phoneme pairs"
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
