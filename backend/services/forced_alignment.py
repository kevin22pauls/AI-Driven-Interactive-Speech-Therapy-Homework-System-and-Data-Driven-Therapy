"""
Forced Alignment Service using Gentle

This module provides phoneme-level forced alignment by interfacing with the
Gentle forced aligner (running in Docker). Gentle aligns audio with a transcript
to produce phoneme-level timestamps.

Key Features:
- Integration with Gentle Docker service
- Phoneme-level timestamp extraction
- Robust error handling for alignment failures
- Fallback mechanisms for production reliability
"""

import os
import logging
import requests
from typing import Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class GentleAligner:
    """
    Wrapper for Gentle forced alignment service.

    Expects Gentle to be running at GENTLE_URL (default: http://localhost:8765)
    Run with: docker run -p 8765:8765 lowerquality/gentle
    """

    def __init__(self, gentle_url: str = "http://localhost:8765"):
        """
        Initialize Gentle aligner.

        Args:
            gentle_url: URL where Gentle service is running
        """
        self.gentle_url = gentle_url
        self.align_endpoint = f"{gentle_url}/transcriptions"
        self._check_service()

    def _check_service(self):
        """Check if Gentle service is available."""
        try:
            response = requests.get(self.gentle_url, timeout=2)
            if response.status_code == 200:
                logger.info(f"Gentle service available at {self.gentle_url}")
            else:
                logger.warning(f"Gentle service responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Gentle service not reachable at {self.gentle_url}: {e}")
            logger.warning("Phoneme analysis will be limited without forced alignment")

    def align(
        self,
        audio_path: str,
        transcript: str
    ) -> Optional[Dict]:
        """
        Perform forced alignment on audio with transcript.

        Args:
            audio_path: Path to audio file
            transcript: Text transcript to align

        Returns:
            Alignment result dictionary with phoneme-level timestamps,
            or None if alignment fails
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None

        if not transcript or not transcript.strip():
            logger.error("Empty transcript provided for alignment")
            return None

        try:
            # Prepare multipart form data
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'audio': audio_file,
                    'transcript': (None, transcript)
                }

                # Send request to Gentle
                logger.info(f"Sending alignment request for: {os.path.basename(audio_path)}")
                response = requests.post(
                    self.align_endpoint,
                    files=files,
                    params={'async': 'false'},
                    timeout=60  # Alignment can take time
                )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Alignment successful for {os.path.basename(audio_path)}")
                return result
            else:
                logger.error(f"Gentle returned status {response.status_code}: {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error("Gentle alignment request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Gentle: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during alignment: {e}")
            return None

    def extract_phoneme_timestamps(
        self,
        alignment_result: Dict
    ) -> List[Dict]:
        """
        Extract phoneme-level timestamps from Gentle alignment result.

        Args:
            alignment_result: Raw result from Gentle align()

        Returns:
            List of phoneme dictionaries with:
            - word: str - the word this phoneme belongs to
            - phoneme: str - the phoneme (ARPAbet format)
            - start: float - start time in seconds
            - end: float - end time in seconds
            - word_start: float - start time of parent word
            - word_end: float - end time of parent word
        """
        phoneme_data = []

        if not alignment_result or 'words' not in alignment_result:
            logger.warning("Invalid alignment result structure")
            return phoneme_data

        for word_info in alignment_result['words']:
            # Gentle marks unaligned words with 'case': 'not-found-in-audio'
            if word_info.get('case') == 'not-found-in-audio':
                logger.debug(f"Word '{word_info.get('word', '?')}' not found in audio")
                continue

            # Only process aligned words
            if 'alignedWord' not in word_info or 'start' not in word_info:
                continue

            word = word_info.get('alignedWord', word_info.get('word', ''))
            word_start = word_info.get('start')
            word_end = word_info.get('end')

            # Extract phonemes
            if 'phones' in word_info:
                for phone_info in word_info['phones']:
                    phoneme_data.append({
                        'word': word,
                        'phoneme': phone_info.get('phone', ''),
                        'start': phone_info.get('start', word_start),
                        'duration': phone_info.get('duration', 0),
                        'end': phone_info.get('start', word_start) + phone_info.get('duration', 0),
                        'word_start': word_start,
                        'word_end': word_end
                    })

        logger.info(f"Extracted {len(phoneme_data)} phoneme timestamps")
        return phoneme_data


# Global instance
_gentle_aligner: Optional[GentleAligner] = None


def get_gentle_aligner() -> GentleAligner:
    """
    Get the global GentleAligner instance (singleton).

    Returns:
        GentleAligner instance
    """
    global _gentle_aligner

    if _gentle_aligner is None:
        # Get URL from environment or use default
        gentle_url = os.environ.get('GENTLE_URL', 'http://localhost:8765')
        _gentle_aligner = GentleAligner(gentle_url)

    return _gentle_aligner


def align_audio(audio_path: str, transcript: str) -> Optional[Dict]:
    """
    Convenience function to perform forced alignment.

    Args:
        audio_path: Path to audio file
        transcript: Transcript text

    Returns:
        Alignment result or None
    """
    aligner = get_gentle_aligner()
    return aligner.align(audio_path, transcript)


def get_phoneme_timestamps(audio_path: str, transcript: str) -> List[Dict]:
    """
    Convenience function to get phoneme timestamps from audio.

    Args:
        audio_path: Path to audio file
        transcript: Transcript text

    Returns:
        List of phoneme timestamp dictionaries
    """
    aligner = get_gentle_aligner()
    alignment_result = aligner.align(audio_path, transcript)

    if alignment_result:
        return aligner.extract_phoneme_timestamps(alignment_result)
    else:
        logger.warning("Alignment failed, returning empty phoneme list")
        return []
