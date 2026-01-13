# utils/file_utils.py
import os
import logging
from config import AUDIO_STORAGE
from datetime import datetime

logger = logging.getLogger(__name__)


def save_audio_file(file_bytes, filename_prefix="audio", filename=None, convert_to_wav=True):
    """
    Save audio file and optionally convert to WAV format.

    Args:
        file_bytes: Raw audio bytes
        filename_prefix: Prefix for the saved filename
        filename: Original filename (used to detect format)
        convert_to_wav: If True, convert non-WAV formats to WAV for ML compatibility

    Returns:
        Path to saved (and possibly converted) audio file
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    if filename:
        ext = os.path.splitext(filename)[1] or ".wav"
    else:
        ext = ".wav"

    # Save the original file first
    original_path = os.path.join(AUDIO_STORAGE, f"{filename_prefix}_{timestamp}{ext}")

    with open(original_path, "wb") as f:
        f.write(file_bytes)

    # Convert to WAV if needed (for ML model compatibility)
    if convert_to_wav and ext.lower() not in ['.wav']:
        try:
            from utils.audio_converter import convert_to_wav as do_convert

            wav_path = os.path.join(AUDIO_STORAGE, f"{filename_prefix}_{timestamp}.wav")
            converted_path = do_convert(original_path, wav_path)

            # Optionally remove original webm to save space
            # os.remove(original_path)

            logger.info(f"Audio converted: {ext} -> .wav")
            return converted_path

        except Exception as e:
            logger.warning(f"Audio conversion failed ({e}), using original format")
            # Fall back to original file if conversion fails
            return original_path

    return original_path
