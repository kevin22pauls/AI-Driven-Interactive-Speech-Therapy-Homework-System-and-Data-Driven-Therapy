"""
Audio Format Converter

Converts browser-recorded audio (webm/ogg) to WAV format for ML model compatibility.
Uses PyAV (which has bundled FFmpeg) as primary, falls back to pydub if available.

Requirements:
- PyAV: pip install av (includes FFmpeg bindings)
- OR pydub + FFmpeg: pip install pydub + system FFmpeg
"""

import os
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


def convert_to_wav_pyav(input_path: str, output_path: str) -> str:
    """Convert audio using PyAV (bundled FFmpeg bindings)."""
    import av
    import wave

    logger.info(f"Converting {input_path} to WAV using PyAV...")

    # Open input file
    container = av.open(input_path)
    audio_stream = container.streams.audio[0]

    # Resample to 16kHz mono
    resampler = av.audio.resampler.AudioResampler(
        format='s16',
        layout='mono',
        rate=16000
    )

    # Collect all audio samples
    samples = []
    for frame in container.decode(audio_stream):
        resampled_frames = resampler.resample(frame)
        for resampled in resampled_frames:
            arr = resampled.to_ndarray()
            samples.append(arr.flatten())

    container.close()

    # Concatenate all samples
    if samples:
        audio_data = np.concatenate(samples)
    else:
        raise RuntimeError("No audio data found in file")

    # Write to WAV
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(16000)
        wav_file.writeframes(audio_data.tobytes())

    logger.info(f"Converted to: {output_path}")
    return output_path


def convert_to_wav_pydub(input_path: str, output_path: str) -> str:
    """Convert audio using pydub (requires system FFmpeg)."""
    from pydub import AudioSegment

    logger.info(f"Converting {input_path} to WAV using pydub...")

    # Load the audio file (pydub auto-detects format)
    audio = AudioSegment.from_file(input_path)

    # Convert to mono, 16kHz (standard for speech models)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    # Export as WAV
    audio.export(output_path, format="wav")

    logger.info(f"Converted to: {output_path}")
    return output_path


def convert_to_wav(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert audio file to WAV format (16kHz, mono).

    Args:
        input_path: Path to input audio file (webm, ogg, mp3, etc.)
        output_path: Optional output path. If None, replaces extension with .wav

    Returns:
        Path to converted WAV file

    Raises:
        RuntimeError: If conversion fails
    """
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".wav"

    # If already a WAV file, just return it
    if input_path.lower().endswith('.wav'):
        return input_path

    # Try PyAV first (has bundled FFmpeg)
    try:
        return convert_to_wav_pyav(input_path, output_path)
    except ImportError:
        logger.warning("PyAV not available, trying pydub...")
    except Exception as e:
        logger.warning(f"PyAV conversion failed: {e}, trying pydub...")

    # Fall back to pydub
    try:
        return convert_to_wav_pydub(input_path, output_path)
    except ImportError:
        logger.error("Neither PyAV nor pydub is installed")
        raise RuntimeError("No audio conversion library available. Install: pip install av")
    except Exception as e:
        error_msg = str(e)
        if "ffmpeg" in error_msg.lower() or "ffprobe" in error_msg.lower():
            logger.error("pydub requires system FFmpeg which is not installed")
            raise RuntimeError(f"Audio conversion failed - FFmpeg not available: {e}")
        else:
            logger.error(f"Audio conversion failed: {e}")
            raise RuntimeError(f"Audio conversion failed: {e}")


def ensure_wav_format(audio_path: str) -> str:
    """
    Ensure audio file is in WAV format, converting if necessary.

    This is a convenience function that:
    - Returns the path as-is if already WAV
    - Converts and returns new path if not WAV
    - Keeps the original file intact

    Args:
        audio_path: Path to audio file

    Returns:
        Path to WAV file (original or converted)
    """
    if audio_path.lower().endswith('.wav'):
        return audio_path

    return convert_to_wav(audio_path)


def convert_webm_bytes_to_wav(webm_bytes: bytes, output_path: str) -> str:
    """
    Convert webm audio bytes directly to WAV file.

    Args:
        webm_bytes: Raw webm audio data
        output_path: Path for output WAV file

    Returns:
        Path to converted WAV file
    """
    import tempfile

    # Write bytes to temp file
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
        tmp.write(webm_bytes)
        tmp_path = tmp.name

    try:
        # Convert to WAV
        result = convert_to_wav(tmp_path, output_path)
        return result
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
