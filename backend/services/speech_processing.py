# services/speech_processing.py
from models.whisper_model import get_whisper_model
from services.metrics import compute_wer, compute_speech_rate, compute_pause_ratio

def analyze_speech(audio_path, expected_text=""):
    model = get_whisper_model()

    segments, info = model.transcribe(audio_path, beam_size=5)

    segments_list = []
    transcript = ""

    for seg in segments:
        segments_list.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text
        })
        transcript += seg.text

    wer_score = compute_wer(expected_text, transcript) if expected_text else None
    speech_rate = compute_speech_rate(segments_list)
    pause_ratio = compute_pause_ratio(segments_list)

    return {
        "transcript": transcript,
        "wer": wer_score,
        "speech_rate": speech_rate,
        "pause_ratio": pause_ratio,
        "segments": segments_list
    }
