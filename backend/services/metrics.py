# services/metrics.py
from jiwer import wer

def compute_wer(expected, transcript):
    try:
        return wer(expected, transcript)
    except Exception:
        return None

def _extract_segment_times(segments):
    """
    Accepts segments as list of objects with .start/.end or dicts with 'start'/'end'.
    Returns list of dicts with numeric start/end.
    """
    cleaned = []
    for seg in segments or []:
        if isinstance(seg, dict):
            s = seg.get("start", None)
            e = seg.get("end", None)
        else:
            # whisper segment object with attributes
            s = getattr(seg, "start", None)
            e = getattr(seg, "end", None)
        # ensure numeric
        try:
            s = float(s) if s is not None else None
            e = float(e) if e is not None else None
        except Exception:
            s, e = None, None
        if s is not None and e is not None:
            cleaned.append({"start": s, "end": e, "text": seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")})
    return cleaned

def compute_speech_rate(segments):
    segs = _extract_segment_times(segments)
    if not segs:
        return 0.0
    total_words = 0
    for seg in segs:
        total_words += len((seg.get("text") or "").split())
    total_time = segs[-1]["end"] - segs[0]["start"]
    return total_words / total_time if total_time > 0 else 0.0

def compute_pause_ratio(segments):
    segs = _extract_segment_times(segments)
    if len(segs) < 2:
        return 0.0
    pauses = 0.0
    for i in range(1, len(segs)):
        diff = segs[i]["start"] - segs[i-1]["end"]
        if diff > 0:
            pauses += diff
    total_time = segs[-1]["end"] - segs[0]["start"]
    return pauses / total_time if total_time > 0 else 0.0
