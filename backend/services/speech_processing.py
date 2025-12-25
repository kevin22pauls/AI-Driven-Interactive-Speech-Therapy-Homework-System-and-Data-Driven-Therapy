# services/speech_processing.py
from typing import Dict, Optional
from models.whisper_model import get_whisper_model
from services.metrics import compute_wer, compute_speech_rate, compute_pause_ratio
from services.semantic_analysis import evaluate_answer
from services.prompts import get_question_type
from services.phoneme_analysis import analyze_phonemes, format_phoneme_result_for_api
from services.fluency_analysis import analyze_fluency, format_fluency_result_for_api
import logging

logger = logging.getLogger(__name__)

def analyze_speech(audio_path: str, prompt_data: Optional[Dict] = None):
    """
    Analyze speech from audio file including transcription, metrics, and semantic evaluation.

    Args:
        audio_path: Path to the audio file
        prompt_data: Optional dictionary containing:
            - object_name: Name of the object being discussed
            - prompt_text: The question/prompt asked
            - expected_answer: Expected answer for semantic evaluation

    Returns:
        Dictionary containing analysis results including transcript, metrics, and semantic evaluation
    """
    model = get_whisper_model()

    try:
        segments, info = model.transcribe(audio_path, beam_size=5)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise

    segments_list = []
    transcript = ""

    for seg in segments:
        segments_list.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text
        })
        transcript += seg.text

    # Compute basic metrics
    # For WER, use expected_answer if provided, otherwise fall back to prompt_text
    expected_text = ""
    if prompt_data:
        expected_text = prompt_data.get("expected_answer", prompt_data.get("prompt_text", ""))

    wer_score = compute_wer(expected_text, transcript) if expected_text else None
    speech_rate = compute_speech_rate(segments_list)
    pause_ratio = compute_pause_ratio(segments_list)

    result = {
        "transcript": transcript.strip(),
        "wer": wer_score,
        "speech_rate": speech_rate,
        "pause_ratio": pause_ratio,
        "segments": segments_list
    }

    # Add semantic evaluation if prompt data is provided
    if prompt_data and all(key in prompt_data for key in ["object_name", "prompt_text", "expected_answer"]):
        # Determine question type
        question_type = get_question_type(prompt_data["prompt_text"])

        # Perform semantic evaluation
        semantic_results = evaluate_answer(
            expected_answer=prompt_data["expected_answer"],
            transcript=transcript,
            object_name=prompt_data["object_name"],
            question_type=question_type
        )

        result["semantic_evaluation"] = semantic_results

        # Log the evaluation for monitoring
        logger.info(f"Semantic evaluation - Classification: {semantic_results['classification']}, "
                   f"Score: {semantic_results['similarity_score']:.2f}, "
                   f"Object: {prompt_data['object_name']}")

    # Add phoneme-level analysis if expected answer is provided
    if prompt_data and prompt_data.get("expected_answer"):
        try:
            logger.info("Performing phoneme-level analysis")
            phoneme_result = analyze_phonemes(
                expected_text=prompt_data["expected_answer"],
                actual_text=transcript,
                audio_path=audio_path
            )

            # Format for API response
            result["phoneme_analysis"] = format_phoneme_result_for_api(phoneme_result)

            logger.info(f"Phoneme analysis - PER: {phoneme_result.per:.2f}, "
                       f"Errors: {len(phoneme_result.errors)}, "
                       f"Total phonemes: {phoneme_result.total_phonemes}")

        except Exception as e:
            logger.error(f"Phoneme analysis failed: {e}", exc_info=True)
            # Don't fail the whole analysis if phoneme analysis fails
            result["phoneme_analysis"] = {
                "error": "Phoneme analysis unavailable",
                "reason": str(e)
            }

    # Add fluency analysis (always run - doesn't need expected answer)
    try:
        logger.info("Performing fluency analysis")
        fluency_result = analyze_fluency(segments_list)

        # Format for API response
        result["fluency_analysis"] = format_fluency_result_for_api(fluency_result)

        logger.info(f"Fluency analysis - LFR: {fluency_result.longest_fluent_run}, "
                   f"Fluency: {fluency_result.fluency_percentage:.1f}%, "
                   f"Pauses: {fluency_result.total_pauses}, "
                   f"Stuttering events: {len(fluency_result.stuttering_events)}")

    except Exception as e:
        logger.error(f"Fluency analysis failed: {e}", exc_info=True)
        # Don't fail the whole analysis if fluency analysis fails
        result["fluency_analysis"] = {
            "error": "Fluency analysis unavailable",
            "reason": str(e)
        }

    return result
