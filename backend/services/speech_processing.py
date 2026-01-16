# services/speech_processing.py
"""
Speech Processing Pipeline with Enhanced Analysis

This module orchestrates the complete speech analysis pipeline including:
- ASR transcription (Whisper) with word-level timestamps
- ML-based VAD for precise pause detection (Silero VAD)
- Basic metrics (WER, speech rate, pause ratio)
- Enhanced phoneme analysis (WPER, conduite d'approche, GOP scores)
- Enhanced semantic analysis (circumlocution, paraphasia detection)
- Enhanced fluency analysis (SSI-4, adaptive LFR, rate variability)
"""

import os
# Set espeak-ng library path for phonemizer (must be before importing phonemizer)
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = r'C:\Program Files\eSpeak NG\libespeak-ng.dll'
os.environ['PHONEMIZER_ESPEAK_PATH'] = r'C:\Program Files\eSpeak NG'

from typing import Dict, Optional, List
from models.whisper_model import get_whisper_model
from services.metrics import compute_wer, compute_speech_rate, compute_pause_ratio
from services.prompts import get_question_type
import logging

# Import both standard and enhanced analysis modules
from services.semantic_analysis import evaluate_answer
from services.phoneme_analysis import analyze_phonemes, format_phoneme_result_for_api
from services.fluency_analysis import analyze_fluency, format_fluency_result_for_api

# Import enhanced modules
from services.enhanced_phoneme_analysis import (
    analyze_phonemes_enhanced,
    format_enhanced_phoneme_result_for_api
)
from services.enhanced_semantic_analysis import (
    evaluate_answer_enhanced,
    format_enhanced_semantic_result_for_api
)
from services.enhanced_fluency_analysis import (
    analyze_fluency_enhanced,
    format_enhanced_fluency_result_for_api
)

# Import ML modules
from services.ml_vad import get_vad_analyzer
from services.ml_stutter import detect_stuttering_ml, format_ml_stutter_events_for_api
from services.ml_phoneme import analyze_phonemes_ml, format_ml_phoneme_result_for_api

logger = logging.getLogger(__name__)

# Flags to control ML features
USE_ML_VAD = True
USE_ML_WORD_TIMESTAMPS = True
USE_ML_STUTTER_DETECTION = True
USE_ML_PHONEME_ANALYSIS = True


def analyze_speech(
    audio_path: str,
    prompt_data: Optional[Dict] = None,
    use_enhanced: bool = True,
    disorder_type: str = 'aphasia'
):
    """
    Analyze speech from audio file including transcription, metrics, and semantic evaluation.

    Args:
        audio_path: Path to the audio file
        prompt_data: Optional dictionary containing:
            - object_name: Name of the object being discussed
            - prompt_text: The question/prompt asked
            - expected_answer: Expected answer for semantic evaluation
            - expected_answers: Optional list of multiple acceptable answers
        use_enhanced: Whether to use enhanced analysis algorithms (default: True)
        disorder_type: Type of disorder for threshold adjustment
                      ('normal', 'aphasia', 'apraxia', 'stuttering', 'dysarthria')

    Returns:
        Dictionary containing analysis results including transcript, metrics, and semantic evaluation
    """
    model = get_whisper_model()

    try:
        # Enable word-level timestamps for more accurate timing
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            language="en",  # Force English-only transcription
            vad_filter=True,  # Enable voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500),  # Filter out silence
            word_timestamps=USE_ML_WORD_TIMESTAMPS  # Enable word-level timestamps
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise

    segments_list = []
    word_level_timings = []  # Store word-level timing from Whisper
    transcript = ""

    for seg in segments:
        seg_dict = {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text
        }

        # Extract word-level timestamps if available
        if USE_ML_WORD_TIMESTAMPS and hasattr(seg, 'words') and seg.words:
            seg_dict["words"] = []
            for word in seg.words:
                word_info = {
                    'text': word.word.strip() if hasattr(word, 'word') else str(word),
                    'start': float(word.start) if hasattr(word, 'start') else 0,
                    'end': float(word.end) if hasattr(word, 'end') else 0,
                    'probability': float(word.probability) if hasattr(word, 'probability') else 1.0
                }
                seg_dict["words"].append(word_info)
                word_level_timings.append(word_info)

        segments_list.append(seg_dict)
        transcript += seg.text

    # Compute basic metrics
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
        "segments": segments_list,
        "analysis_mode": "enhanced" if use_enhanced else "standard"
    }

    # Extract word timings for enhanced analysis
    # Use word-level timestamps from Whisper if available, otherwise fall back to approximation
    if word_level_timings:
        word_timings = word_level_timings
        logger.info(f"Using {len(word_timings)} word-level timestamps from Whisper")
    else:
        word_timings = _extract_word_timings(segments_list)
        logger.info(f"Using approximated word timings ({len(word_timings)} words)")

    # Run ML VAD analysis if enabled
    vad_analysis = None
    if USE_ML_VAD:
        try:
            vad_analyzer = get_vad_analyzer()
            vad_analysis = vad_analyzer.analyze_audio(audio_path)
            logger.info(f"ML VAD: {vad_analysis['statistics']['num_speech_segments']} speech segments, "
                       f"{vad_analysis['statistics']['num_pauses']} pauses detected")
        except Exception as e:
            logger.warning(f"ML VAD analysis failed, continuing without: {e}")

    # Semantic evaluation
    if prompt_data and all(key in prompt_data for key in ["object_name", "prompt_text", "expected_answer"]):
        question_type = get_question_type(prompt_data["prompt_text"])

        if use_enhanced:
            try:
                logger.info("Performing enhanced semantic analysis")
                semantic_result = evaluate_answer_enhanced(
                    expected_answer=prompt_data["expected_answer"],
                    transcript=transcript,
                    object_name=prompt_data["object_name"],
                    question_type=question_type,
                    expected_answers=prompt_data.get("expected_answers")
                )
                result["semantic_evaluation"] = format_enhanced_semantic_result_for_api(semantic_result)

                logger.info(f"Enhanced semantic - Classification: {semantic_result.classification}, "
                           f"Score: {semantic_result.similarity_score:.2f}")

            except Exception as e:
                logger.error(f"Enhanced semantic analysis failed, falling back to standard: {e}")
                # Fall back to standard
                semantic_results = evaluate_answer(
                    expected_answer=prompt_data["expected_answer"],
                    transcript=transcript,
                    object_name=prompt_data["object_name"],
                    question_type=question_type
                )
                result["semantic_evaluation"] = semantic_results
        else:
            semantic_results = evaluate_answer(
                expected_answer=prompt_data["expected_answer"],
                transcript=transcript,
                object_name=prompt_data["object_name"],
                question_type=question_type
            )
            result["semantic_evaluation"] = semantic_results

            logger.info(f"Standard semantic - Classification: {semantic_results['classification']}, "
                       f"Score: {semantic_results['similarity_score']:.2f}")

    # Phoneme-level analysis
    if prompt_data and prompt_data.get("expected_answer"):
        if use_enhanced:
            try:
                # Step 1: Run ML phoneme analysis first (acoustic ground truth)
                ml_phoneme_result = None
                ml_detected_phonemes = None

                if USE_ML_PHONEME_ANALYSIS:
                    try:
                        logger.info("Running ML phoneme analysis for acoustic ground truth")
                        ml_phoneme_result = analyze_phonemes_ml(
                            audio_path=audio_path,
                            expected_text=prompt_data["expected_answer"]
                        )
                        if ml_phoneme_result and ml_phoneme_result.detected_phonemes:
                            ml_detected_phonemes = ml_phoneme_result.detected_phonemes
                            logger.info(f"ML phoneme - detected {len(ml_detected_phonemes)} phonemes, "
                                       f"GOP: {ml_phoneme_result.overall_gop:.2f}, "
                                       f"PER_ML: {ml_phoneme_result.per_ml:.2f}")
                    except Exception as ml_e:
                        logger.warning(f"ML phoneme analysis failed (non-critical): {ml_e}")

                # Step 2: Run enhanced phoneme analysis using ML-detected phonemes
                logger.info("Performing enhanced phoneme analysis")
                phoneme_result = analyze_phonemes_enhanced(
                    expected_text=prompt_data["expected_answer"],
                    actual_text=transcript,
                    word_timings=word_timings,
                    ml_detected_phonemes=ml_detected_phonemes  # Use acoustic ground truth
                )
                result["phoneme_analysis"] = format_enhanced_phoneme_result_for_api(phoneme_result)

                logger.info(f"Enhanced phoneme - PER: {phoneme_result.per_rule:.2f}, "
                           f"WPER: {phoneme_result.wper:.2f}, "
                           f"Errors: {len(phoneme_result.errors)}")

                # Add conduite d'approche flag if detected
                if phoneme_result.multi_attempt_result and phoneme_result.multi_attempt_result.conduite_d_approche:
                    logger.info("Conduite d'approche pattern detected!")

                # Step 3: Add ML analysis results to output
                if ml_phoneme_result:
                    result["phoneme_analysis"]["ml_analysis"] = format_ml_phoneme_result_for_api(ml_phoneme_result)
                    result["phoneme_analysis"]["overall_gop"] = round(ml_phoneme_result.overall_gop, 3)
                    result["phoneme_analysis"]["per_ml"] = round(ml_phoneme_result.per_ml, 3)

            except Exception as e:
                logger.error(f"Enhanced phoneme analysis failed, falling back to standard: {e}", exc_info=True)
                try:
                    phoneme_result = analyze_phonemes(
                        expected_text=prompt_data["expected_answer"],
                        actual_text=transcript,
                        audio_path=audio_path
                    )
                    result["phoneme_analysis"] = format_phoneme_result_for_api(phoneme_result)
                except Exception as e2:
                    logger.error(f"Standard phoneme analysis also failed: {e2}")
                    result["phoneme_analysis"] = {
                        "error": "Phoneme analysis unavailable",
                        "reason": str(e)
                    }
        else:
            try:
                logger.info("Performing standard phoneme analysis")
                phoneme_result = analyze_phonemes(
                    expected_text=prompt_data["expected_answer"],
                    actual_text=transcript,
                    audio_path=audio_path
                )
                result["phoneme_analysis"] = format_phoneme_result_for_api(phoneme_result)

                logger.info(f"Standard phoneme - PER: {phoneme_result.per:.2f}, "
                           f"Errors: {len(phoneme_result.errors)}")

            except Exception as e:
                logger.error(f"Phoneme analysis failed: {e}", exc_info=True)
                result["phoneme_analysis"] = {
                    "error": "Phoneme analysis unavailable",
                    "reason": str(e)
                }

    # Fluency analysis
    if use_enhanced:
        try:
            logger.info(f"Performing enhanced fluency analysis (disorder_type={disorder_type})")
            fluency_result = analyze_fluency_enhanced(
                segments_list,
                disorder_type=disorder_type,
                audio_path=audio_path,
                vad_data=vad_analysis
            )
            result["fluency_analysis"] = format_enhanced_fluency_result_for_api(fluency_result)

            # Add VAD statistics if available
            if vad_analysis:
                result["fluency_analysis"]["vad_statistics"] = vad_analysis.get("statistics", {})
                result["fluency_analysis"]["analysis_method"] = "ml"

            # Run ML stuttering detection for additional confidence and acoustic-based detection
            if USE_ML_STUTTER_DETECTION and word_timings:
                try:
                    ml_stutter_events, stutter_summary = detect_stuttering_ml(
                        audio_path=audio_path,
                        word_timings=word_timings,
                        vad_data=vad_analysis
                    )
                    if ml_stutter_events:
                        result["fluency_analysis"]["ml_stuttering"] = {
                            "events": format_ml_stutter_events_for_api(ml_stutter_events),
                            "summary": stutter_summary,
                            "analysis_method": "ml"
                        }
                        logger.info(f"ML Stutter Detection: {len(ml_stutter_events)} events, "
                                   f"severity: {stutter_summary.get('clinical_severity', 'normal')}")
                except Exception as ml_e:
                    logger.warning(f"ML stuttering detection failed (non-critical): {ml_e}")

            logger.info(f"Enhanced fluency - LFR: {fluency_result.longest_fluent_run}, "
                       f"LFR (tolerant): {fluency_result.lfr_with_tolerance}, "
                       f"Weighted Fluency: {fluency_result.fluency_scores.weighted_fluency_pct:.1f}%, "
                       f"SSI: {fluency_result.ssi_approximation.severity}")

        except Exception as e:
            logger.error(f"Enhanced fluency analysis failed, falling back to standard: {e}", exc_info=True)
            try:
                fluency_result = analyze_fluency(segments_list)
                result["fluency_analysis"] = format_fluency_result_for_api(fluency_result)
            except Exception as e2:
                logger.error(f"Standard fluency analysis also failed: {e2}")
                result["fluency_analysis"] = {
                    "error": "Fluency analysis unavailable",
                    "reason": str(e)
                }
    else:
        try:
            logger.info("Performing standard fluency analysis")
            fluency_result = analyze_fluency(segments_list)
            result["fluency_analysis"] = format_fluency_result_for_api(fluency_result)

            logger.info(f"Standard fluency - LFR: {fluency_result.longest_fluent_run}, "
                       f"Fluency: {fluency_result.fluency_percentage:.1f}%")

        except Exception as e:
            logger.error(f"Fluency analysis failed: {e}", exc_info=True)
            result["fluency_analysis"] = {
                "error": "Fluency analysis unavailable",
                "reason": str(e)
            }

    # Add summary metrics for easy access
    result["summary"] = _generate_summary(result, use_enhanced)

    return result


def _extract_word_timings(segments: List[Dict]) -> List[Dict]:
    """
    Extract word-level timings from segments.

    Args:
        segments: List of segment dictionaries

    Returns:
        List of word timing dictionaries
    """
    words = []

    for seg in segments:
        text = seg.get('text', '').strip()
        if not text:
            continue

        segment_words = text.split()
        if not segment_words:
            continue

        seg_start = seg['start']
        seg_end = seg['end']
        seg_duration = seg_end - seg_start

        time_per_word = seg_duration / len(segment_words) if segment_words else 0

        for i, word in enumerate(segment_words):
            word_start = seg_start + (i * time_per_word)
            word_end = word_start + time_per_word

            words.append({
                'text': word.lower().strip('.,!?;:"\''),
                'start': word_start,
                'end': word_end
            })

    return words


def _generate_summary(result: Dict, use_enhanced: bool) -> Dict:
    """
    Generate a summary of key metrics.

    Args:
        result: Full analysis result
        use_enhanced: Whether enhanced analysis was used

    Returns:
        Summary dictionary
    """
    summary = {
        "transcript": result.get("transcript", ""),
        "analysis_mode": result.get("analysis_mode", "standard")
    }

    # Semantic summary
    semantic = result.get("semantic_evaluation", {})
    summary["semantic_classification"] = semantic.get("classification", "unknown")
    summary["semantic_score"] = semantic.get("similarity_score", 0)

    # Check for special detections in enhanced mode
    if use_enhanced and semantic:
        circum = semantic.get("circumlocution_analysis")
        if circum and isinstance(circum, dict) and circum.get("is_circumlocution"):
            summary["circumlocution_detected"] = True
        if semantic.get("semantic_paraphasia"):
            summary["semantic_paraphasia_detected"] = True

    # Phoneme summary
    phoneme = result.get("phoneme_analysis", {})
    if phoneme and "error" not in phoneme:
        # Use per_rule if available, fall back to per for backwards compatibility
        summary["per_rule"] = phoneme.get("per_rule") or phoneme.get("per", 0)
        if use_enhanced:
            summary["wper"] = phoneme.get("wper", 0)
            # Add ML phoneme metrics if available
            if phoneme.get("per_ml") is not None:
                summary["per_ml"] = phoneme.get("per_ml", 0)
            if phoneme.get("overall_gop") is not None:
                summary["gop_score"] = phoneme.get("overall_gop", 0)
            multi_attempt = phoneme.get("multi_attempt_analysis")
            if multi_attempt and isinstance(multi_attempt, dict) and multi_attempt.get("conduite_d_approche"):
                summary["conduite_d_approche_detected"] = True

    # Fluency summary
    fluency = result.get("fluency_analysis", {})
    if "error" not in fluency:
        summary["longest_fluent_run"] = fluency.get("longest_fluent_run", 0)

        if use_enhanced:
            summary["lfr_with_tolerance"] = fluency.get("lfr_with_tolerance", 0)
            fluency_scores = fluency.get("fluency_scores", {})
            summary["weighted_fluency_pct"] = fluency_scores.get("weighted_fluency_pct", 100)
            summary["clinical_fluency_pct"] = fluency_scores.get("clinical_fluency_pct", 100)

            ssi = fluency.get("ssi_approximation", {})
            summary["ssi_severity"] = ssi.get("severity", "normal")
            summary["ssi_score"] = ssi.get("total_score", 0)

            rate_metrics = fluency.get("speech_rate_metrics", {})
            summary["articulation_rate"] = rate_metrics.get("articulation_rate", 0)
            summary["speaking_rate"] = rate_metrics.get("speaking_rate", 0)

            # Add ML stuttering summary if available
            ml_stuttering = fluency.get("ml_stuttering", {})
            if ml_stuttering:
                stutter_summary = ml_stuttering.get("summary", {})
                summary["ml_stutter_severity"] = stutter_summary.get("clinical_severity", "normal")
                summary["ml_stutter_count"] = stutter_summary.get("total_events", 0)
                summary["ml_stutter_confidence"] = stutter_summary.get("avg_confidence", 0)

            # Add VAD statistics if available
            vad_stats = fluency.get("vad_statistics", {})
            if vad_stats:
                summary["speech_ratio"] = vad_stats.get("speech_ratio", 0)
                summary["pause_count_vad"] = vad_stats.get("num_pauses", 0)
        else:
            summary["fluency_percentage"] = fluency.get("fluency_percentage", 100)

    return summary


# Legacy function for backward compatibility
def analyze_speech_standard(audio_path: str, prompt_data: Optional[Dict] = None):
    """
    Analyze speech using standard (non-enhanced) algorithms.

    This is provided for backward compatibility and comparison.
    """
    return analyze_speech(audio_path, prompt_data, use_enhanced=False)
