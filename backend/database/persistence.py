# database/persistence.py
"""
Database persistence layer for saving and querying speech therapy recordings.

This module provides functions to:
- Save analysis results to the database
- Track patient progress over time
- Query longitudinal data (phoneme trends, fluency improvements, etc.)
"""

from sqlalchemy import text
from .db import SessionLocal
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def save_recording(
    session_id: str,
    object_name: str,
    prompt_text: str,
    expected_answer: str,
    audio_path: str,
    analysis_result: Dict[str, Any],
    patient_id: Optional[str] = None
) -> int:
    """
    Save a recording and its analysis results to the database.

    Args:
        session_id: Unique session identifier (groups multiple prompts)
        object_name: Name of the object being discussed
        prompt_text: The question/prompt that was asked
        expected_answer: Expected answer for comparison
        audio_path: Path to the saved audio file
        analysis_result: Full analysis result from analyze_speech()
        patient_id: Optional patient identifier for longitudinal tracking

    Returns:
        recording_id: The database ID of the saved recording

    Raises:
        Exception: If database save fails
    """
    db = SessionLocal()
    try:
        # Extract basic metrics
        transcript = analysis_result.get("transcript", "")
        wer = analysis_result.get("wer")
        speech_rate = analysis_result.get("speech_rate")
        pause_ratio = analysis_result.get("pause_ratio")

        # Extract semantic evaluation
        semantic_eval = analysis_result.get("semantic_evaluation", {})
        semantic_classification = semantic_eval.get("classification")
        semantic_score = semantic_eval.get("similarity_score")

        # Extract phoneme analysis
        phoneme_analysis = analysis_result.get("phoneme_analysis", {})
        # Support both old "per" and new "per_rule" field names
        per = phoneme_analysis.get("per_rule") or phoneme_analysis.get("per")
        total_phonemes = phoneme_analysis.get("total_phonemes")

        # Convert phoneme errors to JSON
        phoneme_errors_json = None
        if phoneme_analysis.get("errors"):
            phoneme_errors_json = json.dumps(phoneme_analysis["errors"])

        problematic_phonemes_json = None
        if phoneme_analysis.get("problematic_phonemes"):
            problematic_phonemes_json = json.dumps(phoneme_analysis["problematic_phonemes"])

        clinical_notes_json = None
        if phoneme_analysis.get("clinical_notes"):
            clinical_notes_json = json.dumps(phoneme_analysis["clinical_notes"])

        # Extract ML phoneme analysis
        ml_analysis = phoneme_analysis.get("ml_analysis", {})
        ml_per = ml_analysis.get("per_ml")
        ml_gop = ml_analysis.get("overall_gop")
        ml_confidence = ml_analysis.get("model_confidence")

        ml_detected_phonemes_json = None
        if ml_analysis.get("detected_phonemes"):
            ml_detected_phonemes_json = json.dumps(ml_analysis["detected_phonemes"])

        ml_detected_ipa_json = None
        if ml_analysis.get("detected_phonemes_ipa"):
            ml_detected_ipa_json = json.dumps(ml_analysis["detected_phonemes_ipa"])

        ml_alignment_json = None
        if ml_analysis.get("alignment"):
            ml_alignment_json = json.dumps(ml_analysis["alignment"])

        ml_phoneme_scores_json = None
        if ml_analysis.get("phoneme_scores"):
            ml_phoneme_scores_json = json.dumps(ml_analysis["phoneme_scores"])

        # Extract fluency analysis
        fluency_analysis = analysis_result.get("fluency_analysis", {})
        longest_fluent_run = fluency_analysis.get("longest_fluent_run")
        total_pauses = fluency_analysis.get("total_pauses")
        hesitation_count = fluency_analysis.get("hesitation_count")
        block_count = fluency_analysis.get("block_count")
        fluency_percentage = fluency_analysis.get("fluency_percentage")
        dysfluencies_per_100_words = fluency_analysis.get("dysfluencies_per_100_words")
        dysfluencies_per_minute = fluency_analysis.get("dysfluencies_per_minute")
        speech_rate_variability = fluency_analysis.get("speech_rate_variability")

        # Convert fluency data to JSON
        stuttering_events_json = None
        if fluency_analysis.get("stuttering_events"):
            stuttering_events_json = json.dumps(fluency_analysis["stuttering_events"])

        pauses_json = None
        if fluency_analysis.get("pauses"):
            pauses_json = json.dumps(fluency_analysis["pauses"])

        fluency_notes_json = None
        if fluency_analysis.get("clinical_notes"):
            fluency_notes_json = json.dumps(fluency_analysis["clinical_notes"])

        # Insert recording
        insert_sql = text("""
            INSERT INTO recordings (
                session_id, object_name, prompt_text, expected_answer, audio_path,
                transcript, wer, speech_rate, pause_ratio,
                per, total_phonemes, phoneme_errors_json, problematic_phonemes_json, clinical_notes_json,
                ml_per, ml_gop, ml_confidence, ml_detected_phonemes_json, ml_detected_ipa_json,
                ml_alignment_json, ml_phoneme_scores_json,
                semantic_classification, semantic_score,
                longest_fluent_run, total_pauses, hesitation_count, block_count,
                fluency_percentage, dysfluencies_per_100_words, dysfluencies_per_minute,
                speech_rate_variability, stuttering_events_json, pauses_json, fluency_notes_json,
                created_at
            ) VALUES (
                :session_id, :object_name, :prompt_text, :expected_answer, :audio_path,
                :transcript, :wer, :speech_rate, :pause_ratio,
                :per, :total_phonemes, :phoneme_errors_json, :problematic_phonemes_json, :clinical_notes_json,
                :ml_per, :ml_gop, :ml_confidence, :ml_detected_phonemes_json, :ml_detected_ipa_json,
                :ml_alignment_json, :ml_phoneme_scores_json,
                :semantic_classification, :semantic_score,
                :longest_fluent_run, :total_pauses, :hesitation_count, :block_count,
                :fluency_percentage, :dysfluencies_per_100_words, :dysfluencies_per_minute,
                :speech_rate_variability, :stuttering_events_json, :pauses_json, :fluency_notes_json,
                :created_at
            )
        """)

        result = db.execute(insert_sql, {
            "session_id": session_id,
            "object_name": object_name,
            "prompt_text": prompt_text,
            "expected_answer": expected_answer,
            "audio_path": audio_path,
            "transcript": transcript,
            "wer": wer,
            "speech_rate": speech_rate,
            "pause_ratio": pause_ratio,
            "per": per,
            "total_phonemes": total_phonemes,
            "phoneme_errors_json": phoneme_errors_json,
            "problematic_phonemes_json": problematic_phonemes_json,
            "clinical_notes_json": clinical_notes_json,
            "ml_per": ml_per,
            "ml_gop": ml_gop,
            "ml_confidence": ml_confidence,
            "ml_detected_phonemes_json": ml_detected_phonemes_json,
            "ml_detected_ipa_json": ml_detected_ipa_json,
            "ml_alignment_json": ml_alignment_json,
            "ml_phoneme_scores_json": ml_phoneme_scores_json,
            "semantic_classification": semantic_classification,
            "semantic_score": semantic_score,
            "longest_fluent_run": longest_fluent_run,
            "total_pauses": total_pauses,
            "hesitation_count": hesitation_count,
            "block_count": block_count,
            "fluency_percentage": fluency_percentage,
            "dysfluencies_per_100_words": dysfluencies_per_100_words,
            "dysfluencies_per_minute": dysfluencies_per_minute,
            "speech_rate_variability": speech_rate_variability,
            "stuttering_events_json": stuttering_events_json,
            "pauses_json": pauses_json,
            "fluency_notes_json": fluency_notes_json,
            "created_at": datetime.utcnow()
        })

        db.commit()
        recording_id = result.lastrowid

        # Update patient phoneme history if patient_id is provided
        if patient_id and phoneme_analysis.get("problematic_phonemes"):
            update_patient_phoneme_history(db, patient_id, phoneme_analysis["problematic_phonemes"])

        logger.info(f"Saved recording {recording_id} for session {session_id}")
        return recording_id

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save recording: {e}", exc_info=True)
        raise
    finally:
        db.close()


def update_patient_phoneme_history(db, patient_id: str, problematic_phonemes: Dict[str, int]):
    """
    Update the patient_phoneme_history table with phoneme errors.

    Args:
        db: Database session
        patient_id: Patient identifier
        problematic_phonemes: Dict mapping phoneme -> error count
    """
    for phoneme, error_count in problematic_phonemes.items():
        # Check if record exists
        check_sql = text("""
            SELECT id, error_count, occurrence_count
            FROM patient_phoneme_history
            WHERE patient_id = :patient_id AND phoneme = :phoneme
        """)

        existing = db.execute(check_sql, {
            "patient_id": patient_id,
            "phoneme": phoneme
        }).fetchone()

        if existing:
            # Update existing record
            update_sql = text("""
                UPDATE patient_phoneme_history
                SET error_count = error_count + :error_count,
                    occurrence_count = occurrence_count + 1,
                    last_error_date = :last_error_date
                WHERE patient_id = :patient_id AND phoneme = :phoneme
            """)

            db.execute(update_sql, {
                "patient_id": patient_id,
                "phoneme": phoneme,
                "error_count": error_count,
                "last_error_date": datetime.utcnow()
            })
        else:
            # Insert new record
            insert_sql = text("""
                INSERT INTO patient_phoneme_history
                (patient_id, phoneme, error_count, occurrence_count, last_error_date, created_at)
                VALUES (:patient_id, :phoneme, :error_count, 1, :last_error_date, :created_at)
            """)

            db.execute(insert_sql, {
                "patient_id": patient_id,
                "phoneme": phoneme,
                "error_count": error_count,
                "last_error_date": datetime.utcnow(),
                "created_at": datetime.utcnow()
            })


def get_patient_history(
    patient_id: str,
    limit: int = 50,
    object_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get recording history for a patient.

    Args:
        patient_id: Patient identifier
        limit: Maximum number of recordings to return
        object_name: Optional filter by object name

    Returns:
        List of recording dictionaries with all metrics
    """
    db = SessionLocal()
    try:
        if object_name:
            sql = text("""
                SELECT * FROM recordings
                WHERE session_id LIKE :patient_pattern AND object_name = :object_name
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            result = db.execute(sql, {
                "patient_pattern": f"{patient_id}%",
                "object_name": object_name,
                "limit": limit
            })
        else:
            sql = text("""
                SELECT * FROM recordings
                WHERE session_id LIKE :patient_pattern
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            result = db.execute(sql, {
                "patient_pattern": f"{patient_id}%",
                "limit": limit
            })

        recordings = []
        for row in result:
            recording = dict(row._mapping)

            # Parse JSON fields
            if recording.get("phoneme_errors_json"):
                recording["phoneme_errors"] = json.loads(recording["phoneme_errors_json"])
            if recording.get("problematic_phonemes_json"):
                recording["problematic_phonemes"] = json.loads(recording["problematic_phonemes_json"])
            if recording.get("clinical_notes_json"):
                recording["clinical_notes"] = json.loads(recording["clinical_notes_json"])
            if recording.get("stuttering_events_json"):
                recording["stuttering_events"] = json.loads(recording["stuttering_events_json"])
            if recording.get("pauses_json"):
                recording["pauses"] = json.loads(recording["pauses_json"])
            if recording.get("fluency_notes_json"):
                recording["fluency_notes"] = json.loads(recording["fluency_notes_json"])

            # Parse ML phoneme analysis JSON fields
            if recording.get("ml_detected_phonemes_json"):
                recording["ml_detected_phonemes"] = json.loads(recording["ml_detected_phonemes_json"])
            if recording.get("ml_detected_ipa_json"):
                recording["ml_detected_ipa"] = json.loads(recording["ml_detected_ipa_json"])
            if recording.get("ml_alignment_json"):
                recording["ml_alignment"] = json.loads(recording["ml_alignment_json"])
            if recording.get("ml_phoneme_scores_json"):
                recording["ml_phoneme_scores"] = json.loads(recording["ml_phoneme_scores_json"])

            recordings.append(recording)

        return recordings

    finally:
        db.close()


def get_patient_progress(patient_id: str) -> Dict[str, Any]:
    """
    Get aggregated progress metrics for a patient over time.

    Args:
        patient_id: Patient identifier

    Returns:
        Dictionary containing:
        - total_recordings: Total number of recordings
        - avg_fluency: Average fluency percentage
        - avg_per: Average phoneme error rate
        - avg_wer: Average word error rate
        - lfr_trend: List of (date, LFR) tuples over time
        - most_problematic_phonemes: List of (phoneme, error_count, occurrence_count)
        - improvement_areas: Objects where patient shows improvement
    """
    db = SessionLocal()
    try:
        # Get aggregate metrics
        agg_sql = text("""
            SELECT
                COUNT(*) as total_recordings,
                AVG(fluency_percentage) as avg_fluency,
                AVG(per) as avg_per,
                AVG(wer) as avg_wer,
                AVG(longest_fluent_run) as avg_lfr
            FROM recordings
            WHERE session_id LIKE :patient_pattern
        """)

        agg_result = db.execute(agg_sql, {
            "patient_pattern": f"{patient_id}%"
        }).fetchone()

        # Get LFR trend over time
        lfr_sql = text("""
            SELECT
                DATE(created_at) as date,
                AVG(longest_fluent_run) as avg_lfr,
                AVG(fluency_percentage) as avg_fluency
            FROM recordings
            WHERE session_id LIKE :patient_pattern
            GROUP BY DATE(created_at)
            ORDER BY date ASC
        """)

        lfr_trend = []
        for row in db.execute(lfr_sql, {"patient_pattern": f"{patient_id}%"}):
            lfr_trend.append({
                "date": row.date,
                "avg_lfr": float(row.avg_lfr) if row.avg_lfr else 0,
                "avg_fluency": float(row.avg_fluency) if row.avg_fluency else 0
            })

        # Get most problematic phonemes
        phoneme_sql = text("""
            SELECT phoneme, error_count, occurrence_count,
                   CAST(error_count AS FLOAT) / occurrence_count as error_rate
            FROM patient_phoneme_history
            WHERE patient_id = :patient_id
            ORDER BY error_rate DESC, error_count DESC
            LIMIT 10
        """)

        problematic_phonemes = []
        for row in db.execute(phoneme_sql, {"patient_id": patient_id}):
            problematic_phonemes.append({
                "phoneme": row.phoneme,
                "error_count": row.error_count,
                "occurrence_count": row.occurrence_count,
                "error_rate": float(row.error_rate)
            })

        # Get per-object performance
        object_sql = text("""
            SELECT
                object_name,
                COUNT(*) as attempt_count,
                AVG(semantic_score) as avg_semantic_score,
                AVG(per) as avg_per,
                AVG(fluency_percentage) as avg_fluency
            FROM recordings
            WHERE session_id LIKE :patient_pattern
            GROUP BY object_name
            ORDER BY avg_semantic_score DESC
        """)

        object_performance = []
        for row in db.execute(object_sql, {"patient_pattern": f"{patient_id}%"}):
            object_performance.append({
                "object_name": row.object_name,
                "attempt_count": row.attempt_count,
                "avg_semantic_score": float(row.avg_semantic_score) if row.avg_semantic_score else 0,
                "avg_per": float(row.avg_per) if row.avg_per else 0,
                "avg_fluency": float(row.avg_fluency) if row.avg_fluency else 0
            })

        return {
            "total_recordings": agg_result.total_recordings or 0,
            "avg_fluency": float(agg_result.avg_fluency) if agg_result.avg_fluency else 0,
            "avg_per": float(agg_result.avg_per) if agg_result.avg_per else 0,
            "avg_wer": float(agg_result.avg_wer) if agg_result.avg_wer else 0,
            "avg_lfr": float(agg_result.avg_lfr) if agg_result.avg_lfr else 0,
            "lfr_trend": lfr_trend,
            "most_problematic_phonemes": problematic_phonemes,
            "object_performance": object_performance
        }

    finally:
        db.close()


def get_session_summary(session_id: str) -> Dict[str, Any]:
    """
    Get summary of all recordings in a session.

    Args:
        session_id: Session identifier

    Returns:
        Dictionary containing:
        - session_id: The session ID
        - total_prompts: Number of prompts completed
        - object_name: Object practiced in this session
        - recordings: List of all recordings with metrics
        - session_averages: Average metrics across session
    """
    db = SessionLocal()
    try:
        # Get all recordings for this session
        sql = text("""
            SELECT * FROM recordings
            WHERE session_id = :session_id
            ORDER BY created_at ASC
        """)

        recordings = []
        total_fluency = 0
        total_per = 0
        total_lfr = 0
        count = 0
        object_name = None

        for row in db.execute(sql, {"session_id": session_id}):
            recording = dict(row._mapping)

            # Parse JSON fields
            if recording.get("phoneme_errors_json"):
                recording["phoneme_errors"] = json.loads(recording["phoneme_errors_json"])
            if recording.get("problematic_phonemes_json"):
                recording["problematic_phonemes"] = json.loads(recording["problematic_phonemes_json"])
            if recording.get("stuttering_events_json"):
                recording["stuttering_events"] = json.loads(recording["stuttering_events_json"])
            if recording.get("fluency_notes_json"):
                recording["fluency_notes"] = json.loads(recording["fluency_notes_json"])

            recordings.append(recording)

            # Accumulate for averages
            if recording.get("fluency_percentage"):
                total_fluency += recording["fluency_percentage"]
            if recording.get("per"):
                total_per += recording["per"]
            if recording.get("longest_fluent_run"):
                total_lfr += recording["longest_fluent_run"]

            count += 1
            if not object_name:
                object_name = recording.get("object_name")

        return {
            "session_id": session_id,
            "total_prompts": count,
            "object_name": object_name,
            "recordings": recordings,
            "session_averages": {
                "avg_fluency": total_fluency / count if count > 0 else 0,
                "avg_per": total_per / count if count > 0 else 0,
                "avg_lfr": total_lfr / count if count > 0 else 0
            }
        }

    finally:
        db.close()


def get_phoneme_trends(patient_id: str, phoneme: str) -> List[Dict[str, Any]]:
    """
    Get trend data for a specific phoneme across all recordings.

    Args:
        patient_id: Patient identifier
        phoneme: Phoneme to track (e.g., 'AA', 'TH')

    Returns:
        List of dictionaries with date and error information
    """
    db = SessionLocal()
    try:
        # This requires parsing JSON in each row - not ideal but works
        sql = text("""
            SELECT
                created_at,
                object_name,
                problematic_phonemes_json
            FROM recordings
            WHERE session_id LIKE :patient_pattern
            AND problematic_phonemes_json IS NOT NULL
            ORDER BY created_at ASC
        """)

        trend_data = []
        for row in db.execute(sql, {"patient_pattern": f"{patient_id}%"}):
            problematic = json.loads(row.problematic_phonemes_json)
            if phoneme in problematic:
                trend_data.append({
                    "date": row.created_at,
                    "object_name": row.object_name,
                    "error_count": problematic[phoneme]
                })

        return trend_data

    finally:
        db.close()


def save_generated_prompts(
    object_name: str,
    prompts_data: Dict,
    model_name: str = None,
    generation_method: str = "llm"
) -> bool:
    """
    Save LLM-generated prompts to cache for reuse.

    Args:
        object_name: Name of the object
        prompts_data: Dictionary with 'questions' and 'sentences'
        model_name: Name of the LLM model used (e.g., "llama3.2:3b")
        generation_method: "llm" or "fallback"

    Returns:
        True if saved successfully, False otherwise
    """
    db = SessionLocal()
    try:
        prompts_json = json.dumps(prompts_data)

        # Upsert: insert or update if exists
        sql = text("""
            INSERT INTO generated_prompts (object_name, prompts_json, model_name, generation_method, last_used_at)
            VALUES (:object_name, :prompts_json, :model_name, :generation_method, CURRENT_TIMESTAMP)
            ON CONFLICT(object_name) DO UPDATE SET
                prompts_json = :prompts_json,
                model_name = :model_name,
                generation_method = :generation_method,
                last_used_at = CURRENT_TIMESTAMP
        """)

        db.execute(sql, {
            "object_name": object_name.lower().strip(),
            "prompts_json": prompts_json,
            "model_name": model_name,
            "generation_method": generation_method
        })
        db.commit()

        logger.info(f"Cached generated prompts for '{object_name}'")
        return True

    except Exception as e:
        logger.error(f"Failed to save generated prompts: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def get_cached_prompts(object_name: str) -> Optional[Dict]:
    """
    Retrieve cached prompts for an object.

    Args:
        object_name: Name of the object

    Returns:
        Dictionary with 'questions' and 'sentences', or None if not found
    """
    db = SessionLocal()
    try:
        sql = text("""
            SELECT prompts_json, model_name, generation_method
            FROM generated_prompts
            WHERE object_name = :object_name
        """)

        result = db.execute(sql, {"object_name": object_name.lower().strip()}).fetchone()

        if result:
            # Update last_used_at
            update_sql = text("""
                UPDATE generated_prompts
                SET last_used_at = CURRENT_TIMESTAMP
                WHERE object_name = :object_name
            """)
            db.execute(update_sql, {"object_name": object_name.lower().strip()})
            db.commit()

            prompts_data = json.loads(result.prompts_json)
            logger.info(f"Retrieved cached prompts for '{object_name}' (method: {result.generation_method})")
            return prompts_data

        return None

    except Exception as e:
        logger.error(f"Failed to retrieve cached prompts: {e}")
        return None
    finally:
        db.close()


def list_generated_objects() -> List[str]:
    """
    Get list of all objects with generated prompts.

    Returns:
        List of object names
    """
    db = SessionLocal()
    try:
        sql = text("""
            SELECT object_name
            FROM generated_prompts
            ORDER BY last_used_at DESC
        """)

        results = db.execute(sql).fetchall()
        return [row.object_name for row in results]

    except Exception as e:
        logger.error(f"Failed to list generated objects: {e}")
        return []
    finally:
        db.close()


def get_recording_by_id(recording_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a single recording with full analysis details.

    Args:
        recording_id: The database ID of the recording

    Returns:
        Dictionary with all recording data and parsed JSON fields, or None if not found
    """
    db = SessionLocal()
    try:
        sql = text("SELECT * FROM recordings WHERE id = :id")
        row = db.execute(sql, {"id": recording_id}).fetchone()

        if not row:
            return None

        recording = dict(row._mapping)

        # Parse all JSON fields
        json_fields = [
            'phoneme_errors_json', 'problematic_phonemes_json', 'clinical_notes_json',
            'stuttering_events_json', 'pauses_json', 'fluency_notes_json',
            'ml_detected_phonemes_json', 'ml_detected_ipa_json',
            'ml_alignment_json', 'ml_phoneme_scores_json'
        ]

        for field in json_fields:
            if recording.get(field):
                # Remove '_json' suffix to get clean field name
                clean_name = field.replace('_json', '')
                recording[clean_name] = json.loads(recording[field])

        return recording

    except Exception as e:
        logger.error(f"Failed to get recording by ID {recording_id}: {e}", exc_info=True)
        return None
    finally:
        db.close()
