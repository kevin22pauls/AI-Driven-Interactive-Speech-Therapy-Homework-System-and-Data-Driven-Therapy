"""
Unit Tests for Database Persistence Module

These tests verify that recordings are correctly saved to the database
and that longitudinal queries work properly for patient tracking.
"""

import pytest
import sys
import os
import tempfile
import json
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.persistence import (
    save_recording,
    get_patient_history,
    get_patient_progress,
    get_session_summary,
    get_phoneme_trends
)
from database.db import SessionLocal, engine
from database.schema import init_db
from sqlalchemy import text


@pytest.fixture(scope="function")
def test_db():
    """Create a fresh test database for each test"""
    # Initialize tables
    init_db()

    yield

    # Clean up after test
    db = SessionLocal()
    try:
        # Delete all test data
        db.execute(text("DELETE FROM recordings"))
        db.execute(text("DELETE FROM patient_phoneme_history"))
        db.commit()
    finally:
        db.close()


class TestSaveRecording:
    """Tests for saving recordings to database"""

    def test_save_basic_recording(self, test_db):
        """Test saving a recording with basic metrics"""
        analysis_result = {
            "transcript": "I want water",
            "wer": 0.0,
            "speech_rate": 3.5,
            "pause_ratio": 0.1,
            "semantic_evaluation": {
                "classification": "correct",
                "similarity_score": 0.95
            }
        }

        recording_id = save_recording(
            session_id="test_session_001",
            object_name="water",
            prompt_text="What do you want to drink?",
            expected_answer="I want water",
            audio_path="/tmp/test.wav",
            analysis_result=analysis_result
        )

        assert recording_id > 0

        # Verify it was saved
        db = SessionLocal()
        try:
            result = db.execute(text("SELECT * FROM recordings WHERE id = :id"), {"id": recording_id})
            row = result.fetchone()

            assert row is not None
            assert row.session_id == "test_session_001"
            assert row.object_name == "water"
            assert row.transcript == "I want water"
            assert row.wer == 0.0
            assert row.semantic_classification == "correct"
        finally:
            db.close()

    def test_save_recording_with_phoneme_analysis(self, test_db):
        """Test saving recording with phoneme analysis data"""
        analysis_result = {
            "transcript": "I want bootle",
            "wer": 0.33,
            "speech_rate": 3.0,
            "pause_ratio": 0.15,
            "phoneme_analysis": {
                "per_rule": 0.125,
                "total_phonemes": 8,
                "errors": [
                    {
                        "error_type": "substitution",
                        "position": 5,
                        "expected_phoneme": "AA",
                        "actual_phoneme": "UW",
                        "word": "bottle"
                    }
                ],
                "problematic_phonemes": {
                    "AA": 1
                },
                "clinical_notes": ["Patient substituted AA with UW in 'bottle'"]
            }
        }

        recording_id = save_recording(
            session_id="test_session_002",
            object_name="bottle",
            prompt_text="What is this?",
            expected_answer="bottle",
            audio_path="/tmp/test2.wav",
            analysis_result=analysis_result
        )

        assert recording_id > 0

        # Verify phoneme data was saved
        db = SessionLocal()
        try:
            result = db.execute(text("SELECT * FROM recordings WHERE id = :id"), {"id": recording_id})
            row = result.fetchone()

            assert row.per == 0.125
            assert row.total_phonemes == 8
            assert row.phoneme_errors_json is not None

            errors = json.loads(row.phoneme_errors_json)
            assert len(errors) == 1
            assert errors[0]["error_type"] == "substitution"
            assert errors[0]["expected_phoneme"] == "AA"

            problematic = json.loads(row.problematic_phonemes_json)
            assert problematic["AA"] == 1
        finally:
            db.close()

    def test_save_recording_with_fluency_analysis(self, test_db):
        """Test saving recording with fluency analysis data"""
        analysis_result = {
            "transcript": "I I I want um water",
            "wer": 0.2,
            "speech_rate": 2.5,
            "pause_ratio": 0.3,
            "fluency_analysis": {
                "longest_fluent_run": 2,
                "total_pauses": 2,
                "hesitation_count": 1,
                "block_count": 1,
                "fluency_percentage": 40.0,
                "dysfluencies_per_100_words": 80.0,
                "dysfluencies_per_minute": 20.0,
                "speech_rate_variability": 0.35,
                "stuttering_events": [
                    {
                        "event_type": "repetition",
                        "position": 0,
                        "word": "I",
                        "repetition_count": 3
                    },
                    {
                        "event_type": "interjection",
                        "position": 3,
                        "word": "um"
                    }
                ],
                "pauses": [
                    {
                        "position": 1,
                        "duration": 0.5,
                        "pause_type": "hesitation"
                    }
                ],
                "clinical_notes": ["Patient shows word repetition on pronoun 'I'"]
            }
        }

        recording_id = save_recording(
            session_id="test_session_003",
            object_name="water",
            prompt_text="What do you want?",
            expected_answer="I want water",
            audio_path="/tmp/test3.wav",
            analysis_result=analysis_result
        )

        assert recording_id > 0

        # Verify fluency data was saved
        db = SessionLocal()
        try:
            result = db.execute(text("SELECT * FROM recordings WHERE id = :id"), {"id": recording_id})
            row = result.fetchone()

            assert row.longest_fluent_run == 2
            assert row.total_pauses == 2
            assert row.hesitation_count == 1
            assert row.block_count == 1
            assert row.fluency_percentage == 40.0
            assert row.dysfluencies_per_100_words == 80.0

            events = json.loads(row.stuttering_events_json)
            assert len(events) == 2
            assert events[0]["event_type"] == "repetition"
            assert events[1]["event_type"] == "interjection"
        finally:
            db.close()

    def test_save_recording_with_patient_id(self, test_db):
        """Test that patient phoneme history is updated when patient_id is provided"""
        analysis_result = {
            "transcript": "bottle",
            "phoneme_analysis": {
                "per_rule": 0.1,
                "total_phonemes": 10,
                "problematic_phonemes": {
                    "AA": 2,
                    "TH": 1
                }
            }
        }

        recording_id = save_recording(
            session_id="patient123_session001",
            object_name="bottle",
            prompt_text="Say this",
            expected_answer="bottle",
            audio_path="/tmp/test.wav",
            analysis_result=analysis_result,
            patient_id="patient123"
        )

        # Check patient_phoneme_history was updated
        db = SessionLocal()
        try:
            result = db.execute(
                text("SELECT * FROM patient_phoneme_history WHERE patient_id = :pid"),
                {"pid": "patient123"}
            )
            rows = result.fetchall()

            assert len(rows) == 2  # AA and TH

            phoneme_counts = {row.phoneme: row.error_count for row in rows}
            assert phoneme_counts["AA"] == 2
            assert phoneme_counts["TH"] == 1
        finally:
            db.close()


class TestPatientHistory:
    """Tests for querying patient recording history"""

    def test_get_patient_history_empty(self, test_db):
        """Test getting history for patient with no recordings"""
        history = get_patient_history("nonexistent_patient")
        assert len(history) == 0

    def test_get_patient_history_multiple_recordings(self, test_db):
        """Test getting history with multiple recordings"""
        # Save 3 recordings for same patient
        for i in range(3):
            save_recording(
                session_id=f"patient456_session{i}",
                object_name="water",
                prompt_text=f"Prompt {i}",
                expected_answer="water",
                audio_path=f"/tmp/test{i}.wav",
                analysis_result={"transcript": "water"}
            )

        history = get_patient_history("patient456")
        assert len(history) == 3

    def test_get_patient_history_with_object_filter(self, test_db):
        """Test filtering history by object name"""
        # Save recordings for different objects
        save_recording(
            session_id="patient789_s1",
            object_name="water",
            prompt_text="What is this?",
            expected_answer="water",
            audio_path="/tmp/water.wav",
            analysis_result={"transcript": "water"}
        )

        save_recording(
            session_id="patient789_s2",
            object_name="bottle",
            prompt_text="What is this?",
            expected_answer="bottle",
            audio_path="/tmp/bottle.wav",
            analysis_result={"transcript": "bottle"}
        )

        # Filter for only water
        water_history = get_patient_history("patient789", object_name="water")
        assert len(water_history) == 1
        assert water_history[0]["object_name"] == "water"

    def test_get_patient_history_limit(self, test_db):
        """Test that limit parameter works"""
        # Save 5 recordings
        for i in range(5):
            save_recording(
                session_id=f"patient999_s{i}",
                object_name="water",
                prompt_text="Test",
                expected_answer="water",
                audio_path=f"/tmp/test{i}.wav",
                analysis_result={"transcript": "water"}
            )

        # Request only 2
        history = get_patient_history("patient999", limit=2)
        assert len(history) == 2


class TestPatientProgress:
    """Tests for patient progress tracking"""

    def test_patient_progress_no_data(self, test_db):
        """Test progress for patient with no recordings"""
        progress = get_patient_progress("nonexistent")

        assert progress["total_recordings"] == 0
        assert progress["avg_fluency"] == 0
        assert progress["avg_per"] == 0
        assert len(progress["lfr_trend"]) == 0
        assert len(progress["most_problematic_phonemes"]) == 0

    def test_patient_progress_aggregates(self, test_db):
        """Test that progress correctly aggregates metrics"""
        # Save 3 recordings with known metrics
        for i in range(3):
            analysis_result = {
                "transcript": "test",
                "wer": 0.1 * (i + 1),  # 0.1, 0.2, 0.3
                "fluency_analysis": {
                    "longest_fluent_run": 5 + i,  # 5, 6, 7
                    "fluency_percentage": 80.0 + i * 5,  # 80, 85, 90
                    "total_pauses": 2
                },
                "phoneme_analysis": {
                    "per_rule": 0.05 * (i + 1),  # 0.05, 0.10, 0.15
                    "total_phonemes": 10
                }
            }

            save_recording(
                session_id=f"patientX_s{i}",
                object_name="water",
                prompt_text="Test",
                expected_answer="test",
                audio_path=f"/tmp/test{i}.wav",
                analysis_result=analysis_result,
                patient_id="patientX"
            )

        progress = get_patient_progress("patientX")

        assert progress["total_recordings"] == 3
        assert abs(progress["avg_wer"] - 0.2) < 0.01  # (0.1 + 0.2 + 0.3) / 3
        assert abs(progress["avg_fluency"] - 85.0) < 0.01  # (80 + 85 + 90) / 3
        assert abs(progress["avg_per"] - 0.1) < 0.01  # (0.05 + 0.10 + 0.15) / 3
        assert abs(progress["avg_lfr"] - 6.0) < 0.01  # (5 + 6 + 7) / 3

    def test_patient_progress_problematic_phonemes(self, test_db):
        """Test that most problematic phonemes are identified"""
        # Save recording with phoneme errors
        analysis_result = {
            "transcript": "test",
            "phoneme_analysis": {
                "per_rule": 0.2,
                "total_phonemes": 10,
                "problematic_phonemes": {
                    "AA": 5,
                    "TH": 3,
                    "R": 1
                }
            }
        }

        save_recording(
            session_id="patientY_s1",
            object_name="water",
            prompt_text="Test",
            expected_answer="test",
            audio_path="/tmp/test.wav",
            analysis_result=analysis_result,
            patient_id="patientY"
        )

        progress = get_patient_progress("patientY")

        assert len(progress["most_problematic_phonemes"]) == 3

        # Should be sorted by error rate/count (AA first with 5 errors)
        assert progress["most_problematic_phonemes"][0]["phoneme"] == "AA"
        assert progress["most_problematic_phonemes"][0]["error_count"] == 5


class TestSessionSummary:
    """Tests for session summary queries"""

    def test_session_summary_single_recording(self, test_db):
        """Test summary for session with one recording"""
        save_recording(
            session_id="session_abc",
            object_name="water",
            prompt_text="What is this?",
            expected_answer="water",
            audio_path="/tmp/test.wav",
            analysis_result={
                "transcript": "water",
                "fluency_analysis": {
                    "longest_fluent_run": 5,
                    "fluency_percentage": 85.0
                },
                "phoneme_analysis": {
                    "per_rule": 0.1
                }
            }
        )

        summary = get_session_summary("session_abc")

        assert summary["session_id"] == "session_abc"
        assert summary["total_prompts"] == 1
        assert summary["object_name"] == "water"
        assert len(summary["recordings"]) == 1
        assert summary["session_averages"]["avg_fluency"] == 85.0

    def test_session_summary_multiple_recordings(self, test_db):
        """Test summary with multiple recordings in same session"""
        # Save 3 recordings in same session
        for i in range(3):
            analysis_result = {
                "transcript": f"test {i}",
                "fluency_analysis": {
                    "longest_fluent_run": 5 + i,
                    "fluency_percentage": 80.0 + i * 10
                },
                "phoneme_analysis": {
                    "per_rule": 0.1 + i * 0.05
                }
            }

            save_recording(
                session_id="session_xyz",
                object_name="bottle",
                prompt_text=f"Prompt {i}",
                expected_answer="bottle",
                audio_path=f"/tmp/test{i}.wav",
                analysis_result=analysis_result
            )

        summary = get_session_summary("session_xyz")

        assert summary["total_prompts"] == 3
        assert len(summary["recordings"]) == 3
        assert summary["object_name"] == "bottle"

        # Check averages
        avg_fluency = summary["session_averages"]["avg_fluency"]
        assert abs(avg_fluency - 90.0) < 0.01  # (80 + 90 + 100) / 3


class TestPhonemeTrends:
    """Tests for phoneme trend tracking"""

    def test_phoneme_trend_no_errors(self, test_db):
        """Test trend for phoneme with no errors"""
        trend = get_phoneme_trends("patient123", "AA")
        assert len(trend) == 0

    def test_phoneme_trend_multiple_occurrences(self, test_db):
        """Test tracking phoneme errors over time"""
        # Save 3 recordings with AA errors
        for i in range(3):
            analysis_result = {
                "transcript": "bottle",
                "phoneme_analysis": {
                    "per_rule": 0.1,
                    "problematic_phonemes": {
                        "AA": i + 1  # Increasing errors
                    }
                }
            }

            save_recording(
                session_id=f"patientZ_s{i}",
                object_name="bottle",
                prompt_text="Say bottle",
                expected_answer="bottle",
                audio_path=f"/tmp/test{i}.wav",
                analysis_result=analysis_result
            )

        trend = get_phoneme_trends("patientZ", "AA")

        assert len(trend) == 3
        assert all("error_count" in item for item in trend)
        assert all("date" in item for item in trend)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
