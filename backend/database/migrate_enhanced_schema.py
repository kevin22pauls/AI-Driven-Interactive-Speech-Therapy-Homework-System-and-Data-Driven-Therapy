"""
Database Migration Script for Enhanced Analysis Schema

This script adds new columns to the existing database to support the
enhanced analysis algorithms:
- Weighted Phoneme Error Rate (WPER)
- Conduite d'approche detection
- Circumlocution detection
- SSI-4 approximation
- Adaptive LFR
- Multi-scale rate variability

Run this script if you have an existing database and need to add the new columns.

Usage:
    python migrate_enhanced_schema.py
"""

import sqlite3
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_db_path():
    """Get the database path."""
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(backend_dir, "speechtherapy.db")


def get_existing_columns(cursor, table_name):
    """Get list of existing columns in a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]


def add_column_if_not_exists(cursor, table, column, column_type, default=None):
    """Add a column to a table if it doesn't already exist."""
    existing = get_existing_columns(cursor, table)
    if column not in existing:
        default_clause = f" DEFAULT {default}" if default is not None else ""
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}{default_clause}")
        print(f"  Added column: {table}.{column}")
        return True
    else:
        print(f"  Column already exists: {table}.{column}")
        return False


def migrate_recordings_table(cursor):
    """Add new columns to recordings table."""
    print("\nMigrating recordings table...")

    new_columns = [
        # Patient tracking
        ("patient_id", "TEXT", None),

        # Enhanced phoneme metrics
        ("wper", "REAL", None),
        ("phoneme_class_errors_json", "TEXT", None),
        ("error_pattern_analysis_json", "TEXT", None),
        ("conduite_d_approche_detected", "INTEGER", 0),
        ("multi_attempt_analysis_json", "TEXT", None),

        # Enhanced semantic
        ("direct_similarity", "REAL", None),
        ("category_similarity", "REAL", None),
        ("circumlocution_detected", "INTEGER", 0),
        ("circumlocution_features_json", "TEXT", None),
        ("semantic_paraphasia_json", "TEXT", None),

        # Enhanced fluency
        ("lfr_with_tolerance", "INTEGER", None),
        ("lfr_ratio", "REAL", None),

        # Enhanced pause metrics
        ("anomic_pause_count", "INTEGER", 0),
        ("apraxic_pause_count", "INTEGER", 0),
        ("mean_pause_duration", "REAL", None),

        # Speech rate metrics
        ("articulation_rate", "REAL", None),
        ("speaking_rate", "REAL", None),
        ("pause_adjusted_rate", "REAL", None),
        ("phonation_time", "REAL", None),
        ("phonation_ratio", "REAL", None),
        ("pathological_pause_ratio", "REAL", None),

        # Rate variability
        ("local_cv", "REAL", None),
        ("global_cv", "REAL", None),
        ("normalized_pvi", "REAL", None),
        ("rate_trend", "TEXT", None),

        # SSI-4
        ("ssi_frequency_pct", "REAL", None),
        ("ssi_frequency_score", "INTEGER", None),
        ("ssi_avg_duration", "REAL", None),
        ("ssi_duration_score", "INTEGER", None),
        ("ssi_total_score", "INTEGER", None),
        ("ssi_severity", "TEXT", None),

        # Weighted fluency
        ("standard_fluency_pct", "REAL", None),
        ("weighted_fluency_pct", "REAL", None),
        ("clinical_fluency_pct", "REAL", None),
        ("dysfluency_profile_json", "TEXT", None),

        # Metadata
        ("analysis_mode", "TEXT", "'enhanced'"),
        ("disorder_type", "TEXT", "'aphasia'"),
    ]

    added = 0
    for column, col_type, default in new_columns:
        if add_column_if_not_exists(cursor, "recordings", column, col_type, default):
            added += 1

    print(f"  Added {added} new columns to recordings table")


def create_new_tables(cursor):
    """Create new tables if they don't exist."""
    print("\nCreating new tables...")

    # Session progress table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            patient_id TEXT,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            total_recordings INTEGER DEFAULT 0,
            avg_wper REAL,
            avg_per REAL,
            avg_weighted_fluency REAL,
            avg_lfr REAL,
            avg_ssi_score REAL,
            objects_practiced TEXT,
            clinical_summary_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("  Created/verified session_progress table")

    # Patient baselines table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_baselines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            disorder_type TEXT DEFAULT 'aphasia',
            baseline_per REAL,
            baseline_wper REAL,
            baseline_lfr REAL,
            baseline_fluency REAL,
            baseline_articulation_rate REAL,
            mean_pause_duration REAL,
            std_pause_duration REAL,
            cv_rate REAL,
            baseline_date TIMESTAMP,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("  Created/verified patient_baselines table")


def create_indexes(cursor):
    """Create new indexes."""
    print("\nCreating indexes...")

    indexes = [
        ("idx_recordings_patient", "recordings", "patient_id"),
        ("idx_recordings_analysis_mode", "recordings", "analysis_mode"),
        ("idx_session_progress_patient", "session_progress", "patient_id"),
        ("idx_patient_baselines", "patient_baselines", "patient_id"),
    ]

    for idx_name, table, column in indexes:
        try:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})")
            print(f"  Created/verified index: {idx_name}")
        except sqlite3.OperationalError as e:
            print(f"  Index {idx_name} error: {e}")


def run_migration():
    """Run the full migration."""
    db_path = get_db_path()
    print(f"Migrating database: {db_path}")

    if not os.path.exists(db_path):
        print("Database does not exist. It will be created with the new schema when the app starts.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Migrate recordings table
        migrate_recordings_table(cursor)

        # Create new tables
        create_new_tables(cursor)

        # Create indexes
        create_indexes(cursor)

        # Commit changes
        conn.commit()
        print("\n✓ Migration completed successfully!")

    except Exception as e:
        conn.rollback()
        print(f"\n✗ Migration failed: {e}")
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    run_migration()
