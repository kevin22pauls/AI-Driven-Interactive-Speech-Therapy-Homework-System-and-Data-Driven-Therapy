"""
Migration script to add ML phoneme analysis columns to the recordings table.

Run this script to update an existing database with the new ML analysis columns.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'speechtherapy.db')

ML_PHONEME_COLUMNS = [
    ("ml_per", "REAL"),
    ("ml_gop", "REAL"),
    ("ml_confidence", "REAL"),
    ("ml_detected_phonemes_json", "TEXT"),
    ("ml_detected_ipa_json", "TEXT"),
    ("ml_alignment_json", "TEXT"),
    ("ml_phoneme_scores_json", "TEXT"),
]


def migrate():
    """Add ML phoneme columns to recordings table if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(recordings)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Add missing columns
    added = []
    for col_name, col_type in ML_PHONEME_COLUMNS:
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE recordings ADD COLUMN {col_name} {col_type}")
                added.append(col_name)
                print(f"Added column: {col_name} ({col_type})")
            except sqlite3.OperationalError as e:
                print(f"Error adding {col_name}: {e}")

    conn.commit()
    conn.close()

    if added:
        print(f"\nMigration complete. Added {len(added)} columns.")
    else:
        print("No migration needed. All columns already exist.")

    return added


if __name__ == "__main__":
    migrate()
