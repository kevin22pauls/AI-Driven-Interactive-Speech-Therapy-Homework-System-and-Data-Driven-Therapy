"""
Database Migration: Add Fluency Columns

This script adds the fluency analysis columns to an existing recordings table.
Run this if you have existing data you want to preserve.

Usage:
    python backend/database/migrate_add_fluency_columns.py
"""

import sqlite3
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_URL

# Extract database path from DB_URL (remove sqlite:/// prefix)
db_path = DB_URL.replace('sqlite:///', '')

print(f"Migrating database: {db_path}")

if not os.path.exists(db_path):
    print(f"ERROR: Database file not found at {db_path}")
    print("No migration needed - new database will be created with correct schema on startup")
    sys.exit(0)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if columns already exist
cursor.execute("PRAGMA table_info(recordings)")
columns = [row[1] for row in cursor.fetchall()]

fluency_columns = [
    'longest_fluent_run',
    'total_pauses',
    'hesitation_count',
    'block_count',
    'fluency_percentage',
    'dysfluencies_per_100_words',
    'dysfluencies_per_minute',
    'speech_rate_variability',
    'stuttering_events_json',
    'pauses_json',
    'fluency_notes_json'
]

# Add missing columns
added_count = 0
for column in fluency_columns:
    if column not in columns:
        print(f"Adding column: {column}")

        if column.endswith('_json'):
            sql = f"ALTER TABLE recordings ADD COLUMN {column} TEXT"
        elif column in ['longest_fluent_run', 'total_pauses', 'hesitation_count', 'block_count']:
            sql = f"ALTER TABLE recordings ADD COLUMN {column} INTEGER"
        else:  # REAL columns
            sql = f"ALTER TABLE recordings ADD COLUMN {column} REAL"

        try:
            cursor.execute(sql)
            added_count += 1
        except sqlite3.OperationalError as e:
            print(f"WARNING: Could not add {column}: {e}")

conn.commit()
conn.close()

if added_count > 0:
    print(f"\n✅ Migration complete! Added {added_count} fluency columns.")
    print("You can now restart your server.")
else:
    print("\n✅ All fluency columns already exist. No migration needed.")

print("\nDatabase is ready!")
