# database/schema.py
from sqlalchemy import text
from .db import engine
import os

# If you already have a schema.sql file you can load it, otherwise we use inline SQL
SQL_FILE = os.path.join(os.path.dirname(__file__), "schema.sql")

def create_tables_from_sqlfile():
    """
    Execute SQL in schema.sql against the engine.
    If the file doesn't exist, create a minimal recordings table as fallback.
    """
    if os.path.exists(SQL_FILE):
        with open(SQL_FILE, "r", encoding="utf-8") as f:
            sql = f.read()
        # Use raw connection with executescript for multi-statement SQL
        raw_conn = engine.raw_connection()
        try:
            raw_conn.executescript(sql)
            raw_conn.commit()
        finally:
            raw_conn.close()
    else:
        # Fallback minimal table
        fallback_sql = """
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_id INTEGER,
            audio_path TEXT,
            transcript TEXT,
            wer REAL,
            speech_rate REAL,
            pause_ratio REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        with engine.connect() as conn:
            conn.execute(text(fallback_sql))
            conn.commit()

def init_db():
    """Public initializer used by main.py"""
    create_tables_from_sqlfile()
