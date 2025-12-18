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
