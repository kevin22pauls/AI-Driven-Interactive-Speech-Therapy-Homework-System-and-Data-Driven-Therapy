CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    object_name TEXT,
    prompt_text TEXT,
    expected_answer TEXT,
    audio_path TEXT,
    transcript TEXT,
    wer REAL,
    speech_rate REAL,
    pause_ratio REAL,
    per REAL,  -- Phoneme Error Rate
    total_phonemes INTEGER,
    phoneme_errors_json TEXT,  -- JSON array of phoneme errors
    problematic_phonemes_json TEXT,  -- JSON object of problematic phonemes
    clinical_notes_json TEXT,  -- JSON array of clinical notes
    semantic_classification TEXT,  -- 'correct', 'partial', 'wrong'
    semantic_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for tracking patient progress over time
CREATE TABLE IF NOT EXISTS patient_phoneme_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    phoneme TEXT,
    error_count INTEGER DEFAULT 0,
    occurrence_count INTEGER DEFAULT 0,
    last_error_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(patient_id, phoneme)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_recordings_session ON recordings(session_id);
CREATE INDEX IF NOT EXISTS idx_recordings_object ON recordings(object_name);
CREATE INDEX IF NOT EXISTS idx_recordings_created ON recordings(created_at);
CREATE INDEX IF NOT EXISTS idx_patient_phoneme ON patient_phoneme_history(patient_id, phoneme);
