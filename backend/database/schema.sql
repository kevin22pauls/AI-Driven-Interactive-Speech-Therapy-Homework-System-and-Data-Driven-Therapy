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
    -- Fluency metrics
    longest_fluent_run INTEGER,
    total_pauses INTEGER,
    hesitation_count INTEGER,
    block_count INTEGER,
    fluency_percentage REAL,
    dysfluencies_per_100_words REAL,
    dysfluencies_per_minute REAL,
    speech_rate_variability REAL,
    stuttering_events_json TEXT,  -- JSON array of stuttering events
    pauses_json TEXT,  -- JSON array of pauses
    fluency_notes_json TEXT,  -- JSON array of fluency clinical notes
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

-- Table for caching LLM-generated prompts
CREATE TABLE IF NOT EXISTS generated_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_name TEXT UNIQUE NOT NULL,
    prompts_json TEXT NOT NULL,  -- JSON with questions and sentences
    model_name TEXT,  -- e.g., "llama3.2:3b"
    generation_method TEXT,  -- "llm" or "fallback"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_recordings_session ON recordings(session_id);
CREATE INDEX IF NOT EXISTS idx_recordings_object ON recordings(object_name);
CREATE INDEX IF NOT EXISTS idx_recordings_created ON recordings(created_at);
CREATE INDEX IF NOT EXISTS idx_patient_phoneme ON patient_phoneme_history(patient_id, phoneme);
CREATE INDEX IF NOT EXISTS idx_generated_prompts_object ON generated_prompts(object_name);
