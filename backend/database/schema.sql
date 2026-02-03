-- Enhanced Schema for AI-Powered Speech Therapy System
-- Includes support for enhanced analysis algorithms (WPER, SSI-4, etc.)

CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    patient_id TEXT,  -- Added for patient tracking
    object_name TEXT,
    prompt_text TEXT,
    expected_answer TEXT,
    audio_path TEXT,
    transcript TEXT,

    -- Basic metrics
    wer REAL,
    speech_rate REAL,
    pause_ratio REAL,

    -- Standard phoneme metrics
    per REAL,  -- Phoneme Error Rate
    total_phonemes INTEGER,
    phoneme_errors_json TEXT,  -- JSON array of phoneme errors
    problematic_phonemes_json TEXT,  -- JSON object of problematic phonemes
    clinical_notes_json TEXT,  -- JSON array of clinical notes

    -- Enhanced phoneme metrics (NEW)
    wper REAL,  -- Weighted Phoneme Error Rate
    phoneme_class_errors_json TEXT,  -- JSON object of errors by phoneme class
    error_pattern_analysis_json TEXT,  -- JSON with dominant error type, similarity ratio, etc.
    conduite_d_approche_detected INTEGER DEFAULT 0,  -- Boolean flag
    multi_attempt_analysis_json TEXT,  -- JSON with attempt scores

    -- ML Phoneme Analysis (Wav2Vec2-based)
    ml_per REAL,  -- ML-based Phoneme Error Rate
    ml_gop REAL,  -- Goodness of Pronunciation score
    ml_confidence REAL,  -- Model confidence score
    ml_detected_phonemes_json TEXT,  -- JSON array of detected phonemes (ARPAbet)
    ml_detected_ipa_json TEXT,  -- JSON array of detected phonemes (IPA)
    ml_alignment_json TEXT,  -- JSON array of (expected, detected) pairs
    ml_phoneme_scores_json TEXT,  -- JSON array of per-phoneme GOP scores
    ml_transcript TEXT,  -- Raw transcript reconstructed from ML phonemes
    ml_transcript_details_json TEXT,  -- JSON with word-by-word reconstruction

    -- Semantic evaluation
    semantic_classification TEXT,  -- 'correct', 'partial', 'wrong', 'circumlocution', 'semantic_paraphasia'
    semantic_score REAL,
    direct_similarity REAL,  -- NEW: Direct embedding similarity
    category_similarity REAL,  -- NEW: WordNet category similarity
    circumlocution_detected INTEGER DEFAULT 0,  -- NEW: Boolean flag
    circumlocution_features_json TEXT,  -- NEW: JSON with matched features
    semantic_paraphasia_json TEXT,  -- NEW: JSON with paraphasia details

    -- Standard fluency metrics
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

    -- Enhanced fluency metrics (NEW)
    lfr_with_tolerance INTEGER,  -- LFR allowing minor dysfluencies
    lfr_ratio REAL,  -- LFR / total words

    -- Enhanced pause metrics (NEW)
    anomic_pause_count INTEGER DEFAULT 0,
    apraxic_pause_count INTEGER DEFAULT 0,
    mean_pause_duration REAL,

    -- Speech rate metrics (NEW)
    articulation_rate REAL,  -- Syllables per minute (excludes pauses)
    speaking_rate REAL,  -- Syllables per minute (includes all)
    pause_adjusted_rate REAL,  -- Excludes pathological pauses
    phonation_time REAL,  -- Total speech time
    phonation_ratio REAL,  -- Proportion speaking
    pathological_pause_ratio REAL,

    -- Rate variability metrics (NEW)
    local_cv REAL,  -- Local coefficient of variation
    global_cv REAL,  -- Global coefficient of variation
    normalized_pvi REAL,  -- Pairwise Variability Index
    rate_trend TEXT,  -- 'slowing', 'speeding', 'stable'

    -- SSI-4 approximation (NEW)
    ssi_frequency_pct REAL,
    ssi_frequency_score INTEGER,
    ssi_avg_duration REAL,
    ssi_duration_score INTEGER,
    ssi_total_score INTEGER,
    ssi_severity TEXT,  -- 'very_mild', 'mild', 'moderate', 'severe', 'very_severe'

    -- Weighted fluency scores (NEW)
    standard_fluency_pct REAL,
    weighted_fluency_pct REAL,
    clinical_fluency_pct REAL,
    dysfluency_profile_json TEXT,  -- JSON with counts by type

    -- Metadata
    analysis_mode TEXT DEFAULT 'enhanced',  -- 'standard' or 'enhanced'
    disorder_type TEXT DEFAULT 'aphasia',  -- Type used for analysis
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
    model_name TEXT,  -- e.g., "gemma2:2b"
    generation_method TEXT,  -- "llm" or "fallback"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NEW: Table for tracking session-level progress
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
    objects_practiced TEXT,  -- JSON array of objects
    clinical_summary_json TEXT,  -- JSON with session summary
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NEW: Table for tracking patient baseline metrics
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
    cv_rate REAL,  -- Coefficient of variation for rate
    baseline_date TIMESTAMP,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_recordings_session ON recordings(session_id);
CREATE INDEX IF NOT EXISTS idx_recordings_patient ON recordings(patient_id);
CREATE INDEX IF NOT EXISTS idx_recordings_object ON recordings(object_name);
CREATE INDEX IF NOT EXISTS idx_recordings_created ON recordings(created_at);
CREATE INDEX IF NOT EXISTS idx_recordings_analysis_mode ON recordings(analysis_mode);
CREATE INDEX IF NOT EXISTS idx_patient_phoneme ON patient_phoneme_history(patient_id, phoneme);
CREATE INDEX IF NOT EXISTS idx_generated_prompts_object ON generated_prompts(object_name);
CREATE INDEX IF NOT EXISTS idx_session_progress_patient ON session_progress(patient_id);
CREATE INDEX IF NOT EXISTS idx_patient_baselines ON patient_baselines(patient_id);
