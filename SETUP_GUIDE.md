# Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Database Setup

The database will be automatically created on first run. If you encounter schema errors after pulling updates:

```bash
# Delete old database (WARNING: This deletes all existing data)
rm backend/speechtherapy.db

# The new database will be created automatically when you start the server
```

### 3. Start the Server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
Loaded 134000+ words from CMU Dictionary
Gentle service not reachable (optional - phoneme timestamps unavailable)
```

### 4. Open the Frontend

Open `backend/frontend/recorder.html` in your web browser.

### 5. (Optional) Start Gentle for Phoneme Timestamps

If you want phoneme-level timestamps from forced alignment:

```bash
# From project root
docker-compose up -d gentle

# Verify it's running
curl http://localhost:8765
```

---

## Common Issues

### Issue: "sqlite3.OperationalError: no such column"

**Cause**: Old database schema doesn't match new code.

**Solution**:
```bash
rm backend/speechtherapy.db
# Restart the server - new DB will be created
```

### Issue: "Gentle service not reachable"

**Cause**: Gentle Docker container not running (this is optional).

**Solution**:
```bash
docker-compose up -d gentle
```

Or ignore it - the system works without Gentle (you just won't get phoneme timestamps).

### Issue: "CMU Dictionary not found"

**Cause**: Dictionary file not downloaded.

**Solution**:
```bash
cd backend/data
curl -o cmudict-0.7b.txt https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
```

### Issue: "ModuleNotFoundError: No module named 'editdistance'"

**Cause**: New dependencies not installed.

**Solution**:
```bash
cd backend
pip install -r requirements.txt
```

---

## Testing

Run the test suite:

```bash
cd backend
pytest tests/test_phoneme_analysis.py -v
```

Expected output:
```
test_per_perfect_match PASSED
test_per_with_substitution PASSED
test_alignment_perfect_match PASSED
...
```

---

## Project Structure

```
FINAL PROJECT/
├── backend/
│   ├── main.py                  # Start here
│   ├── requirements.txt         # Dependencies
│   ├── speechtherapy.db         # Auto-generated (gitignored)
│   ├── services/                # Core logic
│   ├── routers/                 # API endpoints
│   ├── frontend/recorder.html   # Open this in browser
│   └── data/cmudict-0.7b.txt   # Phoneme dictionary
├── docker-compose.yml           # Gentle service
└── PROJECT_DOCUMENTATION.txt    # Full docs
```

---

## First Run Checklist

- [x] Python 3.8+ installed
- [x] Dependencies installed (`pip install -r requirements.txt`)
- [x] Old database deleted (if upgrading)
- [x] Server running (`uvicorn main:app --reload`)
- [x] Frontend opened (`backend/frontend/recorder.html`)
- [ ] Gentle running (optional: `docker-compose up -d gentle`)

---

## Development

### Starting Fresh

```bash
# 1. Delete database
rm backend/speechtherapy.db

# 2. Delete audio files (optional)
rm backend/storage/audio/*.webm
rm backend/storage/audio/*.wav

# 3. Start server
cd backend
uvicorn main:app --reload
```

### Checking Logs

The server logs show:
- ASR transcription progress
- Phoneme analysis results
- Semantic evaluation scores
- Error messages

Example:
```
INFO: Performing phoneme-level analysis
INFO: Phoneme analysis - PER: 0.20, Errors: 1, Total phonemes: 5
INFO: Semantic evaluation - Classification: partial, Score: 0.78
```

---

## Production Deployment

**WARNING**: This is a research system. For production:

1. **Add authentication** (patient accounts, therapist login)
2. **Encrypt database** (contains patient data)
3. **Use PostgreSQL** instead of SQLite
4. **Enable HTTPS** (currently HTTP only)
5. **Add data backup** (automated backups)
6. **Implement audit logging** (HIPAA compliance)
7. **Get IRB approval** (required for patient data)
8. **Add data retention policies** (GDPR/HIPAA)

---

## Getting Help

1. Check [PROJECT_DOCUMENTATION.txt](PROJECT_DOCUMENTATION.txt) for full technical details
2. Check [README.md](README.md) for usage examples
3. Run tests: `pytest tests/ -v`
4. Check server logs for error messages

---

**You're all set! Start the server and open the frontend to begin recording.**
