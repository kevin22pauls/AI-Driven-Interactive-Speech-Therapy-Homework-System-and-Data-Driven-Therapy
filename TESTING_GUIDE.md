# Testing Guide - Dashboard with Real Data

This guide walks you through creating test data and viewing it in the therapist dashboard.

## Quick Start (5 minutes)

### Step 1: Start the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Step 2: Create Test Data with Recorder

1. **Open the recorder**
   - Navigate to: `backend/frontend/recorder.html`
   - Open in your browser

2. **Enter a patient ID**
   - In the blue "Patient ID" box, enter: `john_doe`
   - This will track all recordings under this patient

3. **Record your first session** (repeat 3-5 times)
   - Click "Get Prompt"
   - Read the prompt aloud
   - Click "Start Recording"
   - Speak your answer (e.g., "This is a bottle")
   - Recording stops automatically after silence
   - View results (you'll see transcript, PER, LFR, fluency metrics)
   - Click "Get Prompt" again for another prompt

4. **Notice the green banner**
   - After first recording, you'll see: "✓ Session active: john_doe_1234567890_abc123"
   - Patient ID field is now locked
   - All recordings in this session use the same patient prefix

5. **Create variety** (for better dashboard visualization)
   - Record 5-10 prompts
   - Try different responses (some correct, some with errors)
   - Try speaking with pauses or repetitions to generate fluency data
   - Click "New Object" to practice different words

### Step 3: View in Dashboard

1. **Open the dashboard**
   - Navigate to: `backend/frontend/dashboard.html`
   - Open in your browser

2. **Load patient data**
   - In the "Patient ID" field, enter: `john_doe`
   - Click "Load Patient Data"

3. **Explore the dashboard**
   - **Summary Cards** - See total recordings, avg PER, avg LFR, avg fluency
   - **Clinical Insights** - Read automated recommendations (green/yellow/red)
   - **Charts** - Hover over data points for exact values
     - LFR trend (is it increasing?)
     - PER trend (is it decreasing?)
     - Fluency percentage
     - Most problematic phonemes (bar chart)
   - **Sessions Table** - Scroll down, click any row to expand details
     - View full transcript
     - See all metrics
     - Review phoneme errors
     - Check stuttering events

## Creating Multiple Patients

To test multi-patient tracking:

1. **Patient 1: "john_doe"**
   - Record 5-10 prompts with this ID
   - Close browser tab

2. **Patient 2: "jane_smith"**
   - Open recorder.html again
   - Enter "jane_smith" as patient ID
   - Record 5-10 prompts
   - Close tab

3. **Patient 3: "patient123"**
   - Repeat with ID "patient123"

4. **View each in dashboard**
   - Dashboard → Enter "john_doe" → Load
   - Dashboard → Enter "jane_smith" → Load
   - Dashboard → Enter "patient123" → Load

Each patient will have separate progress charts!

## Test Scenarios

### Scenario 1: Improving Patient
**Goal**: Show improvement trends in dashboard

1. **Day 1** (simulate with first 3 recordings):
   - Speak with many pauses and repetitions
   - Make pronunciation errors
   - Short fluent runs

2. **Day 2** (next 3 recordings):
   - Speak more fluently
   - Reduce pauses
   - Better pronunciation

3. **Day 3** (next 3 recordings):
   - Speak very fluently
   - Clear pronunciation
   - Long fluent runs

**Expected Dashboard**:
- LFR chart shows upward trend
- PER chart shows downward trend
- Fluency percentage increasing
- Positive insights: "LFR improved from X to Y"

### Scenario 2: Problematic Phonemes
**Goal**: Show phoneme tracking

1. **Record these specific prompts**:
   - Say "bottle" → Mispronounce as "bootle" (AA → UW error)
   - Say "this" → Mispronounce as "dis" (TH → D error)
   - Say "red" → Mispronounce as "wed" (R → W error)

2. **View dashboard**:
   - Problematic Phonemes chart shows AA, TH, R as top errors
   - Clinical insights: "Focus areas: /AA/, /TH/, /R/ sounds"

### Scenario 3: Fluency Issues
**Goal**: Show stuttering detection

1. **Record with dysfluencies**:
   - "I... I... I want water" (repetition)
   - "Um... this is... uh... a bottle" (interjections)
   - Long pauses between words

2. **View dashboard**:
   - Low fluency percentage
   - Stuttering events table shows repetitions and interjections
   - Clinical insights: "Fluency at 45% - consider fluency-building exercises"

## Verifying Data Persistence

### Check Database Directly

```bash
cd backend
sqlite3 speechtherapy.db

# View all recordings
SELECT id, session_id, object_name, transcript, per, longest_fluent_run
FROM recordings
ORDER BY created_at DESC
LIMIT 10;

# View patient phoneme history
SELECT * FROM patient_phoneme_history;

# Exit SQLite
.quit
```

### Check API Endpoints

Open browser console (F12) and run:

```javascript
// Check patient progress
fetch('http://127.0.0.1:8000/patient/john_doe/progress')
  .then(r => r.json())
  .then(data => console.log(data));

// Check patient history
fetch('http://127.0.0.1:8000/patient/john_doe/history')
  .then(r => r.json())
  .then(data => console.log(data));
```

Or visit directly:
- http://127.0.0.1:8000/patient/john_doe/progress
- http://127.0.0.1:8000/patient/john_doe/history
- http://127.0.0.1:8000/docs (FastAPI interactive docs)

## Troubleshooting

### Dashboard shows "No data" or 0 recordings

**Problem**: Patient ID doesn't match session IDs in database

**Solution**:
1. Check session IDs in recorder (green banner shows full session ID)
2. Patient ID must match the prefix before the first underscore
3. Example: If session is "john_doe_123_abc", use patient ID "john_doe"

### Charts are empty

**Problem**: Not enough data points for trends

**Solution**: Record at least 3-5 sessions. Trends need multiple data points.

### "Failed to load patient data" error

**Problem**: Backend not running or patient doesn't exist

**Solution**:
1. Verify backend is running: http://127.0.0.1:8000/docs
2. Check patient has recordings: http://127.0.0.1:8000/patient/john_doe/history
3. Open browser console (F12) for detailed error messages

### Patient ID field is blank in recorder

**Problem**: Auto-generated anonymous ID

**Solution**: Refresh page and enter patient ID BEFORE clicking "Get Prompt"

## Expected Results

After recording 5-10 prompts as "john_doe", the dashboard should show:

**Summary Cards:**
- Total Recordings: 5-10
- Average PER: 5-20% (depending on pronunciation accuracy)
- Average LFR: 3-8 words (depending on fluency)
- Average Fluency: 60-90% (depending on dysfluencies)

**Clinical Insights Panel:**
- 2-5 automated insights
- Mix of green (positive), yellow (neutral), red (focus) items
- Example: "LFR improved from 4.2 to 6.8 words"
- Example: "Focus areas: /TH/, /R/ sounds show highest error rates"

**Charts:**
- LFR Trend: Line chart with 3-10 data points
- PER Trend: Line chart showing error rate changes
- Fluency Trend: Line chart of fluency percentage
- Problematic Phonemes: Bar chart with top 5-10 phonemes

**Sessions Table:**
- 5-10 rows (one per recording)
- Click any row → expands to show:
  - Full transcript
  - Metrics (WER, PER, speech rate, pause ratio)
  - Phoneme errors (if any)
  - Stuttering events (if any)
  - Clinical notes

## Next Steps

Once you've verified the dashboard works with test data:

1. **Clean up test data** (optional):
   ```bash
   cd backend
   rm speechtherapy.db
   # Restart server to create fresh database
   ```

2. **Start collecting real patient data**:
   - Use actual patient IDs (ensure HIPAA compliance if needed)
   - Record therapy sessions regularly
   - Review dashboard weekly to track progress

3. **Export data for research** (see API docs):
   - Use API endpoints to fetch JSON data
   - Process with Python/R for statistical analysis
   - Generate publication figures from trend data

## Tips for Best Results

1. **Consistent Patient IDs**: Use the same format (e.g., "patient001", "patient002")
2. **Multiple Sessions**: Record 5-10 prompts per session for meaningful trends
3. **Regular Recording**: Record weekly to show progress over time
4. **Variety**: Practice different objects and prompt types
5. **Natural Speech**: Speak naturally to capture real fluency patterns

## Demo Video Script

If creating a demo:

1. Open recorder → Enter "demo_patient"
2. Record 3 prompts:
   - First: Slow, with pauses and repetitions
   - Second: Better fluency, some errors
   - Third: Very fluent, clear pronunciation
3. Open dashboard → Enter "demo_patient" → Load
4. Point out:
   - Summary showing 3 recordings
   - LFR increasing (chart going up)
   - Clinical insights showing improvement
   - Click session to show details
5. Show Chart.js interactivity (hover tooltips)

---

**Version**: 2.0.0
**Last Updated**: December 25, 2025
**Status**: Ready for testing
