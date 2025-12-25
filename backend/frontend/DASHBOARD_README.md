# Therapist Dashboard

A comprehensive web-based dashboard for speech therapists to track patient progress, visualize trends, and identify focus areas for therapy.

## Features

### 📊 Summary Metrics
- **Total Recordings**: Track compliance and homework completion
- **Average PER**: Phoneme Error Rate showing pronunciation accuracy
- **Average LFR**: Longest Fluent Run indicating fluency progress
- **Average Fluency %**: Overall fluency percentage

### 📈 Interactive Charts (Chart.js)
1. **LFR Trend** - Line chart showing fluency improvement over time
2. **PER Trend** - Line chart tracking pronunciation error rate
3. **Fluency Trend** - Line chart displaying fluency percentage changes
4. **Problematic Phonemes** - Bar chart ranking the top 10 phoneme errors

### 🔍 Clinical Insights Panel
Automated analysis providing:
- **Positive insights** (✅ green) - Areas of improvement
- **Focus areas** (⚠️ red) - Sounds/skills needing attention
- **Neutral observations** (ℹ️ yellow) - General progress notes

Example insights:
- "LFR improved from 3.5 to 7.2 words (+3.7 improvement)"
- "Focus areas: /TH/, /R/, /L/ sounds show highest error rates"
- "Excellent fluency at 85.3% - patient shows strong fluent speech"

### 📝 Recent Sessions Table
- View all patient recordings sorted by date
- Click any row to expand and see:
  - Full transcript
  - Complete metrics (WER, PER, speech rate, pause ratio)
  - Detailed phoneme errors with error types
  - Stuttering events (repetitions, prolongations, interjections)
  - Clinical notes

### 🎨 Color Coding
- **Green badges** - Correct semantic classification
- **Yellow badges** - Partial match
- **Red badges** - Wrong/off-target responses
- **Green insights** - Positive progress
- **Yellow insights** - Neutral observations
- **Red insights** - Areas needing focus

## Usage

### Quick Start

1. **Start the backend server**
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```

2. **Open the dashboard**
   - Navigate to: `backend/frontend/dashboard.html`
   - Open in your browser (Chrome, Firefox, Safari, Edge)

3. **Load patient data**
   - Enter patient ID (e.g., "john_doe", "patient123")
   - Click "Load Patient Data"
   - Dashboard will populate with real-time data from API

### Patient ID Format

Patient IDs are matched using the `session_id` prefix pattern. For example:
- Session ID: `john_doe_session_001` → Patient ID: `john_doe`
- Session ID: `patient123_abc` → Patient ID: `patient123`

To enable proper patient tracking, use consistent session ID patterns when recording.

### Interpreting the Dashboard

#### 1. Summary Cards
- **Total Recordings**: Higher numbers indicate better compliance
- **Avg PER**: Lower is better (target: <10%)
- **Avg LFR**: Higher is better (improvement = increasing trend)
- **Avg Fluency**: Higher is better (target: >80%)

#### 2. Trend Charts
- **Upward LFR trend** = Improving fluency
- **Downward PER trend** = Improving pronunciation
- **Upward Fluency trend** = Reducing dysfluencies
- **Problematic Phonemes** = Focus therapy on top 3-5 sounds

#### 3. Clinical Insights
- Read these first for quick assessment
- Green insights = celebrate progress with patient
- Red insights = plan targeted exercises
- Yellow insights = general monitoring

#### 4. Session Details
- Click any session to see what happened
- Review phoneme errors to identify patterns
- Check stuttering events for fluency concerns
- Read clinical notes for automated recommendations

## API Integration

The dashboard uses these endpoints:

```javascript
GET /patient/{patient_id}/progress
GET /patient/{patient_id}/history?limit=50
```

See [PROJECT_DOCUMENTATION.txt](../../PROJECT_DOCUMENTATION.txt) for full API details.

## Technical Details

### Technologies
- **Vanilla JavaScript** (ES6+)
- **Chart.js 4.4.0** - Interactive charts
- **Pure CSS** - No frameworks, responsive grid layout
- **Fetch API** - RESTful data loading

### Browser Compatibility
- Chrome 90+ ✅
- Firefox 88+ ✅
- Safari 14+ ✅
- Edge 90+ ✅

### Mobile Support
Fully responsive design:
- **Desktop** - Full 4-column layout
- **Tablet** - 2-column adaptive grid
- **Mobile** - Single column, optimized for scrolling

## Customization

### Changing Backend URL
Edit line 28 in `dashboard.html`:
```javascript
const BACKEND = "http://127.0.0.1:8000";  // Change to your backend URL
```

### Adjusting Chart Colors
Chart colors are defined in each chart creation function:
- `createLFRChart()` - Line 522 (blue: #3498db)
- `createPERChart()` - Line 573 (orange: #f39c12)
- `createPhonemeChart()` - Line 624 (multi-color)
- `createFluencyChart()` - Line 665 (green: #2ecc71)

### Modifying Insight Logic
Insight generation logic is in `generateInsights()` function (line 420).

Thresholds:
- LFR improvement threshold: 1.0 words
- Fluency excellent: ≥80%
- Fluency needs work: <60%
- PER excellent: <10%
- PER moderate: >20%

## Troubleshooting

### "Failed to load patient data"
**Cause**: Backend not running or incorrect patient ID

**Solutions**:
1. Verify backend is running: `http://127.0.0.1:8000/docs`
2. Check patient ID exists (must match session_id prefix)
3. Open browser console (F12) for detailed errors

### No data showing / Empty charts
**Cause**: Patient has no recordings yet

**Solution**: Record some data using `recorder.html` first with matching session ID pattern

### Charts not rendering
**Cause**: Chart.js CDN not loaded

**Solution**: Check internet connection (dashboard requires Chart.js CDN)

### Slow loading
**Cause**: Large number of recordings

**Solution**:
- Reduce `limit` parameter in history query (line 380)
- Current default: 50 recordings
- Recommended: 20-100 depending on use case

## Best Practices

### For Therapists
1. **Review weekly** - Check dashboard once per week to track trends
2. **Before sessions** - Load dashboard to plan focus areas
3. **After sessions** - Refresh to see latest recordings
4. **Document observations** - Use insights to guide clinical notes
5. **Share with patients** - Show charts to visualize progress

### For Researchers
1. **Export data** - Use browser console or API directly for exports
2. **Track cohorts** - Load multiple patients to compare
3. **Time series analysis** - Use trend charts for outcome studies
4. **Phoneme patterns** - Use bar chart to identify common errors
5. **Publication figures** - Chart.js supports screenshot/export

## Future Enhancements

Potential additions:
- [ ] Multi-patient comparison view
- [ ] Date range filtering
- [ ] Export to CSV/PDF
- [ ] Audio playback in session details
- [ ] Custom metric thresholds
- [ ] Therapy goal tracking
- [ ] Print-friendly report view
- [ ] Patient progress notes storage

## Support

For issues or questions:
- Check [PROJECT_DOCUMENTATION.txt](../../PROJECT_DOCUMENTATION.txt)
- Review API endpoints at `http://127.0.0.1:8000/docs`
- Examine browser console for errors (F12)

## License

Part of the AI-Powered Speech Therapy Homework System research project.

---

**Version**: 2.0.0
**Last Updated**: December 25, 2025
**Status**: Production-ready for research use
