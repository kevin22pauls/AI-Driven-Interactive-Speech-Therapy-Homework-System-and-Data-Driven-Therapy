# AI-Powered Speech Therapy Homework System

An intelligent speech therapy system for aphasia patients and speech disorders, featuring **phoneme-level analysis** to identify exactly which sounds patients struggle with.

![Research Grade](https://img.shields.io/badge/status-research%20grade-blue)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Overview

This system provides **multi-level speech analysis** with **longitudinal tracking**:
- **Word-level**: WER, speech rate, pause ratio
- **Phoneme-level**: PER, error classification, problematic sounds
- **Fluency-level**: LFR, stuttering detection, dysfluency metrics
- **Semantic-level**: Answer correctness, paraphrasing detection
- **Progress tracking**: Therapist dashboard with trend visualization

**Example**: When a patient says "bootle" instead of "bottle", the system:
- Identifies: PER: 20%, Error: `AA → UW` substitution at phoneme position 1
- Tracks: Patient has made this error 3 times across 5 sessions
- Recommends: "Consider vowel discrimination exercises"
- Visualizes: Progress chart showing AA error rate decreasing over time

---

## ✨ Key Features

### 🔬 Phoneme-Level Analysis (NEW)
- **Phoneme Error Rate (PER)** calculation
- Error type classification: substitutions, deletions, insertions
- **Identifies problematic phonemes** for targeted therapy
- Longitudinal tracking of phoneme improvement
- Clinical insights generation

### 🎙️ Speech Recognition
- Faster-Whisper ASR (CPU-optimized)
- Automatic silence detection
- Multi-format audio support (webm, wav, mp3)

### 📊 Comprehensive Metrics
- Word Error Rate (WER)
- Speech rate (words/minute)
- Pause ratio analysis
- Semantic similarity scoring

### 🧠 Semantic Analysis
- Sentence transformer-based evaluation
- Three-tier classification (correct/partial/wrong)
- Paraphrase detection
- Object identification verification

### 💻 Dual User Interfaces
**Patient Interface** (recorder.html):
- Browser-based recording
- Real-time feedback
- Color-coded results
- Detailed error visualization

**Therapist Dashboard** (dashboard.html) - NEW:
- Patient progress visualization
- Interactive Chart.js charts (LFR, PER, fluency trends)
- Clinical insights with automated recommendations
- Session drill-down for detailed analysis
- Problematic phoneme identification

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker (optional, for forced alignment)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd "FINAL PROJECT"
```

2. **Install Python dependencies**
```bash
cd backend
pip install -r requirements.txt
```

3. **(Optional) Start Gentle for phoneme timestamps**
```bash
docker-compose up -d gentle
```

4. **Start the backend server**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

5. **Open the frontend**
```
Patient Interface: Open backend/frontend/recorder.html in your web browser
Therapist Dashboard: Open backend/frontend/dashboard.html in your web browser
```

---

## 📖 Usage

### Patient Workflow (recorder.html)

1. **Click "Get Prompt"** → System provides a prompt (e.g., "What is this object?")
2. **Click "Start Recording"** → Speak your answer
3. **Recording auto-stops** after silence (900ms) or 8 seconds
4. **View Results**:
   - Transcript
   - Semantic evaluation (✓ correct / ⚠️ partial / ✗ wrong)
   - **Phoneme Error Rate (PER)**
   - **Longest Fluent Run (LFR)**
   - **Fluency percentage & stuttering events**
   - **Problematic phonemes** (e.g., "AA: 2 errors")
   - **Clinical insights** (e.g., "Focus on fricative production")
   - Detailed error tables

5. **Continue Session**:
   - "Get Prompt" → Another prompt for same object
   - "New Object" → Switch to different object

### Therapist Workflow (dashboard.html) - NEW

1. **Enter Patient ID** (e.g., "john_doe", "patient123")
2. **Load Dashboard** → Click "Load Patient Data"
3. **View Summary**:
   - Metric cards: Total recordings, avg PER, avg LFR, avg fluency
   - Clinical insights with automated recommendations
4. **Analyze Trends**:
   - LFR trend chart (is fluency improving?)
   - PER trend chart (is pronunciation improving?)
   - Fluency percentage trend
   - Problematic phonemes bar chart
5. **Review Sessions**:
   - Scroll to sessions table
   - Click any row to expand details
   - View transcript, metrics, phoneme errors, stuttering events
6. **Identify Focus Areas**:
   - Check clinical insights panel
   - Review most problematic phonemes
   - Plan next therapy session based on data

---

## 🏗️ Architecture

```
Frontend (HTML/JS)
    ↓
FastAPI Backend
    ↓
┌─────────────────────────────────┐
│  Speech Processing Pipeline     │
├─────────────────────────────────┤
│  1. Whisper ASR                 │
│  2. Word-Level Metrics          │
│  3. Phoneme Analysis ⭐ NEW     │
│  4. Semantic Evaluation         │
└─────────────────────────────────┘
    ↓
SQLite Database
```

### Key Components

| Component | Description |
|-----------|-------------|
| `speech_processing.py` | Main analysis pipeline |
| `phoneme_analysis.py` | PER calculation & error detection |
| `phoneme_lookup.py` | CMU Dict wrapper (134k words) |
| `forced_alignment.py` | Gentle integration for timestamps |
| `semantic_analysis.py` | Sentence transformer evaluation |

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/session/start` | POST | Start new session |
| `/session/next-prompt` | POST | Get next prompt |
| `/session/change-object` | POST | Switch object |
| `/record` | POST | Upload audio & analyze |

### Example Response

```json
{
  "transcript": "this is a bootle",
  "wer": 0.25,
  "speech_rate": 120.5,
  "semantic_evaluation": {
    "classification": "partial",
    "similarity_score": 0.78
  },
  "phoneme_analysis": {
    "per": 0.20,
    "error_count": 1,
    "errors": [
      {
        "type": "substitution",
        "expected": "AH",
        "actual": "UW",
        "word": "bottle",
        "position": 1
      }
    ],
    "problematic_phonemes": {"AH": 1},
    "clinical_notes": [
      "Moderate phoneme errors detected",
      "Consider vowel discrimination exercises"
    ]
  }
}
```

---

## 🧪 Testing

Run unit tests:

```bash
cd backend
pytest tests/test_phoneme_analysis.py -v
```

**Test Coverage**:
- ✓ PER calculation (perfect match, substitutions, deletions, insertions)
- ✓ Phoneme alignment algorithm
- ✓ CMU Dictionary lookups
- ✓ Clinical note generation
- ✓ Error classification

---

## 📁 Project Structure

```
FINAL PROJECT/
├── backend/
│   ├── main.py                       # FastAPI app
│   ├── requirements.txt              # Dependencies
│   ├── routers/
│   │   └── analysis.py               # API endpoints
│   ├── services/
│   │   ├── speech_processing.py      # Main pipeline
│   │   ├── phoneme_analysis.py       # ⭐ Phoneme PER
│   │   ├── phoneme_lookup.py         # ⭐ CMU Dict
│   │   ├── forced_alignment.py       # ⭐ Gentle
│   │   ├── semantic_analysis.py      # Transformers
│   │   └── prompts.py                # Prompt system
│   ├── data/
│   │   └── cmudict-0.7b.txt          # ⭐ 134k words
│   ├── frontend/
│   │   └── recorder.html             # UI
│   └── tests/
│       └── test_phoneme_analysis.py  # ⭐ Unit tests
├── docker-compose.yml                # ⭐ Gentle service
├── PROJECT_DOCUMENTATION.txt         # ⭐ Full docs
└── README.md                         # This file
```

---

## 🔬 Research Applications

This system enables:
- **Phoneme-level error tracking** over time
- **Personalized therapy targets** (focus on problematic sounds)
- **Automated homework assessment**
- **Large-scale aphasia research**
- **Outcome measurement** for clinical trials

### Clinical Use Cases
1. **Aphasia rehabilitation** - Track pronunciation improvement
2. **Dysarthria therapy** - Identify motor speech patterns
3. **Speech sound disorders** - Phonological error analysis
4. **Accent modification** - Phoneme-level feedback

---

## 📚 Documentation

- **[PROJECT_DOCUMENTATION.txt](PROJECT_DOCUMENTATION.txt)** - Comprehensive technical documentation
  - Architecture details
  - Algorithm descriptions
  - Database schema
  - Clinical validation guidelines
  - Research considerations

---

## 🛠️ Dependencies

**Core Libraries**:
- `fastapi` - Web framework
- `faster-whisper` - ASR model
- `sentence-transformers` - Semantic analysis
- `editdistance` - Phoneme alignment
- `jiwer` - WER calculation
- `sqlalchemy` - Database ORM

**Full list**: See [backend/requirements.txt](backend/requirements.txt)

---

## ⚙️ Configuration

### Environment Variables

```bash
# Optional: Custom Gentle URL
export GENTLE_URL=http://localhost:8765

# Optional: Whisper model size
# Edit backend/config.py:
WHISPER_MODEL_SIZE = "small"  # or "tiny", "base"
```

---

## 🔍 Known Limitations

1. **No patient management** - Single-user interface (add auth for production)
2. **Database saves not yet implemented** - Results returned via API only
3. **Gentle is optional** - Works without forced alignment
4. **English only** - CMU Dict is American English
5. **Not FDA-approved** - Research use only

---

## 🚧 Future Enhancements

- [ ] Multi-patient dashboard
- [ ] Progress visualization charts
- [ ] GOP (Goodness of Pronunciation) scores
- [ ] Database persistence implementation
- [ ] Mobile app version
- [ ] Real-time feedback during recording
- [ ] Multi-language support
- [ ] HIPAA compliance features

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

This project uses:
- **CMU Pronouncing Dictionary** (CMU Sphinx)
- **Faster-Whisper** (OpenAI Whisper optimized)
- **Gentle Forced Aligner** (lowerquality/gentle)
- **Sentence Transformers** (UKPLab)

---

## 📧 Contact

For research collaboration or questions:
- GitHub Issues: [Create an issue](issues)
- Email: [your-email@example.com]

---

## 📊 Citation

If you use this system in your research, please cite:

```bibtex
@software{speech_therapy_system_2025,
  title={AI-Powered Speech Therapy Homework System with Phoneme-Level Analysis},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/yourrepo}
}
```

---

## ⭐ Key Innovation

**This system uniquely provides phoneme-level error analysis**, answering the critical clinical question:

> "The patient said 'bootle' instead of 'bottle' - but **which phoneme** was mispronounced, and **what** should we practice?"

**Answer**: `AA → UW` substitution → Vowel discrimination exercises needed.

---

**Built with ❤️ for speech therapy research**

*Last updated: December 22, 2025*
