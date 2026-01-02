# Ollama Setup for Dynamic Prompt Generation

This speech therapy system now supports **dynamic prompt generation** using Ollama - a zero-cost, local LLM solution. You can enter ANY object and the system will automatically generate therapy-appropriate prompts.

## Why Ollama?

- ✅ **Zero Cost** - Runs locally, no API fees
- ✅ **Privacy** - No data leaves your machine
- ✅ **Offline** - Works without internet
- ✅ **Fast** - Lightweight models (3B parameters)
- ✅ **Graceful Fallback** - Works even if Ollama isn't installed

## Installation

### Step 1: Install Ollama

#### Windows:
1. Download from: https://ollama.ai/download
2. Run the installer
3. Ollama will start automatically as a background service

#### Mac:
```bash
brew install ollama
ollama serve  # Start the service
```

#### Linux:
```bash
curl https://ollama.ai/install.sh | sh
ollama serve  # Start the service
```

### Step 2: Download the Model

We recommend using **Llama 3.2 3B** - it's fast, lightweight, and perfect for prompt generation:

```bash
ollama pull llama3.2:3b
```

This downloads a ~2GB model. First download takes a few minutes depending on your internet speed.

### Step 3: Verify Installation

```bash
ollama list
```

You should see:
```
NAME              ID              SIZE
llama3.2:3b       abc123def       1.7 GB
```

### Step 4: Test Ollama API

```bash
curl http://localhost:11434/api/tags
```

If you see JSON output, Ollama is running correctly!

## Usage in Speech Therapy System

### How It Works

1. **Therapist enters any object** (e.g., "toothbrush", "umbrella", "laptop")
2. **System checks cache first** - if prompts were generated before, instant retrieval
3. **If not cached:**
   - Tries to generate with Ollama (takes 3-10 seconds)
   - If Ollama unavailable, falls back to simple template prompts
4. **Prompts are cached** in database for instant reuse

### With Ollama (LLM Mode) 🤖

Example for "toothbrush":
```json
{
  "questions": [
    {"text": "What is this object?", "expected_answers": ["toothbrush", "a toothbrush"]},
    {"text": "What do you use it for?", "expected_answers": ["brushing teeth", "cleaning teeth"]},
    {"text": "Where do you keep your toothbrush?", "expected_answers": ["bathroom", "in the bathroom"]},
    {"text": "When do you use it?", "expected_answers": ["morning", "before bed", "twice a day"]}
  ],
  "sentences": [
    {"text": "I brush my teeth every morning.", "difficulty": "simple"},
    {"text": "The toothbrush is in the bathroom.", "difficulty": "simple"},
    {"text": "I use my toothbrush to keep my teeth clean.", "difficulty": "advanced"}
  ]
}
```

✅ Context-aware, natural, clinically appropriate

### Without Ollama (Fallback Mode) 📝

Same object generates:
```json
{
  "questions": [
    {"text": "What is this object?", "expected_answers": ["toothbrush", "a toothbrush"]},
    {"text": "What do you use it for?", "expected_answers": ["using toothbrush"]},
    {"text": "Can you describe the toothbrush?", "expected_answers": ["it's a toothbrush"]}
  ],
  "sentences": [
    {"text": "This is a toothbrush.", "difficulty": "simple"},
    {"text": "I use the toothbrush.", "difficulty": "simple"}
  ]
}
```

✅ Still works, just less sophisticated

## UI Indicators

The recorder.html page shows Ollama status:

- **🟢 LLM Ready** - Ollama is running, will use AI generation
- **🟡 Fallback Mode** - Ollama not available, using templates

When generating prompts, you'll see:
- **🤖 LLM Generated** - Fresh AI generation
- **💾 From Cache** - Retrieved from database (instant)
- **📝 Template** - Fallback prompts

## Troubleshooting

### "Ollama not available"

**Check if Ollama is running:**
```bash
# Windows
Get-Process ollama

# Mac/Linux
ps aux | grep ollama
```

**Restart Ollama:**
```bash
# Mac/Linux
ollama serve

# Windows - restart the Ollama app
```

### "Request timed out"

The model might be too slow. Try a smaller model:
```bash
ollama pull llama3.2:1b  # Even faster, 1B params
```

Then update `backend/services/llm_prompt_generator.py`:
```python
DEFAULT_MODEL = "llama3.2:1b"
```

### "Connection refused"

Ollama isn't running. Start it:
```bash
ollama serve
```

### Check Backend Logs

Look for these messages in uvicorn output:
```
INFO: Generating prompts for object 'umbrella' using llama3.2:3b
INFO: Successfully generated 5 questions and 4 sentences
INFO: Cached generated prompts for 'umbrella'
```

## Performance

### Generation Speed
- **First time (no cache):** 3-10 seconds depending on CPU
- **Cached retrieval:** <100ms

### Model Comparison
| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| llama3.2:1b | 1.7GB | ⚡⚡⚡ Fast | ⭐⭐⭐ Good |
| llama3.2:3b | 2.0GB | ⚡⚡ Medium | ⭐⭐⭐⭐ Great |
| llama3:8b | 4.7GB | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent |

**Recommended:** llama3.2:3b (best balance)

## API Endpoints

### POST /generate-prompts
Generate prompts for an object.

**Request:**
```bash
curl -X POST http://localhost:8000/generate-prompts \
  -F "object_name=umbrella"
```

**Response:**
```json
{
  "object": "umbrella",
  "questions": [...],
  "sentences": [...],
  "source": "llm",
  "model": "llama3.2:3b"
}
```

### GET /ollama/status
Check if Ollama is available.

**Request:**
```bash
curl http://localhost:8000/ollama/status
```

**Response:**
```json
{
  "available": true,
  "message": "Ollama is running"
}
```

### GET /generated-objects
List all objects with cached prompts.

**Request:**
```bash
curl http://localhost:8000/generated-objects
```

**Response:**
```json
{
  "objects": ["toothbrush", "umbrella", "laptop", "spoon"],
  "count": 4
}
```

## Database Caching

Generated prompts are stored in the `generated_prompts` table:

```sql
CREATE TABLE generated_prompts (
    id INTEGER PRIMARY KEY,
    object_name TEXT UNIQUE NOT NULL,
    prompts_json TEXT NOT NULL,
    model_name TEXT,
    generation_method TEXT,  -- 'llm' or 'fallback'
    created_at TIMESTAMP,
    last_used_at TIMESTAMP
);
```

Benefits:
- **Instant retrieval** for repeated objects
- **Persistent across server restarts**
- **Tracks generation method** (know which were AI-generated)
- **LRU tracking** via last_used_at

## Advanced Configuration

### Change Model

Edit `backend/services/llm_prompt_generator.py`:

```python
DEFAULT_MODEL = "llama3.2:3b"  # Change this
```

Available models:
```bash
ollama list  # See installed models
ollama pull <model-name>  # Install new model
```

### Customize Prompts

Edit the `SYSTEM_PROMPT` in `llm_prompt_generator.py` to adjust:
- Number of questions/sentences
- Difficulty levels
- Prompt types
- Clinical focus areas

### Adjust Timeout

If generation is slow, increase timeout:

```python
generate_prompts_with_ollama(
    object_name,
    timeout=60  # Default is 30 seconds
)
```

## System Requirements

- **Minimum:** 4GB RAM, 2 CPU cores
- **Recommended:** 8GB RAM, 4 CPU cores
- **Disk:** ~2GB for model storage

Ollama will use CPU by default. For GPU acceleration, ensure you have compatible hardware and CUDA/Metal support.

## Benefits Over Hardcoded Prompts

### Before (Hardcoded)
- ❌ Only 5 objects (bottle, cup, phone, book, chair)
- ❌ Fixed prompts, limited variety
- ❌ No customization per patient needs
- ❌ Adding objects requires code changes

### After (Dynamic with LLM)
- ✅ **Unlimited objects** - any common or specialized item
- ✅ **Natural variation** - different phrasings each time
- ✅ **Contextually appropriate** - questions match object function
- ✅ **Zero-cost** - no API fees
- ✅ **Works offline** - no internet needed
- ✅ **Cached for performance** - instant retrieval after first generation
- ✅ **Graceful degradation** - falls back if Ollama unavailable

## Example Workflow

1. **Therapist decides to work on kitchen objects**
2. **Enters "toothbrush"** → Generates 5 questions, 4 sentences (10 seconds)
3. **Patient practices** all prompts
4. **Therapist enters "spoon"** → Generates new prompts (8 seconds)
5. **Next session:** Enters "toothbrush" again → Instant (cached)

## FAQ

**Q: Do I need Ollama installed?**
A: No! The system works without it using fallback prompts. Ollama just makes it better.

**Q: Can I use GPT/Claude instead?**
A: Yes, but they cost money. Ollama is free and works offline.

**Q: How much does Ollama cost?**
A: Zero. Completely free and open-source.

**Q: Is my patient data sent to Ollama servers?**
A: No. Ollama runs locally. Nothing leaves your machine.

**Q: Can I use this in production/clinic?**
A: Yes! It's designed for clinical use with HIPAA-compliant local processing.

---

For issues or questions, check the backend logs or create a GitHub issue.
