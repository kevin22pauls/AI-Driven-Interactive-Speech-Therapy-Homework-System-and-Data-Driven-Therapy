"""
LLM-based dynamic prompt generation for speech therapy.
Uses Ollama with local models (zero-cost solution).
"""

import requests
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Ollama API endpoint (default local installation)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"  # Lightweight model for prompt generation

SYSTEM_PROMPT = """You are a speech therapy assistant helping therapists create exercises for aphasia patients.

Your task: Generate speech therapy prompts for a given object.

Requirements:
- Keep language simple and clear (appropriate for aphasia patients)
- Use common, everyday vocabulary
- Generate multiple valid expected answers for each question
- Vary sentence complexity (short for fluency building, longer for advanced practice)

For each object, generate:
1. Identification questions (e.g., "What is this object?")
2. Functional questions (e.g., "What do you use it for?")
3. Descriptive questions (e.g., "What color is it?", "Where do you keep it?")
4. Contextual questions (e.g., "When do you use it?")
5. Repetition sentences (simple statements for the patient to repeat)

Format your response as valid JSON with this exact structure:
{
  "questions": [
    {
      "text": "What is this object?",
      "type": "identification",
      "expected_answers": ["<object>", "a <object>", "it's a <object>"]
    },
    {
      "text": "What do you use it for?",
      "type": "functional",
      "expected_answers": ["<usage 1>", "<usage 2>"]
    }
  ],
  "sentences": [
    {
      "text": "I use my <object> every day.",
      "difficulty": "simple"
    },
    {
      "text": "<longer contextual sentence>",
      "difficulty": "advanced"
    }
  ]
}

Generate 4-5 questions and 3-4 sentences. Make them natural and clinically appropriate."""


def generate_prompts_with_ollama(
    object_name: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 30
) -> Optional[Dict]:
    """
    Generate speech therapy prompts for a given object using Ollama.

    Args:
        object_name: The object to generate prompts for (e.g., "toothbrush", "umbrella")
        model: Ollama model to use (default: llama3.2:3b)
        timeout: Request timeout in seconds

    Returns:
        Dictionary with generated questions and sentences, or None if generation fails

    Example return:
        {
            "questions": [
                {"text": "What is this?", "type": "identification", "expected_answers": ["umbrella", "an umbrella"]},
                {"text": "What do you use it for?", "type": "functional", "expected_answers": ["rain protection", "staying dry"]}
            ],
            "sentences": [
                {"text": "I use my umbrella when it rains.", "difficulty": "simple"}
            ]
        }
    """
    user_prompt = f"Generate speech therapy prompts for this object: {object_name}"

    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant: Here are the speech therapy prompts in JSON format:"

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "temperature": 0.7,  # Slight creativity but mostly consistent
        "top_p": 0.9
    }

    try:
        logger.info(f"Generating prompts for object '{object_name}' using {model}")

        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=timeout
        )

        if response.status_code != 200:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return None

        result = response.json()
        generated_text = result.get("response", "")

        # Extract JSON from response (LLM might add extra text)
        generated_text = generated_text.strip()

        # Try to find JSON block
        if "```json" in generated_text:
            # Extract JSON from markdown code block
            start = generated_text.find("```json") + 7
            end = generated_text.find("```", start)
            json_text = generated_text[start:end].strip()
        elif generated_text.startswith("{"):
            # Already valid JSON
            json_text = generated_text
        else:
            # Try to find first { and last }
            start = generated_text.find("{")
            end = generated_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_text = generated_text[start:end]
            else:
                logger.error(f"Could not extract JSON from LLM response: {generated_text}")
                return None

        # Parse JSON
        prompts_data = json.loads(json_text)

        # Validate structure
        if "questions" not in prompts_data or "sentences" not in prompts_data:
            logger.error(f"Invalid prompt structure: {prompts_data}")
            return None

        logger.info(f"Successfully generated {len(prompts_data['questions'])} questions and {len(prompts_data['sentences'])} sentences")

        return prompts_data

    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to Ollama. Make sure Ollama is running (ollama serve)")
        return None
    except requests.exceptions.Timeout:
        logger.error(f"Ollama request timed out after {timeout}s")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Raw response: {generated_text}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during prompt generation: {e}")
        return None


def generate_fallback_prompts(object_name: str) -> Dict:
    """
    Generate basic fallback prompts if LLM is unavailable.
    Simple template-based generation for reliability.

    Args:
        object_name: The object name

    Returns:
        Dictionary with basic questions and sentences
    """
    return {
        "questions": [
            {
                "text": "What is this object?",
                "type": "identification",
                "expected_answers": [object_name, f"a {object_name}", f"it's a {object_name}"]
            },
            {
                "text": "What do you use it for?",
                "type": "functional",
                "expected_answers": [f"using {object_name}", f"to use {object_name}"]
            },
            {
                "text": f"Can you describe the {object_name}?",
                "type": "descriptive",
                "expected_answers": [f"it's a {object_name}", f"the {object_name} is"]
            }
        ],
        "sentences": [
            {
                "text": f"This is a {object_name}.",
                "difficulty": "simple"
            },
            {
                "text": f"I use the {object_name}.",
                "difficulty": "simple"
            },
            {
                "text": f"The {object_name} is useful.",
                "difficulty": "simple"
            }
        ]
    }


def check_ollama_status() -> bool:
    """
    Check if Ollama is running and accessible.

    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False
