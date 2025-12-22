import random
from typing import Dict, List, Any, Tuple

OBJECT_PROMPTS = {
    "bottle": {
        "questions": [
            "What is this object?",
            "What color is the bottle?",
            "What do you drink from this?"
        ],
        "expected_answers": {
            "What is this object?": "bottle",
            "What color is the bottle?": "variable",  # Any color is acceptable
            "What do you drink from this?": "water, juice, or other beverages"
        },
        "sentences": [
            "I have a bottle.",
            "Drink more water.",
            "The bottle is on the table."
        ],
        "expected_sentence_repetitions": {
            "I have a bottle.": "I have a bottle.",
            "Drink more water.": "Drink more water.",
            "The bottle is on the table.": "The bottle is on the table."
        }
    },
    "cup": {
        "questions": [
            "What is this?",
            "What do you use the cup for?",
            "What color is the cup?"
        ],
        "expected_answers": {
            "What is this?": "cup",
            "What do you use the cup for?": "drinking beverages like water, coffee, or tea",
            "What color is the cup?": "variable"  # Any color is acceptable
        },
        "sentences": [
            "This is a cup.",
            "The cup is full of tea.",
            "I drink coffee from the cup."
        ],
        "expected_sentence_repetitions": {
            "This is a cup.": "This is a cup.",
            "The cup is full of tea.": "The cup is full of tea.",
            "I drink coffee from the cup.": "I drink coffee from the cup."
        }
    },
    "phone": {
        "questions": [
            "What is this object?",
            "What do you use a phone for?",
            "What color is the phone?"
        ],
        "expected_answers": {
            "What is this object?": "phone",
            "What do you use a phone for?": "making calls, texting, or communicating with people",
            "What color is the phone?": "variable"  # Any color is acceptable
        },
        "sentences": [
            "The phone is ringing.",
            "I need to charge my phone.",
            "Can you call me on the phone?"
        ],
        "expected_sentence_repetitions": {
            "The phone is ringing.": "The phone is ringing.",
            "I need to charge my phone.": "I need to charge my phone.",
            "Can you call me on the phone?": "Can you call me on the phone?"
        }
    },
    "book": {
        "questions": [
            "What is this?",
            "What do you do with a book?",
            "What is the book about?"
        ],
        "expected_answers": {
            "What is this?": "book",
            "What do you do with a book?": "read it to learn or enjoy stories",
            "What is the book about?": "variable"  # Any topic is acceptable
        },
        "sentences": [
            "I love reading books.",
            "The book is on the shelf.",
            "This book has many pictures."
        ],
        "expected_sentence_repetitions": {
            "I love reading books.": "I love reading books.",
            "The book is on the shelf.": "The book is on the shelf.",
            "This book has many pictures.": "This book has many pictures."
        }
    },
    "chair": {
        "questions": [
            "What is this object?",
            "What do you use a chair for?",
            "How many legs does the chair have?"
        ],
        "expected_answers": {
            "What is this object?": "chair",
            "What do you use a chair for?": "sitting down or resting",
            "How many legs does the chair have?": "variable"  # Could be 3, 4, or other
        },
        "sentences": [
            "Please sit in the chair.",
            "The chair is comfortable.",
            "I moved the chair to the table."
        ],
        "expected_sentence_repetitions": {
            "Please sit in the chair.": "Please sit in the chair.",
            "The chair is comfortable.": "The chair is comfortable.",
            "I moved the chair to the table.": "I moved the chair to the table."
        }
    }
}

def get_random_prompt() -> Tuple[str, Dict[str, Any]]:
    """
    Get a random object and its associated prompt data.

    Returns:
        Tuple of (object_name, prompt_data)
    """
    obj = random.choice(list(OBJECT_PROMPTS.keys()))
    data = OBJECT_PROMPTS[obj]
    return obj, data


def get_expected_answer(object_name: str, prompt_text: str) -> str:
    """
    Get the expected answer for a specific object and prompt.

    Args:
        object_name: Name of the object
        prompt_text: The question or sentence prompt

    Returns:
        Expected answer string, or empty string if not found
    """
    if object_name not in OBJECT_PROMPTS:
        return ""

    prompt_data = OBJECT_PROMPTS[object_name]

    # Check in expected_answers for questions
    if prompt_text in prompt_data.get("expected_answers", {}):
        return prompt_data["expected_answers"][prompt_text]

    # Check in expected_sentence_repetitions for sentences
    if prompt_text in prompt_data.get("expected_sentence_repetitions", {}):
        return prompt_data["expected_sentence_repetitions"][prompt_text]

    return ""


def get_question_type(prompt_text: str) -> str:
    """
    Determine the type of question being asked.

    Args:
        prompt_text: The question text

    Returns:
        Question type: "identification", "functional", "descriptive"
    """
    prompt_lower = prompt_text.lower()

    if "what is this" in prompt_lower or "what is it" in prompt_lower:
        return "identification"
    elif "what do you use" in prompt_lower or "used for" in prompt_lower:
        return "functional"
    elif "color" in prompt_lower or "how many" in prompt_lower or "about" in prompt_lower:
        return "descriptive"
    else:
        return "general"
