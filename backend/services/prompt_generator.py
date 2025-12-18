import random
from data.objects import OBJECTS

PROMPT_TEMPLATES = [
    "What is this object?",
    "What is this used for?",
    "What color is this?",
    "Can you say: I have a {object}?",
    "Can you say: This {object} is useful?"
]

def generate_prompt():
    obj = random.choice(list(OBJECTS.keys()))
    template = random.choice(PROMPT_TEMPLATES)

    prompt_text = template.format(object=obj)

    return {
        "object": obj,
        "expected_keywords": [obj],
        "prompt_text": prompt_text
    }
