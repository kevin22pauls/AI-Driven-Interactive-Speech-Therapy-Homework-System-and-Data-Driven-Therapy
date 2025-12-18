import random

OBJECT_PROMPTS = {
    "bottle": {
        "questions": [
            "What is this object?",
            "What color is the bottle?",
            "What do you drink from this?"
        ],
        "sentences": [
            "I have a bottle.",
            "Drink more water.",
            "The bottle is on the table."
        ]
    },
    "cup": {
        "questions": [
            "What is this?",
            "What do you use the cup for?"
        ],
        "sentences": [
            "This is a cup.",
            "The cup is full of tea."
        ]
    }
}

def get_random_prompt():
    obj = random.choice(list(OBJECT_PROMPTS.keys()))
    data = OBJECT_PROMPTS[obj]
    return obj, data
