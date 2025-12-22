# routers/analysis.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from utils.file_utils import save_audio_file
from services.prompts import get_random_prompt, OBJECT_PROMPTS
from services.speech_processing import analyze_speech
import uuid
import os
import random
from typing import Dict

router = APIRouter()

# In-memory session storage (in production, use Redis or database)
sessions: Dict[str, Dict] = {}

@router.get("/prompt")
def get_prompt():
    obj, data = get_random_prompt()
    return {"object": obj, "prompts": data}

@router.post("/session/start")
def start_session(object_name: str = None):
    """
    Start a new session and return a session_id and a prompt to the client.
    If object_name is provided, use that object; otherwise pick randomly.
    """
    from services.prompts import get_expected_answer

    session_id = str(uuid.uuid4())

    # If no object specified, pick randomly
    if not object_name:
        object_name = random.choice(list(OBJECT_PROMPTS.keys()))
    elif object_name not in OBJECT_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unknown object: {object_name}")

    # Initialize session data
    sessions[session_id] = {
        "object": object_name,
        "used_prompts": [],
        "prompt_count": 0
    }

    # Get first prompt for this object
    return get_next_prompt(session_id)


@router.post("/session/next-prompt")
def get_next_prompt(session_id: str = Form(...)):
    """
    Get the next prompt for the current object in the session.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    obj = session["object"]
    data = OBJECT_PROMPTS[obj]

    # Get all available prompts (questions + sentences)
    all_prompts = []

    if data.get("questions"):
        all_prompts.extend([(q, "question") for q in data["questions"]])
    if data.get("sentences"):
        all_prompts.extend([(s, "sentence") for s in data["sentences"]])

    # Filter out already used prompts
    available_prompts = [p for p in all_prompts if p[0] not in session["used_prompts"]]

    # If all prompts used, reset the used list
    if not available_prompts:
        session["used_prompts"] = []
        available_prompts = all_prompts

    # Pick a random available prompt
    prompt_text, prompt_type = random.choice(available_prompts)
    session["used_prompts"].append(prompt_text)
    session["prompt_count"] += 1

    # Get expected answer
    if prompt_type == "question":
        expected_answer = data["expected_answers"].get(prompt_text, "")
    else:
        expected_answer = data["expected_sentence_repetitions"].get(prompt_text, prompt_text)

    return {
        "session_id": session_id,
        "object": obj,
        "prompt_text": prompt_text,
        "expected_answer": expected_answer,
        "prompt_number": session["prompt_count"],
        "total_prompts_available": len(all_prompts)
    }


@router.post("/session/change-object")
def change_object(session_id: str = Form(...), new_object: str = Form(None)):
    """
    Change the object for the current session.
    If new_object is not specified, picks a different random object.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    current_object = session["object"]

    if new_object:
        if new_object not in OBJECT_PROMPTS:
            raise HTTPException(status_code=400, detail=f"Unknown object: {new_object}")
        if new_object == current_object:
            raise HTTPException(status_code=400, detail="Already using this object")
    else:
        # Pick a different random object
        available_objects = [obj for obj in OBJECT_PROMPTS.keys() if obj != current_object]
        new_object = random.choice(available_objects)

    # Reset session for new object
    session["object"] = new_object
    session["used_prompts"] = []
    session["prompt_count"] = 0

    # Get first prompt for new object
    return get_next_prompt(session_id)

@router.post("/record")
async def receive_recording(
    session_id: str = Form(...),
    object_name: str = Form(...),
    prompt_text: str = Form(...),
    expected_answer: str = Form(...),
    audio: UploadFile = File(...)
):
    """
    Receives a recorded blob from client (MediaRecorder output),
    saves it, runs analyze_speech and returns JSON result.

    Args:
        session_id: Unique session identifier
        object_name: Name of the object being discussed
        prompt_text: The question/prompt that was asked
        expected_answer: Expected answer for semantic evaluation
        audio: Audio file upload
    """
    # read bytes
    file_bytes = await audio.read()
    # create a filename prefix using session_id
    filename_prefix = f"{session_id}"
    audio_path = save_audio_file(
        file_bytes,
        filename_prefix=filename_prefix,
        filename=audio.filename
    )

    # Prepare prompt data for analysis
    prompt_data = {
        "object_name": object_name,
        "prompt_text": prompt_text,
        "expected_answer": expected_answer
    }

    # Analyze with semantic evaluation
    result = analyze_speech(audio_path, prompt_data)

    # Add session info to result
    result["session_id"] = session_id
    result["object"] = object_name
    result["prompt"] = prompt_text

    # Optionally: save to DB here (not implemented in skeleton)
    return result
