# routers/analysis.py
from fastapi import APIRouter, UploadFile, File, Form
from utils.file_utils import save_audio_file
from services.prompts import get_random_prompt
from services.speech_processing import analyze_speech
import uuid
import os

router = APIRouter()

@router.get("/prompt")
def get_prompt():
    obj, data = get_random_prompt()
    return {"object": obj, "prompts": data}

@router.post("/session/start")
def start_session():
    """
    Start a new session and return a session_id and a prompt to the client.
    Client will then record and send audio to /record with session_id.
    """
    session_id = str(uuid.uuid4())
    obj, data = get_random_prompt()
    # choose one random question or sentence to ask (we return one prompt)
    # prefer question first else sentence
    if data.get("questions"):
        prompt_text = data["questions"][0]
    else:
        prompt_text = data["sentences"][0]

    return {
        "session_id": session_id,
        "object": obj,
        "prompt_text": prompt_text
    }

@router.post("/record")
async def receive_recording(
    session_id: str = Form(...),
    expected_text: str = Form(""),
    audio: UploadFile = File(...)
):
    """
    Receives a recorded blob from client (MediaRecorder output),
    saves it, runs analyze_speech and returns JSON result.
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



    # analyze
    result = analyze_speech(audio_path, expected_text)

    # Optionally: save to DB here (not implemented in skeleton)
    return result
