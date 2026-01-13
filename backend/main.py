from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from routers import analysis
from models.whisper_model import load_whisper_model
from database import schema
import os

app = FastAPI(title="Speech Therapy AI Backend")

# 🔴 CORS MUST BE HERE — IMMEDIATELY AFTER app CREATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # DEV ONLY (fixes your error)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    schema.init_db()
    load_whisper_model()
    print("Backend ready")

app.include_router(analysis.router)

@app.get("/")
def read_root():
    """Serve the frontend HTML file."""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "recorder.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    else:
        return {"message": "Frontend not found. Visit /docs for API documentation."}

@app.get("/recorder")
def read_recorder():
    """Serve the recorder HTML file."""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "recorder.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    else:
        return {"message": "Recorder page not found."}

@app.get("/dashboard")
def read_dashboard():
    """Serve the dashboard HTML file."""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "dashboard.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    else:
        return {"message": "Dashboard page not found."}
