from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import analysis
from models.whisper_model import load_whisper_model
from database import schema

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
