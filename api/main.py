"""FastAPI serving layer for the fine-tuned Urdu ASR model.

Endpoints:
  GET  /health      -> {"status": "ok", "model": MODEL_PATH}
  POST /transcribe  -> multipart audio upload -> JSON with transcript,
                       normalized transcript and latency.

The model is loaded once at startup. Point MODEL_PATH at the fine-tuned
checkpoint (default ./models/whisper-small-ur) or any Whisper model id.

Run:
    MODEL_PATH=./models/whisper-small-ur uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import tempfile
from contextlib import asynccontextmanager

# Allow running from the repo root to import src/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.inference import UrduASR

MODEL_PATH = os.environ.get("MODEL_PATH", "./models/whisper-small-ur")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB — plenty for a 30s 16kHz clip

asr: UrduASR | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr
    print(f"Loading ASR model from {MODEL_PATH} ...")
    try:
        asr = UrduASR(MODEL_PATH)
    except Exception as exc:  # fail fast with a clear message, not a cryptic 500
        raise RuntimeError(f"Could not load model from {MODEL_PATH}: {exc}") from exc
    print("Model loaded.")
    yield
    asr = None


app = FastAPI(title="Lightweight Urdu ASR", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH, "loaded": asr is not None}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> JSONResponse:
    if asr is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data)} bytes > {MAX_UPLOAD_BYTES})",
        )

    # Keep the original extension so librosa/soundfile picks the right decoder.
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        result = asr.transcribe(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return JSONResponse({
        "transcript": result["transcript"],
        "transcript_normalized": result["transcript_normalized"],
        "latency_ms": result["latency_ms"],
        "model": MODEL_PATH,
    })
