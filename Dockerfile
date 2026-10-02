# CPU serving image for the FastAPI app.
FROM python:3.11-slim

# ffmpeg: decode mp3/ogg/m4a uploads. libsndfile1: soundfile wheels need it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/

# MODEL_PATH must point at a mounted/baked-in checkpoint dir.
ENV MODEL_PATH=/app/models/whisper-small-ur

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
