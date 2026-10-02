"""Offline inference helper shared by the FastAPI service and the Gradio demo.

Loads a Whisper model + processor once, then transcribes 16 kHz audio files.

Usage:
    python src/inference.py --model ./models/whisper-small-ur --audio sample.wav
"""

import argparse
import os
import sys
import time

# Allow `python src/inference.py` from the repo root to import src/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.normalize import normalize_urdu

SAMPLE_RATE = 16000


class UrduASR:
    """Thin wrapper around a Whisper model for Urdu transcription."""

    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        # Ask the decoder for Urdu transcription explicitly (works for both
        # fine-tuned checkpoints and base Whisper models).
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="ur", task="transcribe"
        )

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe an audio file.

        Returns dict with raw transcript, normalized transcript and latency_ms.
        """
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        input_features = self.processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.device)

        started = time.perf_counter()
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=self.forced_decoder_ids,
                max_new_tokens=225,
            )
        latency_ms = (time.perf_counter() - started) * 1000

        raw = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return {
            "transcript": raw.strip(),
            "transcript_normalized": normalize_urdu(raw),
            "latency_ms": round(latency_ms, 1),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio file (Urdu ASR)")
    parser.add_argument("--model", type=str, required=True, help="Model dir or HF id")
    parser.add_argument("--audio", type=str, required=True, help="WAV/MP3/OGG file path")
    args = parser.parse_args()

    asr = UrduASR(args.model)
    result = asr.transcribe(args.audio)
    print(f"[{result['latency_ms']} ms] {result['transcript']}")
    print(f"normalized: {result['transcript_normalized']}")


if __name__ == "__main__":
    main()
