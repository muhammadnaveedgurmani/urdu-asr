"""Gradio demo for Lightweight Urdu ASR.

Runs the model in-process (no API server needed): record from the microphone
or upload an audio file, get back the transcript, the normalized transcript,
and the inference latency.

Run from the repo root:
    MODEL_PATH=./models/whisper-small-ur python demo/app.py
"""

import os
import sys

import gradio as gr

# Allow `python demo/app.py` from the repo root to import src/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import UrduASR  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", "./models/whisper-small-ur")

print(f"Loading ASR model from {MODEL_PATH} ...")
asr = UrduASR(MODEL_PATH)
print("Model loaded.")


def transcribe_audio(audio_path: str):
    """Gradio handler: filepath in -> (transcript, normalized, latency)."""
    if not audio_path:
        return "", "", "No audio provided."
    try:
        result = asr.transcribe(audio_path)
    except Exception as exc:  # surface decode errors in the UI, not a traceback
        return "", "", f"Error: {exc}"
    return (
        result["transcript"],
        result["transcript_normalized"],
        f"{result['latency_ms']} ms",
    )


demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath",
                    label="Record or upload Urdu speech"),
    outputs=[
        gr.Textbox(label="Transcript"),
        gr.Textbox(label="Transcript (normalized)"),
        gr.Textbox(label="Latency"),
    ],
    title="Lightweight Urdu ASR — Whisper-small fine-tuned on Common Voice",
    description=(
        "A CPU-friendly Urdu speech recognizer. Speak or upload up to ~30 seconds "
        "of Urdu audio and get an instant transcript."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
