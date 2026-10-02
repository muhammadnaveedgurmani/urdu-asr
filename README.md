# Lightweight Urdu ASR

Fine-tuning **OpenAI Whisper-small** on Mozilla Common Voice (Urdu) for accurate,
CPU-deployable Urdu speech recognition — with honest WER benchmarks, Urdu text
normalization, a FastAPI serving API, and a Gradio demo.

## Problem statement

State-of-the-art speech models keep getting bigger. Whisper-large is accurate but
needs a GPU with 10 GB+ VRAM, which rules it out for budget hardware, campus labs,
and public-sector deployments in developing countries. Meanwhile, the Urdu
performance of *small* Whisper models is little-studied: Urdu is structurally
under-represented in LLM/ASR training data (poor vocabulary coverage and
encoding), and low-resource languages rarely get the benchmarking attention that
English does.

This project asks a practical question: **how far can a 244M-parameter
Whisper-small go on Urdu when fine-tuned on community speech data — and can it
run on a plain CPU?**

## Architecture

```
                        ┌─────────────────────────┐
                        │  Mozilla Common Voice   │
                        │  (Urdu subset, 16 kHz)  │
                        └────────────┬────────────┘
                                     │ src/data_prep.py
                                     ▼
                        ┌─────────────────────────┐
                        │  Urdu text normalization│  src/normalize.py
                        │  (tashkeel, alef/yeh/   │
                        │   kaf variants, kashida)│
                        └────────────┬────────────┘
                                     │ Seq2SeqTrainer (Colab T4)
                                     ▼
                        ┌─────────────────────────┐
                        │  Whisper-small fine-tune│  src/train.py
                        │  forced <|ur|> decoder  │
                        │  prompt, WER metric     │
                        └────────────┬────────────┘
                                     ▼
              ┌──────────────────────┴──────────────────────┐
              │                                             │
   ┌──────────▼──────────┐                       ┌──────────▼──────────┐
   │  WER benchmark      │                       │  Serving            │
   │  base vs fine-tuned │                       │  FastAPI /transcribe│
   │  vs large (skyline) │                       │  Gradio demo        │
   │  src/evaluate.py    │                       │  api/  demo/        │
   └─────────────────────┘                       └─────────────────────┘
```

## Repository structure

```
urdu-asr/
├── src/
│   ├── normalize.py      # Urdu text normalization (diacritics, variants, kashida)
│   ├── data_prep.py      # Common Voice Urdu -> 16kHz, normalized, saved DatasetDict
│   ├── train.py          # Whisper-small fine-tune (Seq2SeqTrainer, WER metric)
│   ├── evaluate.py       # base vs fine-tuned vs large WER benchmark
│   └── inference.py      # UrduASR class: load model, transcribe audio file
├── api/main.py           # FastAPI service (POST /transcribe, GET /health)
├── demo/app.py           # Gradio demo (mic + upload)
├── configs/training_config.yaml
├── COLAB_TRAINING.md     # step-by-step GPU training guide
├── Dockerfile
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/muhammadnaveedgurmani/urdu-asr.git
cd urdu-asr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

System dependency for audio decoding: `ffmpeg` (`sudo apt install ffmpeg`).

## Training (Google Colab, free T4)

This repo has no GPU requirement for inference, but fine-tuning needs one.
Full walkthrough: [COLAB_TRAINING.md](COLAB_TRAINING.md). Short version:

```bash
# 1. Accept the Common Voice terms at
#    https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
#    then: huggingface-cli login
python src/data_prep.py --output_dir ./data/cv_ur
python src/train.py --config configs/training_config.yaml --data_dir ./data/cv_ur
```

## Evaluation methodology

WER is computed with [jiwer](https://github.com/jitsi/jiwer) in two modes:

- **WER (raw)** — straight model output vs reference.
- **WER (normalized)** — both sides passed through `src/normalize.py` first, so
  the score punishes real transcription errors, not orthographic noise
  (Arabic vs Urdu kaf, stray diacritics, kashida).

Three setups are compared on the same held-out test utterances:

| Setup | Role |
|---|---|
| `openai/whisper-small` (zero-shot) | baseline |
| `whisper-small-ur` (fine-tuned) | this project's model |
| `openai/whisper-large` (zero-shot, 25-sample subset) | accuracy skyline |

```bash
python src/evaluate.py --data_dir ./data/cv_ur \
    --finetuned_model ./models/whisper-small-ur \
    --output results/wer_results.json
```

## Results

*To be filled after the Colab training run (`src/evaluate.py` prints this table).*

| Model | n | WER (raw) | WER (normalized) | Mean latency (CPU) |
|---|---|---|---|---|
| whisper-small (zero-shot) | TBD | TBD | TBD | TBD |
| whisper-small-ur (fine-tuned) | TBD | TBD | TBD | TBD |
| whisper-large (zero-shot) | TBD | TBD | TBD | TBD |

## API usage

```bash
MODEL_PATH=./models/whisper-small-ur uvicorn api.main:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@sample_urdu.wav" | python -m json.tool
```

Response:

```json
{
  "transcript": "میں نے کتاب پڑھی",
  "transcript_normalized": "میں نے کتاب پڑھی",
  "latency_ms": 812.4,
  "model": "./models/whisper-small-ur"
}
```

Docker:

```bash
docker build -t urdu-asr .
docker run -p 8000:8000 -v $(pwd)/models:/app/models urdu-asr
```

## Demo

```bash
MODEL_PATH=./models/whisper-small-ur python demo/app.py
# Gradio UI at http://localhost:7860 — mic recording or file upload
```

## Roadmap

- [ ] QLoRA fine-tune of Whisper-small for comparison with full fine-tuning
- [ ] ONNX / CTranslate2 export for sub-second CPU inference
- [ ] Streaming transcription (chunked long-form audio)
- [ ] Roman-Urdu output mode (transliteration head)
- [ ] Broader eval: Common Voice validated split + FLEURS Urdu

## References

- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper), 2022.
- Mozilla Common Voice dataset: https://commonvoice.mozilla.org
- UrduLLaMA 1.0 — continual pretraining analysis of Urdu under-representation in LLMs.
- Arif et al., 2025 — first conversational Urdu ASR benchmark; Whisper vs SeamlessM4T; Urdu text-normalization need.
- *Assessing the feasibility of lightweight Whisper models for low-resource Urdu transcription*, 2025.
