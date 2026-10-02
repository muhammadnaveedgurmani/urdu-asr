"""Benchmark: base vs fine-tuned vs large — honest WER comparison.

Evaluates three setups on the held-out test split and reports WER both on raw
model output and on normalized text (see src/normalize.py), so orthographic
noise does not inflate the error rates:

  1. openai/whisper-small            (zero-shot baseline)
  2. <fine-tuned checkpoint>         (this project's model)
  3. openai/whisper-large            (zero-shot skyline, sampled subset only —
                                      large is slow on CPU, hence --large_max_samples)

Usage:
    python src/evaluate.py --data_dir ./data/cv_ur \\
        --finetuned_model ./models/whisper-small-ur \\
        --output results/wer_results.json --max_samples 200 --large_max_samples 25
"""

import argparse
import json
import os
import sys
import tempfile

# Allow `python src/evaluate.py` from the repo root to import src/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
from datasets import load_from_disk
from jiwer import wer as jiwer_wer

from src.inference import SAMPLE_RATE, UrduASR
from src.normalize import normalize_urdu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WER benchmark for Urdu ASR models")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Prepared DatasetDict from src/data_prep.py")
    parser.add_argument("--finetuned_model", type=str, required=True,
                        help="Fine-tuned checkpoint dir")
    parser.add_argument("--output", type=str, default="results/wer_results.json",
                        help="Where to save the results JSON")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Test utterances for small/base + fine-tuned models")
    parser.add_argument("--large_max_samples", type=int, default=25,
                        help="Subset for whisper-large (slow on CPU)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_wav(array: np.ndarray, path: str) -> None:
    sf.write(path, array, SAMPLE_RATE)


def evaluate_model(asr: UrduASR, audios: list, name: str) -> dict:
    """Transcribe a list of waveforms; return hypotheses + mean latency."""
    hypotheses, latencies = [], []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, waveform in enumerate(audios):
            wav_path = os.path.join(tmpdir, f"sample_{i}.wav")
            _write_wav(np.asarray(waveform, dtype=np.float32), wav_path)
            result = asr.transcribe(wav_path)
            hypotheses.append(result["transcript"])
            latencies.append(result["latency_ms"])
    print(f"  {name}: {len(hypotheses)} samples, "
          f"mean latency {sum(latencies) / max(len(latencies), 1):.0f} ms")
    return {"hypotheses": hypotheses,
            "mean_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1)}


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading test split from {args.data_dir} ...")
    test_ds = load_from_disk(args.data_dir)["test"]
    idx = rng.choice(len(test_ds), size=min(args.max_samples, len(test_ds)), replace=False)
    subset = test_ds.select(sorted(idx.tolist()))
    references = subset["sentence"]          # already normalized in data_prep
    audios = [ex["array"] for ex in subset["audio"]]

    results: dict = {}

    # 1 & 2: base small and fine-tuned on the full subset.
    for name, model_id in [
        ("whisper-small (zero-shot)", "openai/whisper-small"),
        ("whisper-small-ur (fine-tuned)", args.finetuned_model),
    ]:
        print(f"Evaluating {name} ...")
        asr = UrduASR(model_id)
        out = evaluate_model(asr, audios, name)
        results[name] = {
            "n_samples": len(audios),
            "wer_raw": round(100 * jiwer_wer(references, out["hypotheses"]), 2),
            "wer_normalized": round(100 * jiwer_wer(
                [normalize_urdu(r) for r in references],
                [normalize_urdu(h) for h in out["hypotheses"]],
            ), 2),
            "mean_latency_ms": out["mean_latency_ms"],
        }
        del asr  # free VRAM/RAM before loading the next model

    # 3: large on a small subset (skyline reference).
    k = min(args.large_max_samples, len(audios))
    print(f"Evaluating whisper-large (zero-shot) on {k} samples ...")
    asr_large = UrduASR("openai/whisper-large")
    out = evaluate_model(asr_large, audios[:k], "whisper-large (zero-shot)")
    results["whisper-large (zero-shot)"] = {
        "n_samples": k,
        "wer_raw": round(100 * jiwer_wer(references[:k], out["hypotheses"]), 2),
        "wer_normalized": round(100 * jiwer_wer(
            [normalize_urdu(r) for r in references[:k]],
            [normalize_urdu(h) for h in out["hypotheses"]],
        ), 2),
        "mean_latency_ms": out["mean_latency_ms"],
    }

    # --- Report ---
    header = f"{'model':38} {'n':>5} {'WER raw':>9} {'WER norm':>9} {'lat ms':>8}"
    print("\n" + header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:38} {m['n_samples']:>5} {m['wer_raw']:>8.2f}% "
              f"{m['wer_normalized']:>8.2f}% {m['mean_latency_ms']:>8.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
