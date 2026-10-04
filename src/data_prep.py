"""Prepare Urdu Common Voice for Whisper fine-tuning.

Pipeline:
  1. Load the processed Urdu Common Voice dataset from Hugging Face Hub.
  2. Resample all audio to 16 kHz (Whisper's native sample rate).
  3. Normalize transcripts with src/normalize.py.
  4. Drop empty transcripts and clips longer than --max_seconds.
  5. Save a Hugging Face DatasetDict to disk for src/train.py.

Dataset: UmarRamzan/common-voice-urdu-processed (public, no gating).
Note: mozilla-foundation/common_voice_17_0 was removed from HF in Oct 2025
(Mozilla Data Collective only now), so we use this processed mirror.

Usage:
    python src/data_prep.py --output_dir ./data/cv_ur --max_train_samples 2000
"""

import argparse
import os
import sys

# Allow `python src/data_prep.py` from the repo root: `src/` is put on
# sys.path by the interpreter, so add the repo root for `src.*` imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Audio, DatasetDict, load_dataset

from src.normalize import normalize_urdu

DATASET_ID = "UmarRamzan/common-voice-urdu-processed"
SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Common Voice Urdu data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save the prepared DatasetDict")
    parser.add_argument("--max_seconds", type=float, default=30.0,
                        help="Drop clips longer than this (Whisper window is 30s)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Optional cap on train rows (quick experiments)")
    parser.add_argument("--max_test_samples", type=int, default=500,
                        help="Cap on test rows kept for evaluation")
    parser.add_argument("--num_proc", type=int, default=4,
                        help="Parallel workers for map/filter")
    return parser.parse_args()


def _duration_hours(dataset) -> float:
    """Total audio duration in hours (dataset must have 16kHz audio)."""
    total_samples = sum(len(clip["array"]) for clip in dataset["audio"])
    return total_samples / SAMPLE_RATE / 3600


def main() -> None:
    args = parse_args()

    print(f"Loading {DATASET_ID} ...")
    # The processed mirror has train/test splits, no language config needed.
    # token=True reuses the saved `hf auth login` credential.
    raw = DatasetDict({
        "train": load_dataset(DATASET_ID, split="train", token=True),
        "test": load_dataset(DATASET_ID, split="test", token=True),
    })

    # 1. Resample to 16 kHz on the fly.
    raw = raw.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    # 2. Normalize transcripts; keep a copy of the raw text for reference.
    def normalize_batch(batch):
        batch["sentence_raw"] = batch["sentence"]
        batch["sentence"] = [normalize_urdu(s) for s in batch["sentence"]]
        return batch

    raw = raw.map(normalize_batch, batched=True, num_proc=args.num_proc,
                  desc="Normalizing transcripts")

    # 3. Filter: non-empty transcript, duration within Whisper's 30s window.
    def keep(example):
        n_samples = len(example["audio"]["array"])
        return bool(example["sentence"].strip()) and (n_samples / SAMPLE_RATE) <= args.max_seconds

    raw = raw.filter(keep, num_proc=args.num_proc, desc="Filtering clips")

    # 4. Optional caps (useful on Colab free tier / quick smoke tests).
    if args.max_train_samples:
        raw["train"] = raw["train"].select(range(min(args.max_train_samples, len(raw["train"]))))
    if args.max_test_samples:
        raw["test"] = raw["test"].select(range(min(args.max_test_samples, len(raw["test"]))))

    # 5. Stats.
    for split in ("train", "test"):
        ds = raw[split]
        print(f"{split}: {len(ds)} utterances, {_duration_hours(ds):.2f} hours of audio")

    # 6. Save.
    raw.save_to_disk(args.output_dir)
    print(f"Saved prepared dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
