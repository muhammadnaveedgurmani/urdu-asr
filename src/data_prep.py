"""Prepare Mozilla Common Voice (Urdu) for Whisper fine-tuning.

Pipeline:
  1. Load the Urdu subset of Common Voice from the Hugging Face Hub.
  2. Resample all audio to 16 kHz (Whisper's native sample rate).
  3. Normalize transcripts with src/normalize.py.
  4. Drop empty transcripts and clips longer than --max_seconds.
  5. Save a Hugging Face DatasetDict to disk for src/train.py.

NOTE ON ACCESS: Common Voice is a *gated* dataset. Before running this you must
  a) create a Hugging Face account,
  b) open https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
     and click "Agree and access repository" (accept the terms), and
  c) run `huggingface-cli login` (or set HF_TOKEN) on the machine that runs this.

Usage:
    python src/data_prep.py --output_dir ./data/cv_ur --max_train_samples 2000
"""

import argparse

from datasets import Audio, DatasetDict, load_dataset

from src.normalize import normalize_urdu

DATASET_ID = "mozilla-foundation/common_voice_17_0"
LANGUAGE = "ur"
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

    print(f"Loading {DATASET_ID} [{LANGUAGE}] ...")
    # split="train" / "test" exist for Common Voice; validation is folded into train.
    raw = DatasetDict({
        "train": load_dataset(DATASET_ID, LANGUAGE, split="train", trust_remote_code=True),
        "test": load_dataset(DATASET_ID, LANGUAGE, split="test", trust_remote_code=True),
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
