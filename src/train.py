"""Fine-tune Whisper-small on prepared Urdu speech data.

Follows the official Hugging Face "Fine-Tune Whisper" recipe
(Sanchit Gandhi): log-Mel input features, label masking with -100,
and generation-based WER evaluation via Seq2SeqTrainer.

Run on a Colab T4 GPU (see COLAB_TRAINING.md):
    python src/train.py --config configs/training_config.yaml \\
        --data_dir ./data/cv_ur --output_dir ./models/whisper-small-ur

Config values can be overridden from the CLI, e.g. --learning_rate 5e-6.
"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import yaml
from datasets import load_from_disk
from jiwer import wer as jiwer_wer
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.normalize import normalize_urdu


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads log-Mel features and labels; masks label padding with -100.

    Standard collator from the HF Whisper fine-tuning guide: the -100 ids are
    ignored by the cross-entropy loss, and the decoder start token is stripped
    from the labels so the model learns to generate it.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Pad the audio features (variable length -> longest in batch).
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad the tokenized transcripts.
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Mask padding tokens so the loss ignores them.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Cut the BOS token if every label sequence starts with it: during
        # training the decoder input is shifted internally, so keeping BOS in
        # the labels would teach the model to emit it twice.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-small on Urdu ASR data")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Prepared DatasetDict from src/data_prep.py")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Overrides configs output_dir")
    # Common hyperparameter overrides (all optional; None = use YAML).
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="If set, overrides num_train_epochs")
    return parser.parse_args()


def load_config(path: str, args: argparse.Namespace) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # CLI overrides win over YAML.
    for key in ("output_dir", "learning_rate", "num_train_epochs",
                "per_device_train_batch_size", "max_steps"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args)

    print(f"Loading prepared data from {args.data_dir} ...")
    dataset = load_from_disk(args.data_dir)

    print(f"Loading processor + model: {cfg['model_name']}")
    processor = WhisperProcessor.from_pretrained(
        cfg["model_name"], language=cfg["language"], task=cfg["task"]
    )
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model_name"])
    # Force the decoder prompt (<|ur|><|transcribe|><|notimestamps|>) both in
    # training labels (via the processor) and in eval-time generation.
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=cfg["language"], task=cfg["task"]
    )
    model.config.suppress_tokens = []
    # T4-friendly: recompute activations instead of storing them.
    model.config.use_cache = False

    # --- Feature extraction: waveform -> log-Mel, transcript -> label ids ---
    def prepare_dataset(batch):
        sampling_rate = batch["audio"]["sampling_rate"][0]  # uniform 16 kHz
        batch["input_features"] = processor.feature_extractor(
            batch["audio"]["array"], sampling_rate=sampling_rate
        ).input_features
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    vectorized = dataset.map(
        prepare_dataset,
        batched=True,
        batch_size=32,
        remove_columns=["audio", "sentence", "sentence_raw"],
        desc="Extracting log-Mel features",
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # --- Metric: WER on normalized text (so orthographic noise is not punished) ---
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [normalize_urdu(s) for s in pred_str]
        label_str = [normalize_urdu(s) for s in label_str]

        return {"wer": 100 * jiwer_wer(label_str, pred_str)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg.get("max_steps", -1),
        gradient_checkpointing=cfg["gradient_checkpointing"],
        fp16=cfg["fp16"],
        max_grad_norm=cfg["max_grad_norm"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        logging_steps=cfg["logging_steps"],
        logging_dir=cfg["logging_dir"],
        predict_with_generate=cfg["predict_with_generate"],
        generation_max_length=cfg["generation_max_length"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        report_to=cfg["report_to"],
        seed=cfg["seed"],
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=vectorized["train"],
        eval_dataset=vectorized["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,  # lets Trainer save the processor
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    processor.save_pretrained(cfg["output_dir"])
    print(f"Best checkpoint saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
