# Training on Google Colab (free T4 GPU)

This machine has no GPU, so training runs on Colab. Everything else
(data prep logic, eval, serving) runs anywhere.

## 1. Open Colab with a GPU
1. Go to https://colab.research.google.com and open a new notebook.
2. Runtime → Change runtime type → Hardware accelerator → **T4 GPU** → Save.

## 2. Install dependencies
```python
!pip install -q -r requirements.txt
```
(Upload `requirements.txt`, or clone the repo first and run from its folder.)

## 3. Log in to Hugging Face and accept the dataset terms
1. In a browser tab, open https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 and click **"Agree and access repository"**.
2. Back in Colab:
```python
!huggingface-cli login
```
Paste a **read** token from https://huggingface.co/settings/tokens.

## 4. Clone this repo
```python
!git clone https://github.com/<your-username>/urdu-asr.git
%cd urdu-asr
!pip install -q -r requirements.txt
```

## 5. Prepare the data
```python
!python src/data_prep.py --output_dir ./data/cv_ur --max_train_samples 4000
```
Start with 4000 samples (~2–3h of audio) for a first run; remove the cap for the full run. Note the printed hours/utterances for the README.

## 6. Train
```python
!python src/train.py --config configs/training_config.yaml --data_dir ./data/cv_ur
```
Watch WER in the TensorBoard logs. A smoke test first is wise:
```python
!python src/train.py --config configs/training_config.yaml --data_dir ./data/cv_ur --max_steps 50
```

## 7. Evaluate
```python
!python src/evaluate.py --data_dir ./data/cv_ur \
    --finetuned_model ./models/whisper-small-ur \
    --output results/wer_results.json
```
Copy the printed table into the README results section.

## 8. Download the checkpoint
1. Zip it: `!zip -r whisper-small-ur.zip models/whisper-small-ur`
2. Download via the Colab file browser (left sidebar → files → right-click → Download).

## 9. Serve locally
```bash
MODEL_PATH=./models/whisper-small-ur uvicorn api.main:app --port 8000
python demo/app.py   # Gradio UI on :7860
```

## Troubleshooting
- **401 on dataset load** → you skipped step 3 (terms not accepted or not logged in).
- **CUDA OOM** → halve `per_device_train_batch_size` in the YAML or raise `gradient_accumulation_steps`.
- **Slow eval** → lower `--max_samples` in `src/evaluate.py`; whisper-large runs on a 25-sample subset by default.
