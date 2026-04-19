# DIVR Thesis

This repository provides a unified entry point for running audio and audio-text classification experiments directly from `src/__main__.py`.

The current workflow is built around task folders under `tasks/`, automatic training and validation, periodic and best-checkpoint saving, automatic testing after training, and CSV-first analysis outputs.

## Features

- Run experiments from a single CLI entry point.
- Read datasets directly from `tasks/<task_name>/train.yml`, `val.yml`, and `test.yml`.
- Support experiment modes with mandatory audio input:
  - audio only
  - audio + text fusion
- Save checkpoints every N epochs and always keep the best model.
- Automatically test the best checkpoint after training.
- Save test predictions to CSV before generating analysis artifacts.
- Generate confusion matrices plus overall accuracy and macro F1 summaries, including age-bucket accuracy when age metadata is available.

## Repository Layout

```text
data_loader/         Task loading, batching, and YAML parsing
experiments/         Runtime, training, testing, and analysis code
model/               Feature extractors, fusion modules, and classifiers
src/__main__.py      Main CLI entry point
tasks/               Task folders with train/val/test YAML files
```

## Requirements

The project expects a Python environment with the main dependencies already installed.

Core packages used by the current pipeline:

- `torch`
- `torchaudio`
- `librosa`
- `s3prl`
- `transformers`
- `PyYAML`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`
- `shap`

If you want TensorBoard logging, install `tensorboard`. If it is not installed, run with `--disable-tensorboard`.

## Task Format

Each task folder must contain:

- `train.yml`
- `val.yml`
- `test.yml`

Each YAML file is a mapping from sample ID to metadata. Example:

```yaml
sample_0001:
  age: 56
  gender: female
  label: organic
  speaker_id: 0001
  text_keys:
    - /absolute/path/to/audio.wav
  texts:
    - age=56; gender=female
```

Field conventions:

- `label`: class label used for supervision
- `text_keys`: audio path list
- `texts`: text payload list
- other fields: optional metadata kept for analysis

The pipeline does not hardcode age or gender as the only metadata. Any additional fields in the task YAML can be preserved and used as metadata.

## Main Entry Point

Run experiments with:

```bash
python src/__main__.py [OPTIONS]
```

List available tasks:

```bash
python src/__main__.py --list-tasks
```

## Main Options

- `--task-dir`: path to a task folder
- `--combine-mode`: one of `audio`, `concatenation`, `cross_attention`, `gated`, `film`
- `--feature-model`: audio pretrained model name (S3PRL upstream name or HuggingFace model ID)
- `--epochs`: number of training epochs
- `--batch-size`: batch size
- `--learning-rate`: optimizer learning rate
- `--save-every`: save a checkpoint every N epochs
- `--device`: `auto`, `cpu`, or `cuda`
- `--text-fields`: select which key-value fields from `texts` should be used as text input
- `--text-equals`: filter text payloads by key-value conditions
- `--disable-tensorboard`: disable TensorBoard logging

Notes:

- In `audio` mode, `--text-fields` and `--text-equals` are ignored.
- This experiment always requires audio input and `--feature-model`.

## Example Commands

Audio only:

```bash
python src/__main__.py \
  --task-dir tasks/femh \
  --combine-mode audio \
  --feature-model wav2vec_large \
  --epochs 50 \
  --save-every 10 \
  --batch-size 16
```

Audio + text with concatenation fusion:

```bash
python src/__main__.py \
  --task-dir tasks/femh \
  --combine-mode concatenation \
  --feature-model wavlm_base \
  --text-fields age gender \
  --epochs 50 \
  --save-every 10 \
  --batch-size 16
```

Filter text payloads:

```bash
python src/__main__.py \
  --task-dir tasks/femh \
  --combine-mode concatenation \
  --feature-model wavlm_base \
  --text-fields age gender \
  --text-equals gender=female
```

Model selection behavior:

- S3PRL upstream names are supported directly.
- HuggingFace model IDs are also supported when `transformers` is installed.

## Training and Checkpoints

For each run, the system creates a timestamped run directory under `.cache/runs/`.

The training pipeline:

- trains for the requested number of epochs
- evaluates on the validation split every epoch
- records validation accuracy and macro F1 in `results/history.csv`
- saves `best.pt` when validation accuracy improves
- saves `last.pt` every epoch
- saves `epoch_XXXX.pt` every `--save-every` epochs

## Testing and Analysis

After training finishes, the best checkpoint is automatically loaded and evaluated on the test split.

Outputs are written under the run directory, typically in:

- `results/predictions.csv`
- `results/test_summary.json`
- `results/training_summary.json`
- `results/history.csv`
- `results/analysis/confusion_matrix.csv`
- `results/analysis/confusion_matrix.png`
- `results/analysis/accuracy_by_label.csv`
- `results/analysis/accuracy_by_label.png`
- `results/analysis/shap_contribution_by_class.csv`
- `results/analysis/shap_abs_audio_demographic_by_class.png`
- `results/analysis/shap_abs_demographic_by_class.png`
- `results/analysis/shap_signed_audio_demographic_by_class.png`
- `results/analysis/shap_signed_demographic_by_class.png`

`results/training_summary.json` includes the best checkpoint epoch and validation metrics. `results/test_summary.json` includes the loaded checkpoint epoch plus test accuracy and macro F1.

SHAP outputs are generated when `shap` is installed and test samples contain metadata. The SHAP CSV reports per-class contribution ratios for audio features, age, gender, smoking, and drinking.

## How Text Input Works

Text input is constructed from the `texts` payloads in each sample.

- If `--text-fields all` is used, the full payload string is kept.
- If specific fields are provided, the loader extracts matching key-value pairs.
- If `--text-equals` is set, only payloads matching the requested filters are kept.

This makes the text branch generic enough for metadata beyond age and gender.

## Output Naming

Run directories follow this pattern:

```text
.cache/runs/<task_name>_<model_key>_<combine_mode>_<timestamp>
```

Examples:

- `femh_wav2vec_large_audio_20260311_152902`
- `femh_wavlm_base_concatenation_20260311_142952`

## Development Notes

- The current codebase has been refactored away from many handwritten experiment scripts into a single configurable runtime.
- Task YAML files are now the source of truth.
- CSV is the first test artifact, and analysis is derived from that CSV.

## Quick Start

If you only want to verify that the pipeline runs:

```bash
python src/__main__.py \
  --task-dir tasks/femh \
  --combine-mode audio \
  --feature-model wavlm_base \
  --text-fields age gender \
  --epochs 1 \
  --batch-size 8 \
  --device cpu \
  --disable-tensorboard
```