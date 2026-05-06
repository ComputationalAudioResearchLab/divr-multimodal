# divr_llm

This project is self-contained and only uses code inside the `divr_llm` folder.

## Qwen2Audio LoRA

The default fine-tuning preset is in `lora/qwen2audio.yaml` and targets the `tasks/voiced_0_test` split.

Dry-run the resolved configuration:

```bash
python -m src --config lora/qwen2audio.yaml
```

Start LoRA fine-tuning:

```bash
python -m src --config lora/qwen2audio.yaml --train
```

## Task format

Each task YAML entry should follow the existing pattern in `tasks/`:

- `text_keys`: local or absolute audio file paths
- `texts`: target answers or supervision text

## Local artifacts

Model caches and training outputs are written under `model/` so the project stays isolated from the other workspaces.