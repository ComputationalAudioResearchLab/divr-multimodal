from __future__ import annotations

import csv
import inspect
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

from data_loader import VoiceSample, VoiceYamlDataset, collect_labels, load_voice_yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")

    config["_config_path"] = str(path)
    config["_project_root"] = str(_find_project_root(path))
    return config


def train(config_path: str | Path) -> Path:
    config = load_config(config_path)
    _set_seed_from_config(config)

    train_path, val_path, test_path = _dataset_paths(config)
    train_dataset = VoiceYamlDataset(train_path)
    val_dataset = VoiceYamlDataset(val_path) if val_path else None
    test_samples = load_voice_yaml(test_path) if test_path else []
    labels = _labels_from_config(config, train_dataset.samples, val_dataset.samples if val_dataset else [], test_samples)

    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "label_map.json", {"labels": labels})

    processor, model = _load_processor_and_model(config, train_mode=True)
    supervised_collator = Qwen2AudioSupervisedCollator(processor=processor, labels=labels, config=config)

    training_args = _training_args(config, output_dir, has_eval=val_dataset is not None)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": supervised_collator,
    }
    _attach_processor_to_trainer_kwargs(trainer_kwargs, processor)

    from transformers import Trainer

    trainer = Trainer(**trainer_kwargs)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    resume_from_checkpoint = _get(config, "training.resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    _write_json(adapter_dir / "label_map.json", {"labels": labels})
    return adapter_dir


def evaluate(config_path: str | Path, checkpoint: str | Path | None = None) -> Path:
    config = load_config(config_path)
    _set_seed_from_config(config)

    train_path, val_path, test_path = _dataset_paths(config)
    if not test_path:
        raise ValueError("No test dataset was configured.")

    train_samples = load_voice_yaml(train_path)
    val_samples = load_voice_yaml(val_path) if val_path else []
    test_dataset = VoiceYamlDataset(test_path)
    labels = _labels_from_config(config, train_samples, val_samples, test_dataset.samples)

    adapter_path = Path(checkpoint).resolve() if checkpoint else _adapter_path(config)
    processor, model = _load_processor_and_model(config, train_mode=False, adapter_path=adapter_path)
    result_dir = _analysis_run_dir(config)
    result_dir.mkdir(parents=True, exist_ok=True)

    predictions = generate_predictions(
        model=model,
        processor=processor,
        samples=test_dataset.samples,
        labels=labels,
        config=config,
    )
    write_analysis_outputs(predictions=predictions, labels=labels, output_dir=result_dir)
    return result_dir


def run(config_path: str | Path) -> Path:
    adapter_dir = train(config_path)
    return evaluate(config_path, checkpoint=adapter_dir)


def check_data(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    train_path, val_path, test_path = _dataset_paths(config)
    split_paths = {"train": train_path, "val": val_path, "test": test_path}
    split_samples = {
        split: load_voice_yaml(path) if path else []
        for split, path in split_paths.items()
    }
    labels = _labels_from_config(
        config,
        split_samples["train"],
        split_samples["val"],
        split_samples["test"],
    )

    missing_audio: list[dict[str, str]] = []
    for split, samples in split_samples.items():
        for sample in samples:
            for audio_path in sample.audio_paths:
                if not Path(audio_path).exists():
                    missing_audio.append(
                        {"split": split, "sample_id": sample.sample_id, "audio_path": audio_path}
                    )

    return {
        "labels": labels,
        "splits": {
            split: {
                "path": str(split_paths[split]) if split_paths[split] else None,
                "num_samples": len(samples),
            }
            for split, samples in split_samples.items()
        },
        "missing_audio_count": len(missing_audio),
        "missing_audio_examples": missing_audio[:20],
    }


class Qwen2AudioSupervisedCollator:
    def __init__(self, processor: Any, labels: Sequence[str], config: dict[str, Any]):
        self.processor = processor
        self.labels = list(labels)
        self.config = config
        self.max_length = _get(config, "data.max_length", 2048)
        self.max_audio_seconds = _get(config, "data.max_audio_seconds")
        self.max_audios_per_sample = int(_get(config, "data.max_audios_per_sample", 1))
        self.sampling_rate = _processor_sampling_rate(processor)

    def __call__(self, batch: Sequence[VoiceSample]) -> dict[str, Any]:
        prompt_texts = [
            _chat_text(self.processor, sample, self.labels, self.config, include_answer=False)
            for sample in batch
        ]
        full_texts = [
            _chat_text(self.processor, sample, self.labels, self.config, include_answer=True)
            for sample in batch
        ]
        audios = _load_batch_audios(
            batch,
            sampling_rate=self.sampling_rate,
            max_audio_seconds=self.max_audio_seconds,
            max_audios_per_sample=self.max_audios_per_sample,
        )
        inputs = _call_processor(
            self.processor,
            texts=full_texts,
            audios=audios,
            sampling_rate=self.sampling_rate,
            max_length=self.max_length,
        )

        labels = inputs["input_ids"].clone()
        prompt_token_ids = [
            self.processor.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            ).input_ids
            for text in prompt_texts
        ]
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        for row, prompt_ids in enumerate(prompt_token_ids):
            _mask_prompt_tokens(
                labels=labels,
                input_ids=inputs["input_ids"],
                row=row,
                prompt_ids=prompt_ids,
                pad_token_id=pad_token_id,
            )

        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        inputs["labels"] = labels
        return dict(inputs)


class Qwen2AudioInferenceCollator:
    def __init__(self, processor: Any, labels: Sequence[str], config: dict[str, Any]):
        self.processor = processor
        self.labels = list(labels)
        self.config = config
        self.max_length = _get(config, "data.max_length", 2048)
        self.max_audio_seconds = _get(config, "data.max_audio_seconds")
        self.max_audios_per_sample = int(_get(config, "data.max_audios_per_sample", 1))
        self.sampling_rate = _processor_sampling_rate(processor)

    def __call__(self, batch: Sequence[VoiceSample]) -> dict[str, Any]:
        prompt_texts = [
            _chat_text(self.processor, sample, self.labels, self.config, include_answer=False)
            for sample in batch
        ]
        audios = _load_batch_audios(
            batch,
            sampling_rate=self.sampling_rate,
            max_audio_seconds=self.max_audio_seconds,
            max_audios_per_sample=self.max_audios_per_sample,
        )
        inputs = _call_processor(
            self.processor,
            texts=prompt_texts,
            audios=audios,
            sampling_rate=self.sampling_rate,
            max_length=self.max_length,
        )
        return {"inputs": dict(inputs), "samples": list(batch)}


def generate_predictions(
    model: Any,
    processor: Any,
    samples: Sequence[VoiceSample],
    labels: Sequence[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    batch_size = int(_get(config, "evaluation.batch_size", _get(config, "training.per_device_eval_batch_size", 1)))
    collator = Qwen2AudioInferenceCollator(processor=processor, labels=labels, config=config)
    loader = DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model.eval()
    results: list[dict[str, Any]] = []
    device = _model_input_device(model)
    generation_kwargs = _generation_kwargs(config, processor)

    for batch in loader:
        inputs = _move_tensors(batch["inputs"], device)
        with torch.no_grad():
            generated = model.generate(**inputs, **generation_kwargs)

        prompt_width = inputs["input_ids"].shape[1]
        generated = generated[:, prompt_width:]
        decoded = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for sample, raw_output in zip(batch["samples"], decoded):
            prediction = normalize_prediction(raw_output, labels)
            results.append(
                {
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "prediction": prediction,
                    "raw_output": raw_output.strip(),
                    "audio_paths": "|".join(sample.audio_paths),
                    "texts": sample.text,
                }
            )
    return results


def normalize_prediction(raw_output: str, labels: Sequence[str]) -> str:
    text = raw_output.strip()
    if not text:
        return "__unknown__"

    normalized_text = _normalize_label_text(text)
    label_lookup = {_normalize_label_text(label): label for label in labels}
    first_line = _normalize_label_text(text.splitlines()[0])
    if first_line in label_lookup:
        return label_lookup[first_line]

    for normalized_label, label in label_lookup.items():
        pattern = rf"(^|[^a-z0-9]){re.escape(normalized_label)}($|[^a-z0-9])"
        if re.search(pattern, normalized_text):
            return label
    return "__unknown__"


def write_analysis_outputs(
    predictions: Sequence[dict[str, Any]],
    labels: Sequence[str],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, matrix_labels, matrix = compute_metrics(predictions, labels)
    _write_json(output_dir / "metrics.json", metrics)
    _write_text(output_dir / "accuracy.txt", f"accuracy: {metrics['accuracy']:.6f}\n")
    _write_confusion_csv(output_dir / "confusion_matrix.csv", matrix_labels, matrix)
    _write_predictions_csv(output_dir / "predictions.csv", predictions)
    _try_write_confusion_png(output_dir / "confusion_matrix.png", matrix_labels, matrix)


def compute_metrics(
    predictions: Sequence[dict[str, Any]],
    labels: Sequence[str],
) -> tuple[dict[str, Any], list[str], list[list[int]]]:
    matrix_labels = list(labels)
    if any(item["prediction"] not in labels for item in predictions):
        matrix_labels.append("__unknown__")

    index = {label: pos for pos, label in enumerate(matrix_labels)}
    matrix = [[0 for _ in matrix_labels] for _ in matrix_labels]
    correct = 0
    for item in predictions:
        true_label = item["label"]
        predicted_label = item["prediction"] if item["prediction"] in index else "__unknown__"
        if true_label not in index:
            continue
        matrix[index[true_label]][index[predicted_label]] += 1
        correct += int(true_label == predicted_label)

    total = len(predictions)
    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in labels:
        row = index[label]
        true_positive = matrix[row][row]
        false_positive = sum(matrix[r][row] for r in range(len(matrix_labels)) if r != row)
        false_negative = sum(matrix[row][c] for c in range(len(matrix_labels)) if c != row)
        precision = _safe_div(true_positive, true_positive + false_positive)
        recall = _safe_div(true_positive, true_positive + false_negative)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[row]),
        }

    metrics = {
        "accuracy": _safe_div(correct, total),
        "total": total,
        "correct": correct,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "labels": list(labels),
        "per_label": per_label,
    }
    return metrics, matrix_labels, matrix


def _load_processor_and_model(
    config: dict[str, Any],
    train_mode: bool,
    adapter_path: Path | None = None,
) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

    model_name = _get(config, "model.name", "Qwen/Qwen2-Audio-7B-Instruct")
    trust_remote_code = bool(_get(config, "model.trust_remote_code", True))
    cache_dir = _resolve_optional_path(config, _get(config, "model.cache_dir"))

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    quantization_config = None
    load_in_4bit = bool(_get(config, "quantization.load_in_4bit", True))
    load_in_8bit = bool(_get(config, "quantization.load_in_8bit", False))
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose only one quantization mode: load_in_4bit or load_in_8bit.")
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=_get(config, "quantization.bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=_torch_dtype(_get(config, "quantization.bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=bool(_get(config, "quantization.bnb_4bit_use_double_quant", True)),
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=bool(
                _get(config, "quantization.llm_int8_enable_fp32_cpu_offload", False)
            ),
        )

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": _get(config, "model.device_map", "auto"),
        "cache_dir": str(cache_dir) if cache_dir else None,
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": bool(_get(config, "model.low_cpu_mem_usage", True)),
    }
    max_memory = _get(config, "model.max_memory")
    if max_memory:
        model_kwargs["max_memory"] = dict(max_memory)
    torch_dtype = _get(config, "model.torch_dtype", "bfloat16")
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = _torch_dtype(torch_dtype)
    attn_implementation = _get(config, "model.attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name, **model_kwargs)

    if train_mode:
        model.config.use_cache = False
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool(_get(config, "training.gradient_checkpointing", True)),
            )

        lora_config = LoraConfig(
            r=int(_get(config, "lora.r", 16)),
            lora_alpha=int(_get(config, "lora.alpha", 32)),
            lora_dropout=float(_get(config, "lora.dropout", 0.05)),
            bias=_get(config, "lora.bias", "none"),
            task_type=_get(config, "lora.task_type", "CAUSAL_LM"),
            target_modules=list(_get(config, "lora.target_modules", _default_lora_targets())),
        )
        model = get_peft_model(model, lora_config)
    elif adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path))

    return processor, model


def _training_args(config: dict[str, Any], output_dir: Path, has_eval: bool) -> Any:
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(_get(config, "training.num_train_epochs", 3)),
        "per_device_train_batch_size": int(_get(config, "training.per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(_get(config, "training.per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(_get(config, "training.gradient_accumulation_steps", 8)),
        "learning_rate": float(_get(config, "training.learning_rate", 2e-4)),
        "weight_decay": float(_get(config, "training.weight_decay", 0.0)),
        "warmup_ratio": float(_get(config, "training.warmup_ratio", 0.03)),
        "logging_steps": int(_get(config, "training.logging_steps", 10)),
        "save_steps": int(_get(config, "training.save_steps", 100)),
        "eval_steps": int(_get(config, "training.eval_steps", 100)),
        "save_total_limit": int(_get(config, "training.save_total_limit", 2)),
        "optim": _get(config, "training.optim", "paged_adamw_8bit"),
        "gradient_checkpointing": bool(_get(config, "training.gradient_checkpointing", True)),
        "remove_unused_columns": False,
        "report_to": list(_get(config, "training.report_to", [])),
        "bf16": bool(_get(config, "training.bf16", True)),
        "fp16": bool(_get(config, "training.fp16", False)),
    }
    if "max_grad_norm" in _get(config, "training", {}):
        kwargs["max_grad_norm"] = float(_get(config, "training.max_grad_norm"))

    eval_strategy = _get(config, "training.eval_strategy", "steps" if has_eval else "no")
    save_strategy = _get(config, "training.save_strategy", "steps")
    signature = inspect.signature(TrainingArguments.__init__)
    if "torch_empty_cache_steps" in _get(config, "training", {}) and "torch_empty_cache_steps" in signature.parameters:
        kwargs["torch_empty_cache_steps"] = int(_get(config, "training.torch_empty_cache_steps"))
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy
    kwargs["save_strategy"] = save_strategy
    return TrainingArguments(**kwargs)


def _attach_processor_to_trainer_kwargs(trainer_kwargs: dict[str, Any], processor: Any) -> None:
    from transformers import Trainer

    signature = inspect.signature(Trainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor


def _chat_text(
    processor: Any,
    sample: VoiceSample,
    labels: Sequence[str],
    config: dict[str, Any],
    include_answer: bool,
) -> str:
    label_text = ", ".join(labels)
    system_prompt = _get(
        config,
        "prompt.system",
        "You are an expert assistant for voice disorder classification.",
    )
    user_template = _get(
        config,
        "prompt.user_template",
        (
            "Classify the voice disorder from the audio and metadata. "
            "Return only one label from: {labels}.\nMetadata: {text}"
        ),
    )
    user_text = user_template.format(labels=label_text, text=sample.text, sample_id=sample.sample_id)
    content = [
        {"type": "audio", "audio_url": path}
        for path in sample.audio_paths[: int(_get(config, "data.max_audios_per_sample", 1))]
    ]
    content.append({"type": "text", "text": user_text})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": sample.label})
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _load_batch_audios(
    batch: Sequence[VoiceSample],
    sampling_rate: int,
    max_audio_seconds: float | None,
    max_audios_per_sample: int,
) -> list[Any]:
    import librosa
    import numpy as np

    audios: list[Any] = []
    for sample in batch:
        for path in sample.audio_paths[:max_audios_per_sample]:
            audio, _ = librosa.load(
                path,
                sr=sampling_rate,
                mono=True,
                duration=max_audio_seconds,
            )
            audios.append(np.asarray(audio, dtype=np.float32))
    return audios


def _call_processor(
    processor: Any,
    texts: Sequence[str],
    audios: Sequence[Any],
    sampling_rate: int,
    max_length: int | None,
) -> Any:
    base_kwargs = {
        "text": list(texts),
        "return_tensors": "pt",
        "padding": True,
    }
    if max_length:
        base_kwargs["truncation"] = True
        base_kwargs["max_length"] = int(max_length)

    attempts = [
        {"audios": list(audios), "sampling_rate": sampling_rate},
        {"audios": list(audios)},
        {"audio": list(audios), "sampling_rate": sampling_rate},
        {"audio": list(audios)},
    ]
    last_error: TypeError | None = None
    for audio_kwargs in attempts:
        try:
            return processor(**base_kwargs, **audio_kwargs)
        except TypeError as exc:
            last_error = exc
    raise last_error or TypeError("Unable to call Qwen2Audio processor.")


def _mask_prompt_tokens(
    labels: Any,
    input_ids: Any,
    row: int,
    prompt_ids: Sequence[int],
    pad_token_id: int | None,
) -> None:
    token_ids = input_ids[row].tolist()
    start = _find_subsequence(token_ids, list(prompt_ids))
    if start is None:
        start = _first_non_pad_index(token_ids, pad_token_id)
    end = min(start + len(prompt_ids), len(token_ids))
    labels[row, start:end] = -100


def _find_subsequence(values: Sequence[int], subsequence: Sequence[int]) -> int | None:
    if not subsequence or len(subsequence) > len(values):
        return None
    width = len(subsequence)
    for start in range(0, len(values) - width + 1):
        if list(values[start : start + width]) == list(subsequence):
            return start
    return None


def _first_non_pad_index(values: Sequence[int], pad_token_id: int | None) -> int:
    if pad_token_id is None:
        return 0
    for index, value in enumerate(values):
        if value != pad_token_id:
            return index
    return 0


def _generation_kwargs(config: dict[str, Any], processor: Any) -> dict[str, Any]:
    kwargs = {
        "max_new_tokens": int(_get(config, "generation.max_new_tokens", 8)),
        "do_sample": bool(_get(config, "generation.do_sample", False)),
    }
    temperature = _get(config, "generation.temperature")
    top_p = _get(config, "generation.top_p")
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if top_p is not None:
        kwargs["top_p"] = float(top_p)

    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None) or eos_token_id
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def _dataset_paths(config: dict[str, Any]) -> tuple[Path, Path | None, Path | None]:
    dataset = _get(config, "dataset", {})
    task_dir = dataset.get("task_dir")
    if task_dir:
        base = _resolve_path(config, task_dir)
        train_path = _resolve_path(config, dataset.get("train_file", "train.yml"), base=base)
        val_path = _resolve_path(config, dataset.get("val_file", "val.yml"), base=base)
        test_path = _resolve_path(config, dataset.get("test_file", "test.yml"), base=base)
    else:
        train_path = _resolve_path(config, dataset["train_path"])
        val_path = _resolve_optional_path(config, dataset.get("val_path"))
        test_path = _resolve_optional_path(config, dataset.get("test_path"))

    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_path}")
    if val_path and not val_path.exists():
        val_path = None
    if test_path and not test_path.exists():
        test_path = None
    return train_path, val_path, test_path


def _labels_from_config(
    config: dict[str, Any],
    train_samples: Sequence[VoiceSample],
    val_samples: Sequence[VoiceSample],
    test_samples: Sequence[VoiceSample],
) -> list[str]:
    configured = _get(config, "dataset.labels")
    if configured:
        return [str(label) for label in configured]
    return collect_labels([train_samples, val_samples, test_samples])


def _analysis_run_dir(config: dict[str, Any]) -> Path:
    task_name = _task_name(config)
    run_name = _get(config, "run_name", "qwen2audio_lora")
    return _resolve_path(config, _get(config, "analysis_dir", "analysis")) / task_name / _slug(run_name)


def _output_dir(config: dict[str, Any]) -> Path:
    task_name = _task_name(config)
    run_name = _get(config, "run_name", "qwen2audio_lora")
    default = Path("outputs") / _slug(run_name) / task_name
    return _resolve_path(config, _get(config, "output_dir", str(default)))


def _adapter_path(config: dict[str, Any]) -> Path:
    configured = _get(config, "evaluation.adapter_path")
    if configured:
        return _resolve_path(config, configured)
    return _output_dir(config) / "final_adapter"


def _task_name(config: dict[str, Any]) -> str:
    dataset = _get(config, "dataset", {})
    if dataset.get("name"):
        return _slug(dataset["name"])
    if dataset.get("task_dir"):
        return _resolve_path(config, dataset["task_dir"]).name
    train_path = dataset.get("train_path")
    return _resolve_path(config, train_path).parent.name if train_path else "task"


def _find_project_root(path: Path) -> Path:
    for parent in [path.parent, *path.parents]:
        if (parent / "tasks").exists() and (parent / "src").exists():
            return parent
    return path.parent


def _resolve_path(config: dict[str, Any], value: str | Path, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base or Path(config["_project_root"])) / path


def _resolve_optional_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    return _resolve_path(config, value) if value else None


def _processor_sampling_rate(processor: Any) -> int:
    feature_extractor = getattr(processor, "feature_extractor", None)
    sampling_rate = getattr(feature_extractor, "sampling_rate", None)
    if sampling_rate is None:
        raise ValueError("Processor feature_extractor has no sampling_rate.")
    return int(sampling_rate)


def _model_input_device(model: Any) -> Any:
    for parameter in model.parameters():
        return parameter.device
    return "cpu"


def _move_tensors(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _torch_dtype(value: str) -> Any:
    import torch

    normalized = str(value).lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def _set_seed_from_config(config: dict[str, Any]) -> None:
    from transformers import set_seed

    seed = _get(config, "seed")
    if seed is not None:
        set_seed(int(seed))


def _default_lora_targets() -> list[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _normalize_label_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text.strip("_") or "run"


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def _write_confusion_csv(path: Path, labels: Sequence[str], matrix: Sequence[Sequence[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label/predicted_label", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def _write_predictions_csv(path: Path, predictions: Sequence[dict[str, Any]]) -> None:
    fieldnames = ["sample_id", "label", "prediction", "raw_output", "audio_paths", "texts"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _try_write_confusion_png(path: Path, labels: Sequence[str], matrix: Sequence[Sequence[int]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    size = max(4.0, min(12.0, 1.2 * len(labels) + 2.5))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    max_value = max([max(row) for row in matrix] or [0])
    threshold = max_value / 2 if max_value else math.inf
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            color = "white" if value > threshold else "black"
            ax.text(column_index, row_index, str(value), ha="center", va="center", color=color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
