from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from analysis.report import DEFAULT_LABEL_PROMPT_TEMPLATE, extract_label_from_text, write_test_report
from data_loader.qwen2audio import (
    PROJECT_ROOT,
    Qwen2AudioDataset,
    Qwen2AudioSample,
    SafeFormatDict,
    load_audio_array,
    load_samples_from_files,
    resolve_path,
)


def build_user_content(audio_paths: Sequence[Path], prompt: str) -> list[dict[str, str]]:
    content = [{"type": "audio", "audio_url": str(audio_path)} for audio_path in audio_paths]
    content.append({"type": "text", "text": prompt})
    return content


def render_conversation_text(
    processor: Any,
    audio_paths: Sequence[Path],
    prompt: str,
    target_text: str | None = None,
    system_prompt: str | None = None,
) -> str:
    conversation: list[dict[str, Any]] = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": build_user_content(audio_paths, prompt)})
    if target_text is not None:
        conversation.append({"role": "assistant", "content": target_text})
    return processor.apply_chat_template(conversation, add_generation_prompt=target_text is None, tokenize=False)


class Qwen2AudioCollator:
    def __init__(self, processor: Any, sampling_rate: int, system_prompt: str | None = None) -> None:
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.system_prompt = system_prompt

    def __call__(self, batch: Sequence[Qwen2AudioSample]) -> dict[str, Any]:
        prompt_lengths: list[int] = []
        full_texts: list[str] = []
        flat_audios: list[Any] = []

        for sample in batch:
            prompt_text = render_conversation_text(
                self.processor,
                sample.audio_paths,
                sample.prompt,
                system_prompt=self.system_prompt,
            )
            full_text = render_conversation_text(
                self.processor,
                sample.audio_paths,
                sample.prompt,
                target_text=sample.target_text,
                system_prompt=self.system_prompt,
            )

            full_texts.append(full_text)
            prompt_lengths.append(len(self.processor.tokenizer(prompt_text, add_special_tokens=False).input_ids))
            flat_audios.extend(load_audio_array(audio_path, self.sampling_rate) for audio_path in sample.audio_paths)

        batch_inputs = self.processor(
            text=full_texts,
            audios=flat_audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        labels = batch_inputs["input_ids"].clone()
        labels[batch_inputs["attention_mask"] == 0] = -100
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, :prompt_length] = -100

        batch_inputs["labels"] = labels
        return batch_inputs


def load_training_components() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise SystemExit(
            "This project needs transformers, peft, torch, and torchaudio to run Qwen2Audio LoRA training."
        ) from exc

    return torch, LoraConfig, TaskType, get_peft_model, AutoProcessor, Qwen2AudioForConditionalGeneration, Trainer, TrainingArguments, set_seed


def resolve_torch_dtype(torch_module: Any, dtype_name: Any) -> Any:
    if dtype_name in (None, "", "auto"):
        return None
    dtype_lookup = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "half": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if dtype_name not in dtype_lookup:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype_lookup[dtype_name]


def normalize_string_list(value: Any, default: Sequence[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value]
    raise ValueError("Expected a string or list of strings")


def build_task_slug(task_files: Sequence[str | Path]) -> str:
    task_names: list[str] = []
    for task_file in task_files:
        task_name = resolve_path(task_file).parent.name
        if task_name not in task_names:
            task_names.append(task_name)
    return "__".join(task_names) if task_names else "default"


def resolve_analysis_output_dir(config: dict[str, Any]) -> Path:
    analysis_config = config.get("analysis", {})
    return resolve_path(analysis_config.get("output_dir", "analysis/qwen2audio_lora"))


def load_finetuned_model_and_processor(config: dict[str, Any]) -> tuple[Any, Any]:
    torch, *_ = load_training_components()
    from peft import PeftModel

    model_config = config.get("model", {})
    train_config = config.get("train", {})

    model_name_or_path = model_config.get("model_name_or_path", "Qwen/Qwen2-Audio-7B-Instruct")
    cache_dir = model_config.get("cache_dir")
    trust_remote_code = bool(model_config.get("trust_remote_code", True))
    torch_dtype = resolve_torch_dtype(torch, model_config.get("torch_dtype", "bfloat16"))
    output_dir = resolve_path(train_config.get("output_dir", "model/qwen2audio_lora"))

    if not output_dir.is_dir():
        raise FileNotFoundError(f"Trained model directory does not exist: {output_dir}")

    _, _, _, _, AutoProcessor, Qwen2AudioForConditionalGeneration, _, _, _ = load_training_components()
    processor = AutoProcessor.from_pretrained(
        str(output_dir),
        trust_remote_code=trust_remote_code,
        cache_dir=str(resolve_path(cache_dir)) if cache_dir else None,
    )
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        cache_dir=str(resolve_path(cache_dir)) if cache_dir else None,
        torch_dtype=torch_dtype,
    )
    model = PeftModel.from_pretrained(base_model, str(output_dir))
    model.eval()
    return model, processor


def load_test_samples_for_report(config: dict[str, Any]) -> tuple[list[Qwen2AudioSample], list[str]]:
    data_config = config.get("data", {})
    test_task_files = normalize_string_list(data_config.get("test_task_files"))
    if not test_task_files:
        raise ValueError("data.test_task_files must contain at least one task file for test evaluation")

    test_samples = load_samples_from_files(test_task_files, "", "label", 0)
    label_names = sorted(
        {
            sample.label
            for sample in test_samples
            if sample.label is not None
        }
    )
    if not label_names:
        raise ValueError("Test samples must include a label field to build a confusion matrix")
    return test_samples, label_names


def collect_test_predictions(
    model: Any,
    processor: Any,
    samples: Sequence[Qwen2AudioSample],
    label_names: Sequence[str],
    analysis_config: dict[str, Any],
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    torch, *_ = load_training_components()

    prompt_template = str(analysis_config.get("test_prompt_template", DEFAULT_LABEL_PROMPT_TEMPLATE))
    max_new_tokens = int(analysis_config.get("max_new_tokens", 16))
    do_sample = bool(analysis_config.get("do_sample", False))
    num_beams = int(analysis_config.get("num_beams", 1))
    temperature = float(analysis_config.get("temperature", 0.0))
    sampling_rate = int(processor.feature_extractor.sampling_rate)
    label_choices = ", ".join(label_names)
    device = next(model.parameters()).device

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_beams": num_beams,
        "pad_token_id": getattr(processor.tokenizer, "pad_token_id", None)
        or getattr(processor.tokenizer, "eos_token_id", None),
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    records: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            if sample.label is None:
                raise ValueError(f"{sample.source_file}::{sample.example_id} is missing label metadata")

            prompt = prompt_template.format_map(SafeFormatDict({"label_choices": label_choices}))
            conversation_text = render_conversation_text(
                processor,
                sample.audio_paths,
                prompt,
                system_prompt=system_prompt,
            )
            audio_inputs = [load_audio_array(audio_path, sampling_rate) for audio_path in sample.audio_paths]
            model_inputs = processor(
                text=[conversation_text],
                audios=audio_inputs,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            if hasattr(model_inputs, "to"):
                model_inputs = model_inputs.to(device)

            generated_ids = model.generate(**model_inputs, **generation_kwargs)
            prompt_length = int(model_inputs["input_ids"].shape[1])
            generated_text = processor.tokenizer.decode(
                generated_ids[0, prompt_length:],
                skip_special_tokens=True,
            ).strip()
            predicted_label = extract_label_from_text(generated_text, label_names)

            records.append(
                {
                    "sample_id": sample.example_id,
                    "source_file": str(sample.source_file),
                    "audio_paths": "|".join(str(path) for path in sample.audio_paths),
                    "actual_label": sample.label,
                    "predicted_label": predicted_label,
                    "raw_prediction": generated_text,
                    "target_text": sample.target_text,
                    "correct": int(predicted_label == sample.label),
                }
            )

    return records


def build_trainer(config: dict[str, Any]) -> tuple[Any, Any, Path, dict[str, Any]]:
    (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        AutoProcessor,
        Qwen2AudioForConditionalGeneration,
        Trainer,
        TrainingArguments,
        set_seed,
    ) = load_training_components()

    seed = int(config.get("seed", 42))
    set_seed(seed)

    model_config = config.get("model", {})
    lora_config = config.get("lora", {})
    data_config = config.get("data", {})
    train_config = config.get("train", {})

    model_name_or_path = model_config.get("model_name_or_path", "Qwen/Qwen2-Audio-7B-Instruct")
    cache_dir = model_config.get("cache_dir")
    trust_remote_code = bool(model_config.get("trust_remote_code", True))
    torch_dtype = resolve_torch_dtype(torch, model_config.get("torch_dtype", "bfloat16"))

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        cache_dir=str(resolve_path(cache_dir)) if cache_dir else None,
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        cache_dir=str(resolve_path(cache_dir)) if cache_dir else None,
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False

    if train_config.get("gradient_checkpointing", True) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_target_modules = normalize_string_list(
        lora_config.get("target_modules"),
        default=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_config.get("r", 16)),
            lora_alpha=int(lora_config.get("alpha", lora_config.get("lora_alpha", 32))),
            lora_dropout=float(lora_config.get("dropout", lora_config.get("lora_dropout", 0.05))),
            bias=str(lora_config.get("bias", "none")),
            target_modules=lora_target_modules,
        ),
    )

    train_task_files = normalize_string_list(data_config.get("train_task_files"))
    eval_task_files = normalize_string_list(data_config.get("eval_task_files"))
    if not train_task_files:
        raise ValueError("data.train_task_files must contain at least one task file")

    prompt_template = str(data_config.get("prompt_template", "请根据语音内容回答问题。"))
    response_key = str(data_config.get("response_key", "texts"))
    response_index = int(data_config.get("response_index", 0))
    system_prompt = data_config.get("system_prompt")
    if system_prompt is not None:
        system_prompt = str(system_prompt)

    train_samples = load_samples_from_files(train_task_files, prompt_template, response_key, response_index)
    eval_samples = load_samples_from_files(eval_task_files, prompt_template, response_key, response_index) if eval_task_files else []

    train_dataset = Qwen2AudioDataset(train_samples)
    eval_dataset = Qwen2AudioDataset(eval_samples) if eval_samples else None

    output_dir = resolve_path(train_config.get("output_dir", "model/qwen2audio_lora"))
    output_dir.mkdir(parents=True, exist_ok=True)

    precision = str(train_config.get("precision", "bf16")).lower()
    bf16 = precision in {"bf16", "bfloat16"}
    fp16 = precision in {"fp16", "float16", "half"}

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_config.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(train_config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_config.get("learning_rate", 2e-4)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.03)),
        logging_steps=int(train_config.get("logging_steps", 10)),
        save_steps=int(train_config.get("save_steps", 100)),
        eval_steps=int(train_config.get("eval_steps", 100)),
        save_total_limit=int(train_config.get("save_total_limit", 2)),
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to=normalize_string_list(train_config.get("report_to"), default=()),
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        logging_strategy="steps",
        optim=str(train_config.get("optim", "adamw_torch")),
        lr_scheduler_type=str(train_config.get("lr_scheduler_type", "cosine")),
        dataloader_num_workers=int(train_config.get("dataloader_num_workers", 0)),
        run_name=str(train_config.get("run_name", "qwen2audio-lora")),
        seed=seed,
    )

    collator = Qwen2AudioCollator(
        processor=processor,
        sampling_rate=int(processor.feature_extractor.sampling_rate),
        system_prompt=system_prompt,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    return trainer, processor, output_dir, {
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "model_name_or_path": model_name_or_path,
        "cache_dir": cache_dir,
        "system_prompt": system_prompt,
    }


def describe_plan(config: dict[str, Any], config_path: Path) -> None:
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    train_config = config.get("train", {})

    train_task_files = normalize_string_list(data_config.get("train_task_files"))
    eval_task_files = normalize_string_list(data_config.get("eval_task_files"))
    test_task_files = normalize_string_list(data_config.get("test_task_files"))
    prompt_template = str(data_config.get("prompt_template", "请根据语音内容回答问题。"))
    response_key = str(data_config.get("response_key", "texts"))
    response_index = int(data_config.get("response_index", 0))

    train_samples = load_samples_from_files(train_task_files, prompt_template, response_key, response_index) if train_task_files else []
    eval_samples = load_samples_from_files(eval_task_files, prompt_template, response_key, response_index) if eval_task_files else []
    test_samples = load_samples_from_files(test_task_files, "", "label", 0) if test_task_files else []

    summary = {
        "config": str(config_path),
        "project_root": str(PROJECT_ROOT),
        "model_name_or_path": model_config.get("model_name_or_path", "Qwen/Qwen2-Audio-7B-Instruct"),
        "cache_dir": str(resolve_path(model_config.get("cache_dir"))) if model_config.get("cache_dir") else None,
        "output_dir": str(resolve_path(train_config.get("output_dir", "model/qwen2audio_lora"))),
        "train_examples": len(train_samples),
        "eval_examples": len(eval_samples),
        "test_examples": len(test_samples),
        "train_task_files": [str(resolve_path(task_file)) for task_file in train_task_files],
        "eval_task_files": [str(resolve_path(task_file)) for task_file in eval_task_files],
        "test_task_files": [str(resolve_path(task_file)) for task_file in test_task_files],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_training(config: dict[str, Any]) -> tuple[Any, Any, Path]:
    trainer, processor, output_dir, metadata = build_trainer(config)
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "train_examples": len(metadata["train_samples"]),
                "eval_examples": len(metadata["eval_samples"]),
                "model_name_or_path": metadata["model_name_or_path"],
                "cache_dir": metadata["cache_dir"],
                "system_prompt": metadata["system_prompt"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return trainer.model, processor, output_dir


def run_test_report(
    config: dict[str, Any],
    model: Any | None = None,
    processor: Any | None = None,
) -> dict[str, Any]:
    data_config = config.get("data", {})
    analysis_config = config.get("analysis", {})
    system_prompt = analysis_config.get("test_system_prompt", data_config.get("system_prompt"))
    if system_prompt is not None:
        system_prompt = str(system_prompt)

    test_samples, label_names = load_test_samples_for_report(config)
    if model is None or processor is None:
        model, processor = load_finetuned_model_and_processor(config)

    records = collect_test_predictions(
        model=model,
        processor=processor,
        samples=test_samples,
        label_names=label_names,
        analysis_config=analysis_config,
        system_prompt=system_prompt,
    )

    report_root = resolve_analysis_output_dir(config)
    report_dir = report_root / build_task_slug(normalize_string_list(data_config.get("test_task_files")))
    summary = write_test_report(records, label_names, report_dir)
    summary.update(
        {
            "report_dir": str(report_dir),
            "test_task_files": [str(resolve_path(task_file)) for task_file in normalize_string_list(data_config.get("test_task_files"))],
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
