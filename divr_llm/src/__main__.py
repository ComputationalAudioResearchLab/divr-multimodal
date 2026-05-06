from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "lora" / "qwen2audio.yaml"


@dataclass(frozen=True)
class Qwen2AudioSample:
	source_file: Path
	example_id: str
	audio_paths: tuple[Path, ...]
	prompt: str
	target_text: str


class SafeFormatDict(dict[str, Any]):
	def __missing__(self, key: str) -> str:
		return "{" + key + "}"


def read_yaml(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as handle:
		return yaml.safe_load(handle)


def resolve_path(value: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
	path = Path(value)
	if path.is_absolute():
		return path.resolve()
	return (base_dir / path).resolve()


def normalize_text_list(value: Any, field_name: str) -> list[str]:
	if isinstance(value, str):
		return [value]
	if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
		return [str(item) for item in value]
	raise ValueError(f"Expected {field_name} to be a string or sequence of strings")


def pick_response_text(value: Any, response_index: int, source_file: Path, example_id: str) -> str:
	if isinstance(value, str):
		return value
	if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
		if not value:
			raise ValueError(f"{source_file}::{example_id} has an empty response list")
		index = min(max(response_index, 0), len(value) - 1)
		return str(value[index])
	if value is None:
		raise ValueError(f"{source_file}::{example_id} is missing the configured response field")
	return str(value)


def load_task_samples(
	task_file: Path,
	prompt_template: str,
	response_key: str,
	response_index: int,
) -> list[Qwen2AudioSample]:
	raw_data = read_yaml(task_file)
	if not isinstance(raw_data, dict):
		raise ValueError(f"{task_file} must contain a YAML mapping of examples")

	samples: list[Qwen2AudioSample] = []
	for example_id, payload in raw_data.items():
		if not isinstance(payload, dict):
			raise ValueError(f"{task_file}::{example_id} must map to a YAML object")

		audio_values = payload.get("text_keys") or payload.get("audio_keys") or payload.get("audio_paths")
		if audio_values is None:
			raise ValueError(f"{task_file}::{example_id} is missing audio paths")

		audio_paths = tuple(
			resolve_path(audio_path, base_dir=task_file.parent)
			for audio_path in normalize_text_list(audio_values, "audio paths")
		)
		prompt = prompt_template.format_map(SafeFormatDict(payload))
		target_text = pick_response_text(payload.get(response_key), response_index, task_file, str(example_id))

		samples.append(
			Qwen2AudioSample(
				source_file=task_file,
				example_id=str(example_id),
				audio_paths=audio_paths,
				prompt=prompt,
				target_text=target_text,
			)
		)

	return samples


def load_samples_from_files(
	task_files: Sequence[str | Path],
	prompt_template: str,
	response_key: str,
	response_index: int,
) -> list[Qwen2AudioSample]:
	samples: list[Qwen2AudioSample] = []
	for task_file in task_files:
		resolved_task_file = resolve_path(task_file)
		samples.extend(load_task_samples(resolved_task_file, prompt_template, response_key, response_index))
	return samples


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


def load_audio_array(audio_path: Path, target_sample_rate: int) -> Any:
	try:
		import torchaudio
	except ImportError as exc:
		raise SystemExit("torchaudio is required to load local audio files") from exc

	waveform, sample_rate = torchaudio.load(str(audio_path))
	if waveform.ndim > 1 and waveform.size(0) > 1:
		waveform = waveform.mean(dim=0, keepdim=True)
	if sample_rate != target_sample_rate:
		waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
	return waveform.squeeze(0).contiguous().cpu().numpy()


class Qwen2AudioDataset:
	def __init__(self, samples: Sequence[Qwen2AudioSample]) -> None:
		self._samples = list(samples)

	def __len__(self) -> int:
		return len(self._samples)

	def __getitem__(self, index: int) -> Qwen2AudioSample:
		return self._samples[index]


class Qwen2AudioCollator:
	def __init__(self, processor: Any, sampling_rate: int, system_prompt: str | None = None) -> None:
		self.processor = processor
		self.sampling_rate = sampling_rate
		self.system_prompt = system_prompt

	def __call__(self, batch: Sequence[Qwen2AudioSample]) -> dict[str, Any]:
		prompt_texts: list[str] = []
		full_texts: list[str] = []
		prompt_lengths: list[int] = []
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

			prompt_texts.append(prompt_text)
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


def load_training_components() -> tuple[Any, Any, Any, Any, Any, Any]:
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

	output_dir = resolve_path(train_config.get("output_dir", "model/qwen2audio-lora"))
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


def load_config(path: Path) -> dict[str, Any]:
	config = read_yaml(path)
	if not isinstance(config, dict):
		raise ValueError(f"{path} must contain a YAML mapping")
	return config


def describe_plan(config: dict[str, Any], config_path: Path) -> None:
	data_config = config.get("data", {})
	model_config = config.get("model", {})
	train_config = config.get("train", {})

	train_task_files = normalize_string_list(data_config.get("train_task_files"))
	eval_task_files = normalize_string_list(data_config.get("eval_task_files"))
	prompt_template = str(data_config.get("prompt_template", "请根据语音内容回答问题。"))
	response_key = str(data_config.get("response_key", "texts"))
	response_index = int(data_config.get("response_index", 0))

	train_samples = load_samples_from_files(train_task_files, prompt_template, response_key, response_index) if train_task_files else []
	eval_samples = load_samples_from_files(eval_task_files, prompt_template, response_key, response_index) if eval_task_files else []

	summary = {
		"config": str(config_path),
		"project_root": str(PROJECT_ROOT),
		"model_name_or_path": model_config.get("model_name_or_path", "Qwen/Qwen2-Audio-7B-Instruct"),
		"cache_dir": str(resolve_path(model_config.get("cache_dir"))) if model_config.get("cache_dir") else None,
		"output_dir": str(resolve_path(train_config.get("output_dir", "model/qwen2audio-lora"))),
		"train_examples": len(train_samples),
		"eval_examples": len(eval_samples),
		"train_task_files": [str(resolve_path(task_file)) for task_file in train_task_files],
		"eval_task_files": [str(resolve_path(task_file)) for task_file in eval_task_files],
	}
	print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_training(config: dict[str, Any]) -> None:
	trainer, processor, output_dir, metadata = build_trainer(config)
	trainer.train()
	trainer.save_model(str(output_dir))
	processor.save_pretrained(str(output_dir))

	metadata_path = output_dir / "run_metadata.json"
	metadata_path.write_text(json.dumps({
		"train_examples": len(metadata["train_samples"]),
		"eval_examples": len(metadata["eval_samples"]),
		"model_name_or_path": metadata["model_name_or_path"],
		"cache_dir": metadata["cache_dir"],
		"system_prompt": metadata["system_prompt"],
	}, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Qwen2Audio LoRA training entrypoint for divr_llm")
	parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to a YAML config under divr_llm")
	parser.add_argument("--train", action="store_true", help="Run fine-tuning instead of printing the resolved plan")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	config_path = resolve_path(args.config)
	config = load_config(config_path)

	if not args.train:
		describe_plan(config, config_path)
		return 0

	run_training(config)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
