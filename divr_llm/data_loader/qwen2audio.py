from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Qwen2AudioSample:
    source_file: Path
    example_id: str
    audio_paths: tuple[Path, ...]
    prompt: str
    target_text: str
    label: str | None = None


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
        label_value = payload.get("label")

        samples.append(
            Qwen2AudioSample(
                source_file=task_file,
                example_id=str(example_id),
                audio_paths=audio_paths,
                prompt=prompt,
                target_text=target_text,
                label=None if label_value is None else str(label_value),
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
