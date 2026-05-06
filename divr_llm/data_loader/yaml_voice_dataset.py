from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VoiceSample:
    sample_id: str
    audio_paths: list[str]
    texts: list[str]
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(str(item) for item in self.texts if item is not None).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def load_voice_yaml(path: str | Path) -> list[VoiceSample]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a top-level YAML mapping.")

    samples: list[VoiceSample] = []
    for sample_id, item in raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{sample_id} must be a mapping.")

        audio_paths = [str(value) for value in _as_list(item.get("text_keys")) if value]
        texts = [str(value) for value in _as_list(item.get("texts")) if value is not None]
        label = item.get("label")
        if not audio_paths:
            raise ValueError(f"{path}:{sample_id} has no audio path in text_keys.")
        if label is None:
            raise ValueError(f"{path}:{sample_id} has no label.")

        metadata = {key: value for key, value in item.items() if key not in {"text_keys", "texts", "label"}}
        samples.append(
            VoiceSample(
                sample_id=str(sample_id),
                audio_paths=audio_paths,
                texts=texts,
                label=str(label),
                metadata=metadata,
            )
        )

    return samples


def collect_labels(sample_sets: Iterable[Iterable[VoiceSample]]) -> list[str]:
    labels = {sample.label for samples in sample_sets for sample in samples}
    return sorted(labels)


class VoiceYamlDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.samples = load_voice_yaml(self.path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> VoiceSample:
        return self.samples[index]
