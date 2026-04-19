from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

from data_loader.dtypes import (
    Batch,
    BatchMetadata,
    DemographicTensors,
    InputTensors,
    TaskRecord,
)


RESERVED_TASK_KEYS = {"label", "text_keys", "texts"}


def parse_text_payload(payload: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for part in payload.split(";"):
        entry = part.strip()
        if not entry or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        normalized_key = key.strip().lower()
        normalized_value = value.strip()
        if normalized_key:
            metadata[normalized_key] = normalized_value
    return metadata


def normalize_text_fields(
    text_fields: Sequence[str] | None,
) -> list[str] | None:
    if text_fields is None:
        return None
    normalized = [
        field.strip().lower()
        for field in text_fields
        if field.strip()
    ]
    if not normalized:
        raise ValueError("text_fields cannot be empty")
    if len(normalized) == 1 and normalized[0] == "all":
        return None
    return normalized


def normalize_text_equals(
    text_equals: Sequence[str] | None,
) -> list[tuple[str | None, str, str]] | None:
    if text_equals is None:
        return None
    normalized: list[tuple[str | None, str, str]] = []
    for item in text_equals:
        if item is None:
            continue
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                "Invalid --text-equals entry. Expected key=value or "
                "dataset.key=value."
            )
        raw_key, raw_value = entry.split("=", 1)
        key = raw_key.strip().lower()
        value = raw_value.strip()
        scope: str | None = None
        field = key
        if "." in key:
            scope, field = key.split(".", 1)
            scope = scope.strip() or None
            field = field.strip()
        if not field:
            raise ValueError(f"Invalid --text-equals key: {entry}")
        normalized.append((scope, field, value))
    return normalized or None


def try_parse_age(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


@dataclass(slots=True)
class TaskSample:
    sample_id: str
    label: str
    audio_paths: list[str]
    selected_text: str
    metadata: dict[str, Any]


class TaskDataset(TorchDataset[TaskSample]):
    def __init__(
        self,
        records: Sequence[TaskRecord],
        include_audio: bool,
        include_text: bool,
        text_fields: Sequence[str] | None,
        text_equals: Sequence[str] | None,
    ) -> None:
        self.include_audio = include_audio
        self.include_text = include_text
        self.text_fields = (
            normalize_text_fields(text_fields)
            if include_text
            else None
        )
        self.text_equals = (
            normalize_text_equals(text_equals)
            if include_text
            else None
        )
        self.samples = self._build_samples(records)
        if not self.samples:
            raise ValueError(
                "No usable samples were found after applying text filters"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TaskSample:
        return self.samples[index]

    def _build_samples(
        self,
        records: Sequence[TaskRecord],
    ) -> list[TaskSample]:
        samples: list[TaskSample] = []
        for record in records:
            selected_text = ""
            if self.include_text:
                selected_text_entries = self._select_text_entries(record)
                selected_text = " ; ".join(selected_text_entries).strip()
            if self.include_audio and not record.audio_paths:
                continue
            if self.include_text and not selected_text:
                continue
            samples.append(
                TaskSample(
                    sample_id=record.sample_id,
                    label=record.label,
                    audio_paths=list(record.audio_paths),
                    selected_text=selected_text,
                    metadata=dict(record.metadata),
                )
            )
        return samples

    def _select_text_entries(self, record: TaskRecord) -> list[str]:
        if not record.texts:
            if self.text_fields is None:
                return []
            fallback = self._select_fields(record.metadata)
            return [fallback] if fallback else []

        selected_entries: list[str] = []
        for payload in record.texts:
            merged_metadata = self._payload_metadata(
                record=record,
                payload=payload,
            )
            if not self._matches_text_equals(merged_metadata):
                continue
            if self.text_fields is None:
                entry = payload.strip()
            else:
                entry = self._select_fields(merged_metadata)
            if entry:
                selected_entries.append(entry)
        return selected_entries

    def _payload_metadata(
        self,
        record: TaskRecord,
        payload: str,
    ) -> dict[str, str]:
        payload_metadata = parse_text_payload(payload)
        for key, value in record.metadata.items():
            normalized_key = str(key).strip().lower()
            if (
                normalized_key
                and normalized_key not in payload_metadata
                and value is not None
            ):
                payload_metadata[normalized_key] = str(value)
        if "label" not in payload_metadata:
            payload_metadata["label"] = record.label
        if "sample_id" not in payload_metadata:
            payload_metadata["sample_id"] = record.sample_id
        return payload_metadata

    def _matches_text_equals(self, payload_metadata: dict[str, str]) -> bool:
        if self.text_equals is None:
            return True
        dataset_name = payload_metadata.get("dataset", "").strip().lower()
        for scope, key, expected in self.text_equals:
            if scope is not None and dataset_name != scope:
                continue
            if payload_metadata.get(key) != expected:
                return False
        return True

    def _select_fields(self, payload_metadata: dict[str, Any]) -> str:
        assert self.text_fields is not None
        selected: list[str] = []
        for key in self.text_fields:
            value = payload_metadata.get(key)
            if value is None:
                continue
            text_value = str(value).strip()
            if text_value:
                selected.append(f"{key}={text_value}")
        return "; ".join(selected)


class TaskDataModule:
    def __init__(
        self,
        task_dir: Path,
        sample_rate: int,
        batch_size: int,
        random_seed: int,
        include_audio: bool,
        include_text: bool,
        text_fields: Sequence[str] | None,
        text_equals: Sequence[str] | None,
        num_workers: int = 0,
    ) -> None:
        self.task_dir = task_dir
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.include_audio = include_audio
        self.include_text = include_text
        self.num_workers = num_workers

        train_records = self._load_split("train")
        val_records = self._load_split("val")
        test_records = self._load_split("test")

        self.train_dataset = TaskDataset(
            records=train_records,
            include_audio=include_audio,
            include_text=include_text,
            text_fields=text_fields,
            text_equals=text_equals,
        )
        self.val_dataset = TaskDataset(
            records=val_records,
            include_audio=include_audio,
            include_text=include_text,
            text_fields=text_fields,
            text_equals=text_equals,
        )
        self.test_dataset = TaskDataset(
            records=test_records,
            include_audio=include_audio,
            include_text=include_text,
            text_fields=text_fields,
            text_equals=text_equals,
        )

        label_names = sorted(
            {
                sample.label
                for dataset in (
                    self.train_dataset,
                    self.val_dataset,
                    self.test_dataset,
                )
                for sample in dataset.samples
            }
        )
        self.label_to_index = {
            label: index for index, label in enumerate(label_names)
        }
        self.label_names = label_names
        self.unique_diagnosis = label_names

        train_label_ids = [
            self.label_to_index[sample.label]
            for sample in self.train_dataset.samples
        ]
        counts = np.bincount(train_label_ids, minlength=len(self.label_names))
        self.class_counts = torch.tensor(counts, dtype=torch.long)

    def train(self) -> TorchDataLoader[Batch]:
        return self._make_loader(self.train_dataset, shuffle=True)

    def eval(self) -> TorchDataLoader[Batch]:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test(self) -> TorchDataLoader[Batch]:
        return self._make_loader(self.test_dataset, shuffle=False)

    def _make_loader(
        self,
        dataset: TaskDataset,
        shuffle: bool,
    ) -> TorchDataLoader[Batch]:
        generator = torch.Generator()
        generator.manual_seed(self.random_seed)
        return TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_batch,
            generator=generator,
        )

    def _load_split(self, split: str) -> list[TaskRecord]:
        file_path = self.task_dir / f"{split}.yml"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Task split file does not exist: {file_path}"
            )
        with open(file_path, "r", encoding="utf-8") as handle:
            raw_data = yaml.safe_load(handle) or {}
        records: list[TaskRecord] = []
        for sample_id, payload in raw_data.items():
            label = str(payload.get("label", "")).strip()
            if not label:
                continue
            audio_paths = [
                self._resolve_audio_path(path)
                for path in payload.get("text_keys", [])
            ]
            texts = [str(text) for text in payload.get("texts", [])]
            metadata = {
                key: value
                for key, value in payload.items()
                if key not in RESERVED_TASK_KEYS
            }
            records.append(
                TaskRecord(
                    sample_id=str(sample_id),
                    label=label,
                    audio_paths=audio_paths,
                    texts=texts,
                    metadata=metadata,
                )
            )
        return records

    def _resolve_audio_path(self, path_value: Any) -> str:
        path_text = str(path_value)
        path = Path(path_text)
        if path.is_absolute():
            return str(path)
        return str((self.task_dir.parent / path).resolve())

    def _collate_batch(self, batch: Sequence[TaskSample]) -> Batch:
        sample_ids = [sample.sample_id for sample in batch]
        labels = torch.tensor(
            [self.label_to_index[sample.label] for sample in batch],
            dtype=torch.long,
        )

        audio_inputs = None
        if self.include_audio:
            audio_inputs = self._collate_audio(batch)

        demographic_inputs = None
        if self.include_text:
            demographic_inputs = self._collate_demographics(batch)

        metadata = self._collate_metadata(batch)
        return Batch(
            sample_ids=sample_ids,
            labels=labels,
            audio_inputs=audio_inputs,
            demographic_inputs=demographic_inputs,
            audio_paths=[list(sample.audio_paths) for sample in batch],
            selected_texts=[sample.selected_text for sample in batch],
            metadata=metadata,
        )

    def _collate_audio(self, batch: Sequence[TaskSample]) -> InputTensors:
        audios = [self._load_audio(sample.audio_paths) for sample in batch]
        max_audio_len = max(audio.shape[0] for audio in audios)
        audio_tensor = np.zeros((len(batch), max_audio_len), dtype=np.float32)
        audio_lens = np.zeros((len(batch),), dtype=np.int64)
        for index, audio in enumerate(audios):
            audio_len = audio.shape[0]
            audio_tensor[index, :audio_len] = audio
            audio_lens[index] = audio_len
        return (
            torch.tensor(audio_tensor, dtype=torch.float32),
            torch.tensor(audio_lens, dtype=torch.long),
        )

    def _collate_metadata(self, batch: Sequence[TaskSample]) -> BatchMetadata:
        keys = sorted(
            {key for sample in batch for key in sample.metadata.keys()}
        )
        metadata: BatchMetadata = {
            key: [sample.metadata.get(key) for sample in batch] for key in keys
        }
        ages = [try_parse_age(sample.metadata.get("age")) for sample in batch]
        if any(age is not None for age in ages):
            metadata["age"] = ages
        return metadata

    def _collate_demographics(
        self,
        batch: Sequence[TaskSample],
    ) -> DemographicTensors:
        gender_to_id = {
            "male": 0,
            "m": 0,
            "female": 1,
            "f": 1,
        }
        smoking_to_id = {
            "never": 0,
            "past": 1,
            "active": 2,
            "e-cigarette": 3,
            "unknown": 4,
        }
        drinking_to_id = {
            "never": 0,
            "past": 1,
            "active": 2,
            "unknown": 3,
        }

        ages: list[int] = []
        genders: list[int] = []
        smokings: list[int] = []
        drinkings: list[int] = []
        for sample in batch:
            age_value = try_parse_age(sample.metadata.get("age"))
            ages.append(-1 if age_value is None else int(age_value))

            raw_gender = str(
                sample.metadata.get("gender", "unknown")
            ).strip().lower()
            genders.append(gender_to_id.get(raw_gender, 2))

            raw_smoking = str(
                sample.metadata.get("smoking", "unknown")
            ).strip().lower()
            smokings.append(smoking_to_id.get(raw_smoking, 4))

            raw_drinking = str(
                sample.metadata.get("drinking", "unknown")
            ).strip().lower()
            drinkings.append(drinking_to_id.get(raw_drinking, 3))

        return (
            torch.tensor(ages, dtype=torch.long),
            torch.tensor(genders, dtype=torch.long),
            torch.tensor(smokings, dtype=torch.long),
            torch.tensor(drinkings, dtype=torch.long),
        )

    def _load_audio(self, audio_paths: Sequence[str]) -> np.ndarray:
        if not audio_paths:
            raise ValueError(
                "Audio-enabled runs require at least one audio path"
            )
        segments: list[np.ndarray] = []
        silence = np.zeros((max(1, self.sample_rate // 20),), dtype=np.float32)
        for path in audio_paths:
            audio, _ = librosa.load(path=path, sr=self.sample_rate)
            segments.append(self._normalize(audio.astype(np.float32)))
        if len(segments) == 1:
            return segments[0]
        stitched: list[np.ndarray] = []
        for index, segment in enumerate(segments):
            if index > 0:
                stitched.append(silence)
            stitched.append(segment)
        return np.concatenate(stitched)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        stddev = float(np.std(audio))
        if stddev < 1e-8:
            return audio - float(np.mean(audio))
        return (audio - float(np.mean(audio))) / stddev


DataLoader = TaskDataModule
DataLoaderWithFeature = TaskDataModule
DataLoaderWithDemographics = TaskDataModule
DataLoaderWithFeatureAndDemographics = TaskDataModule
