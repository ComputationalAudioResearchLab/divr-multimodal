import yaml
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import List

from .task import Task
from .databases import Base as Database
from typing import Protocol


class DatabaseFunc(Protocol):
    async def __call__(
        self,
        name: str,
        min_tasks: int | None = None,
    ) -> Database:
        ...


@dataclass
class Dataset:
    train: List[Task]
    val: List[Task]
    test: List[Task]


class Generator:
    _common_text_fields = {
        "dataset",
        "speaker_id",
        "age",
        "gender",
        "original_label",
        "label",
    }
    _private_text_fields_by_dataset = {
        "femh": {
            "smoking",
            "drinking",
        },
        "svd": {
            "svd_utterance",
        },
        "voiced": {
            "smoking",
            "drinking",
        },
    }
    _supported_dataset_scopes = set(_private_text_fields_by_dataset.keys())
    _supported_text_fields = {
        *_common_text_fields,
        *{
            field
            for fields in _private_text_fields_by_dataset.values()
            for field in fields
        },
    }

    def to_task_file(
        self,
        tasks: List[Task],
        output_path: Path,
        text_fields: List[str] | None = None,
        text_equals: list[tuple[str | None, str, str]] | None = None,
        labels: set[str] | None = None,
    ) -> None:
        tasks_dict = {}
        exported_tasks: List[Task] = []
        for task in tasks:
            if labels is not None and task.label.name not in labels:
                continue
            if len(task.text_keys) < 1:
                raise ValueError(f"Invalid task (no text keys): {task.id}")
            if len(task.texts) < 1:
                raise ValueError(f"Invalid task (no texts): {task.id}")

            task = self._apply_text_equals(task=task, text_equals=text_equals)
            if task is None:
                continue

            task = self._apply_text_fields(task=task, text_fields=text_fields)
            task_data = task.__dict__.copy()
            del task_data["id"]
            task_data["label"] = task.label.name
            tasks_dict[task.id] = task_data
            exported_tasks += [task]
        with open(f"{output_path}.yml", "w") as output_file:
            yaml.dump(tasks_dict, output_file)
        self.generate_demographics(
            tasks=exported_tasks,
            output_path=output_path,
        )

    def generate_demographics(
        self,
        tasks: List[Task],
        output_path: Path,
    ) -> None:
        demographics = {}
        for task in tasks:
            diagnosis_name = task.label.name
            if diagnosis_name not in demographics:
                demographics[diagnosis_name] = {}
            diagnosis = demographics[diagnosis_name]
            gender = task.gender
            if gender not in diagnosis:
                diagnosis[gender] = {"ages": [], "total": 0}
            diagnosis[gender]["total"] += 1
            if task.age is not None:
                diagnosis[gender]["ages"] += [task.age]
        for diagnosis in demographics:
            for gender in demographics[diagnosis]:
                ages = demographics[diagnosis][gender]["ages"]
                total = demographics[diagnosis][gender]["total"]
                total_ages = len(ages)
                age_stats = None
                if total_ages > 0:
                    age_stats = {
                        "mean": statistics.mean(ages),
                        "std": (
                            statistics.stdev(ages) if total_ages > 1 else None
                        ),
                        "min": min(ages),
                        "max": max(ages),
                    }

                demographics[diagnosis][gender] = {
                    "total": total,
                    "age_stats": age_stats,
                }
        with open(f"{output_path}.demographics.yml", "w") as output_file:
            yaml.dump(demographics, output_file)

    @classmethod
    def normalize_text_fields(
        cls,
        text_fields: List[str] | None,
    ) -> List[str] | None:
        if text_fields is None:
            return None
        normalized = [
            field.strip().lower()
            for field in text_fields
            if field.strip()
        ]
        if len(normalized) == 1 and normalized[0] == "all":
            return None
        if len(normalized) == 0:
            raise ValueError("text_fields cannot be empty")
        invalid = [
            field
            for field in normalized
            if field not in cls._supported_text_fields
        ]
        if len(invalid) > 0:
            raise ValueError(
                f"Unsupported text_fields: {invalid}. "
                f"Supported: {sorted(cls._supported_text_fields)}"
            )
        return normalized

    @classmethod
    def normalize_text_equals(
        cls,
        text_equals: List[str] | None,
    ) -> list[tuple[str | None, str, str]] | None:
        if text_equals is None:
            return None

        normalized: list[tuple[str | None, str, str]] = []
        seen_global_keys: set[str] = set()
        seen_scoped_keys: set[tuple[str, str]] = set()
        for item in text_equals:
            if item is None:
                continue
            entries = [entry.strip() for entry in item.split(",")]
            for entry in entries:
                if entry == "":
                    continue
                if "=" not in entry:
                    raise ValueError(
                        "Invalid text_equals entry. "
                        "Expected key=value or dataset.key=value, "
                        f"got: {entry}"
                    )
                key, value = entry.split("=", 1)
                normalized_key = key.strip().lower()
                normalized_value = value.strip()
                if normalized_key == "":
                    raise ValueError(
                        f"Invalid text_equals key in entry: {entry}"
                    )

                scope: str | None = None
                field_key = normalized_key
                if "." in normalized_key:
                    scope_part, field_part = normalized_key.split(".", 1)
                    scope = scope_part.strip()
                    field_key = field_part.strip()
                    if scope == "" or field_key == "":
                        raise ValueError(
                            f"Invalid scoped text_equals key: {normalized_key}"
                        )
                    if scope not in cls._supported_dataset_scopes:
                        raise ValueError(
                            f"Unsupported text_equals scope: {scope}. "
                            "Supported scopes: "
                            f"{sorted(cls._supported_dataset_scopes)}"
                        )
                    supported_scoped_fields = (
                        cls._common_text_fields
                        | cls._private_text_fields_by_dataset[scope]
                    )
                    if field_key not in supported_scoped_fields:
                        raise ValueError(
                            "Unsupported scoped text_equals key: "
                            f"{scope}.{field_key}. "
                            f"Supported for {scope}: "
                            f"{sorted(supported_scoped_fields)}"
                        )

                if field_key not in cls._supported_text_fields:
                    raise ValueError(
                        f"Unsupported text_equals key: {field_key}. "
                        f"Supported: {sorted(cls._supported_text_fields)}"
                    )
                if normalized_value == "":
                    raise ValueError(
                        f"Invalid text_equals value for key: {normalized_key}"
                    )

                if scope is None:
                    if field_key in seen_global_keys:
                        raise ValueError(
                            "Duplicate text_equals key is not allowed: "
                            f"{field_key}"
                        )
                    seen_global_keys.add(field_key)
                else:
                    scoped_key = (scope, field_key)
                    if scoped_key in seen_scoped_keys:
                        raise ValueError(
                            "Duplicate scoped text_equals key is not allowed: "
                            f"{scope}.{field_key}"
                        )
                    seen_scoped_keys.add(scoped_key)

                normalized += [(scope, field_key, normalized_value)]

        if len(normalized) == 0:
            raise ValueError("text_equals cannot be empty")
        return normalized

    @classmethod
    def normalize_labels(
        cls,
        labels: List[str] | None,
    ) -> set[str] | None:
        if labels is None:
            return None

        normalized: set[str] = set()
        for label in labels:
            if label is None:
                continue
            entries = [entry.strip() for entry in label.split(",")]
            normalized.update(entry for entry in entries if entry)

        return normalized if len(normalized) > 0 else None

    def _apply_text_fields(
        self,
        task: Task,
        text_fields: List[str] | None,
    ) -> Task:
        if text_fields is None:
            return task
        updated_texts = []
        for payload in task.texts:
            metadata = self._parse_text_payload(payload=payload)
            if "speaker_id" not in metadata:
                metadata["speaker_id"] = str(task.speaker_id)
            if "age" not in metadata and task.age is not None:
                metadata["age"] = str(task.age)
            if "gender" not in metadata:
                metadata["gender"] = str(task.gender)
            if "diagnosis" not in metadata:
                metadata["diagnosis"] = task.label.name
            if "label" not in metadata:
                metadata["label"] = task.label.name

            selected = []
            for key in text_fields:
                value = metadata.get(key)
                if value is None or value == "":
                    continue
                selected.append(f"{key}={value}")

            if len(selected) > 0:
                updated_texts.append("; ".join(selected))
            else:
                updated_texts.append(payload)

        task.texts = updated_texts
        return task

    def _task_text_metadata(self, task: Task, payload: str) -> dict[str, str]:
        metadata = self._parse_text_payload(payload=payload)
        if "speaker_id" not in metadata:
            metadata["speaker_id"] = str(task.speaker_id)
        if "age" not in metadata and task.age is not None:
            metadata["age"] = str(task.age)
        if "gender" not in metadata:
            metadata["gender"] = str(task.gender)
        if "diagnosis" not in metadata:
            metadata["diagnosis"] = task.label.name
        if "label" not in metadata:
            metadata["label"] = task.label.name
        return metadata

    def _apply_text_equals(
        self,
        task: Task,
        text_equals: list[tuple[str | None, str, str]] | None,
    ) -> Task | None:
        if text_equals is None:
            return task

        filtered_texts: List[str] = []
        filtered_text_keys: List[str] = []
        for text_key, payload in zip(task.text_keys, task.texts):
            metadata = self._task_text_metadata(task=task, payload=payload)
            dataset_name = metadata.get("dataset", "").strip().lower()
            matches = True
            for scope, key, expected in text_equals:
                if scope is not None and dataset_name != scope:
                    continue
                if metadata.get(key) != expected:
                    matches = False
                    break
            if matches:
                filtered_text_keys += [text_key]
                filtered_texts += [payload]

        if len(filtered_texts) == 0:
            return None

        task.text_keys = filtered_text_keys
        task.texts = filtered_texts
        return task

    def _parse_text_payload(self, payload: str) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for part in payload.split(";"):
            item = part.strip()
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            metadata[key.strip().lower()] = value.strip()
        return metadata

    def filter_tasks_by_demographics(
        self,
        tasks: List[Task],
        genders: List[str] | None = None,
        min_age: int | None = None,
        max_age: int | None = None,
    ) -> List[Task]:
        normalized_genders = None
        if genders is not None:
            normalized_genders = {gender.strip().lower() for gender in genders}

        filtered: List[Task] = []
        for task in tasks:
            if normalized_genders is not None:
                if task.gender.strip().lower() not in normalized_genders:
                    continue

            if task.age is not None:
                if min_age is not None and task.age < min_age:
                    continue
                if max_age is not None and task.age > max_age:
                    continue
            elif min_age is not None or max_age is not None:
                continue

            filtered.append(task)
        return filtered

    def truncate_low_resource_classes(
        self, task_list: List[List[Task]], min_examples: int
    ) -> List[List[Task]]:
        to_remove = []
        for tasks in task_list:
            counts = {}
            for task in tasks:
                diagnosis_name = task.label.name
                if diagnosis_name not in counts:
                    counts[diagnosis_name] = 0
                counts[diagnosis_name] += 1
            for label, count in counts.items():
                if count < min_examples:
                    to_remove += [label]
        new_list = []
        for tasks in task_list:
            new_tasks = []
            for task in tasks:
                diagnosis_name = task.label.name
                if diagnosis_name not in to_remove:
                    new_tasks += [task]
            new_list += [new_tasks]
        return new_list
