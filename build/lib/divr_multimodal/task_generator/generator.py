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
    _supported_text_fields = {
        "dataset",
        "speaker_id",
        "session_id",
        "age",
        "gender",
        "diagnosis",
        "classification",
        "pathologies",
        "utterance",
        "smoking",
    }

    def to_task_file(
        self,
        tasks: List[Task],
        output_path: Path,
        text_fields: List[str] | None = None,
    ) -> None:
        tasks_dict = {}
        for task in tasks:
            if len(task.text_keys) < 1:
                raise ValueError(f"Invalid task (no text keys): {task.id}")
            if len(task.texts) < 1:
                raise ValueError(f"Invalid task (no texts): {task.id}")
            task = self._apply_text_fields(task=task, text_fields=text_fields)
            task_data = task.__dict__.copy()
            del task_data["id"]
            task_data["label"] = task.label.name
            tasks_dict[task.id] = task_data
        with open(f"{output_path}.yml", "w") as output_file:
            yaml.dump(tasks_dict, output_file)
        self.generate_demographics(tasks=tasks, output_path=output_path)

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
0