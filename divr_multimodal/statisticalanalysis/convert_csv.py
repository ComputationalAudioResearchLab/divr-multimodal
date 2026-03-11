import csv
from pathlib import Path
from typing import Iterable

from divr_diagnosis import DiagnosisMap

from ..task_generator.generator import Generator
from ..task_generator.task import Task
from ..task_generator.databases import (
    FEMH,
    SVD,
    Voiced,
    Base as Database,
)


DB_MAP: dict[str, type[Database]] = {
    FEMH.DB_NAME: FEMH,
    SVD.DB_NAME: SVD,
    Voiced.DB_NAME: Voiced,
}


def parse_label_filter(labels: list[str] | None) -> set[str] | None:
    if labels is None:
        return None
    parsed: set[str] = set()
    for label in labels:
        parts = [item.strip() for item in label.split(",")]
        parsed.update(item for item in parts if item)
    return parsed if len(parsed) > 0 else None


def normalize_databases(databases: list[str] | None) -> list[str]:
    if databases is None:
        return list(DB_MAP.keys())

    selected = [name.strip().lower() for name in databases if name.strip()]
    if len(selected) == 0:
        raise ValueError("datasets cannot be empty")

    unsupported = [name for name in selected if name not in DB_MAP]
    if len(unsupported) > 0:
        raise ValueError(
            f"Unsupported database(s): {unsupported}. "
            f"Supported: {list(DB_MAP.keys())}"
        )
    return selected


def task_rows(
    tasks: Iterable[Task],
    split: str,
    selected_fields: list[str],
    text_equals: list[tuple[str | None, str, str]] | None,
    label_filter: set[str] | None,
    parser: Generator,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task in tasks:
        label_name = task.label.name
        if label_filter is not None and label_name not in label_filter:
            continue

        for text_index, payload in enumerate(task.texts):
            metadata = parser._task_text_metadata(task=task, payload=payload)

            if text_equals is not None:
                dataset_name = metadata.get("dataset", "").strip().lower()
                matched = True
                for scope, key, expected in text_equals:
                    if scope is not None and dataset_name != scope:
                        continue
                    if metadata.get(key) != expected:
                        matched = False
                        break
                if not matched:
                    continue

            row: dict[str, str] = {
                "split": split,
                "task_id": task.id,
                "text": payload,
                "label": label_name,
            }
            for field in selected_fields:
                value = metadata.get(field)
                row[field] = "" if value is None else str(value)
            rows.append(row)
    return rows


async def convert_text_csv(
    source_path: Path,
    output_csv_path: Path,
    diagnosis_map: DiagnosisMap,
    diag_level: int = 0,
    databases: list[str] | None = None,
    text_fields: list[str] | None = None,
    text_equals: list[str] | None = None,
    labels: list[str] | None = None,
) -> None:
    if not source_path.is_dir():
        raise ValueError(f"source_path does not exist: {source_path}")
    if diag_level < 0:
        raise ValueError("diag_level must be >= 0")

    parser = Generator()
    normalized_fields = parser.normalize_text_fields(text_fields)
    normalized_equals = parser.normalize_text_equals(text_equals)
    selected_fields = (
        sorted(parser._supported_text_fields)
        if normalized_fields is None
        else normalized_fields
    )
    selected_dbs = normalize_databases(databases)
    selected_labels = parse_label_filter(labels)

    rows: list[dict[str, str]] = []
    for db_name in selected_dbs:
        db = DB_MAP[db_name](source_path=source_path)
        await db.init(
            diagnosis_map=diagnosis_map,
            allow_incomplete_classification=False,
            min_tasks=None,
        )

        rows.extend(
            task_rows(
                tasks=db.all_train(level=diag_level),
                split="train",
                selected_fields=selected_fields,
                text_equals=normalized_equals,
                label_filter=selected_labels,
                parser=parser,
            )
        )
        rows.extend(
            task_rows(
                tasks=db.all_val(level=diag_level),
                split="val",
                selected_fields=selected_fields,
                text_equals=normalized_equals,
                label_filter=selected_labels,
                parser=parser,
            )
        )
        rows.extend(
            task_rows(
                tasks=db.all_test(level=diag_level),
                split="test",
                selected_fields=selected_fields,
                text_equals=normalized_equals,
                label_filter=selected_labels,
                parser=parser,
            )
        )

    fieldnames = ["split", "task_id", "label", "text"]
    for field in selected_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved csv: {output_csv_path}")
    print(f"rows={len(rows)}")
    if selected_labels is not None:
        print(f"label filter: {sorted(selected_labels)}")
    if normalized_equals is not None:
        print(f"text_equals: {normalized_equals}")
    print(f"text_fields: {selected_fields}")
    print(f"datasets: {selected_dbs}")
