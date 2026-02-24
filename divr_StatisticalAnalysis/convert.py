import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


def load_split(yml_path: Path, audio_suffix: str) -> list[tuple[int, str, str, str]]:
    with yml_path.open("r", encoding="utf-8") as f:
        data: dict[str, dict[str, Any]] = yaml.safe_load(f)

    converted: list[tuple[int, str, str, str]] = []
    for value in data.values():
        age = value["age"]
        gender = value["gender"]
        label = value["label"]
        audio_keys = value.get("audio_keys", [])

        for audio_key in audio_keys:
            if isinstance(audio_key, str) and audio_key.endswith(audio_suffix):
                converted.append((age, gender, label, audio_key))

    return converted


def format_dataset_block(name: str, rows: list[tuple[int, str, str, str]]) -> str:
    lines = [f"{name}: Dataset = ["]
    lines.extend(
        f'    ({age}, "{gender}", "{label}", "{audio_key}"),'
        for age, gender, label, audio_key in rows
    )
    lines.append("]")
    return "\n".join(lines)


def build_output(
    train: list[tuple[int, str, str, str]],
    val: list[tuple[int, str, str, str]],
    test: list[tuple[int, str, str, str]],
) -> str:
    header = [
        '"""',
        "Task 1",
        "",
        "This task includes files from SVD with phrases as inputs",
        '"""',
        "",
        "from ...data_loader import Database, Dataset",
        "",
    ]

    blocks = [
        format_dataset_block("train", train),
        format_dataset_block("val", val),
        format_dataset_block("test", test),
        "database: Database = (train, val, test)",
        "",
    ]
    return "\n".join(header + blocks)


def write_csv_output(
    csv_path: Path,
    train: list[tuple[int, str, str, str]],
    val: list[tuple[int, str, str, str]],
    test: list[tuple[int, str, str, str]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "age", "gender", "label", "audio_key"])
        for split_name, rows in (("train", train), ("val", val), ("test", test)):
            for age, gender, label, audio_key in rows:
                writer.writerow([split_name, age, gender, label, audio_key])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SVD train/val/test YAMLs to task_1.py format"
    )
    parser.add_argument(
        "--input-dir",
        default="tasks/svd_tasks",
        help="Directory containing train.yml, val.yml, and test.yml",
    )
    parser.add_argument(
        "--output",
        default="task_1_generated.py",
        help="Output Python file path",
    )
    parser.add_argument(
        "--output-format",
        choices=["py", "csv", "both"],
        default="py",
        help="Output format: py (task_1 style), csv, or both",
    )
    parser.add_argument(
        "--csv-output",
        default="task_1_generated.csv",
        help="Output CSV file path when output format includes csv",
    )
    parser.add_argument(
        "--audio-suffix",
        default="-phrase.wav",
        help="Only keep audio paths ending with this suffix",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    train = load_split(input_dir / "train.yml", args.audio_suffix)
    val = load_split(input_dir / "val.yml", args.audio_suffix)
    test = load_split(input_dir / "test.yml", args.audio_suffix)

    if args.output_format in ("py", "both"):
        output_text = build_output(train, val, test)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"Saved py: {output_path}")

    if args.output_format in ("csv", "both"):
        csv_output_path = Path(args.csv_output)
        write_csv_output(csv_output_path, train, val, test)
        print(f"Saved csv: {csv_output_path}")

    print(f"train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()

# Example usage:
# python convert.py --input-dir ./tasks/svd_tasks --output task_1.py --audio-suffix -phrase.wav
# python convert.py --input-dir ./tasks/svd_tasks --output-format csv --csv-output task_1.csv
# python convert.py --input-dir ./tasks/svd_tasks --output-format both --output task_1.py --csv-output task_1.csv
