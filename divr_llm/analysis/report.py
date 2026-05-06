from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

UNKNOWN_LABEL = "__unknown__"
DEFAULT_LABEL_PROMPT_TEMPLATE = "请根据语音判断该样本属于哪个标签，只能从 {label_choices} 中选择，并只输出标签本身。"


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _unique_preserve_order(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def extract_label_from_text(raw_text: str, label_names: Sequence[str]) -> str:
    raw_lower = raw_text.casefold()
    normalized_text = _normalize_text(raw_text)
    ordered_labels = sorted(_unique_preserve_order(label_names), key=len, reverse=True)

    for label in ordered_labels:
        label_lower = label.casefold()
        if label_lower and label_lower in raw_lower:
            return label

    for label in ordered_labels:
        normalized_label = _normalize_text(label)
        if normalized_label and normalized_label in normalized_text:
            return label

    return UNKNOWN_LABEL


def _save_confusion_plot(confusion_frame: pd.DataFrame, output_path: Path) -> None:
    figure_size = max(6, len(confusion_frame) * 0.9)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size), constrained_layout=True)
    image = ax.imshow(confusion_frame.values, cmap=plt.cm.Blues, aspect="auto")
    ax.set_xticks(range(len(confusion_frame.columns)))
    ax.set_yticks(range(len(confusion_frame.index)))
    ax.set_xticklabels(confusion_frame.columns, rotation=45, ha="right")
    ax.set_yticklabels(confusion_frame.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for row_index in range(confusion_frame.shape[0]):
        for col_index in range(confusion_frame.shape[1]):
            ax.text(
                col_index,
                row_index,
                str(confusion_frame.iat[row_index, col_index]),
                ha="center",
                va="center",
                color="black",
            )

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_accuracy_plot(per_label_frame: pd.DataFrame, output_path: Path) -> None:
    figure_width = max(8, len(per_label_frame) * 0.9)
    fig, ax = plt.subplots(figsize=(figure_width, 5), constrained_layout=True)
    ax.bar(per_label_frame["actual_label"], per_label_frame["accuracy"], color="#1f6f8b")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy by Label")
    ax.set_xlabel("Label")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=45)

    for index, value in enumerate(per_label_frame["accuracy"]):
        ax.text(index, min(1.0, float(value) + 0.03), f"{float(value):.2f}", ha="center", va="bottom")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_test_report(
    records: Sequence[dict[str, Any]],
    label_names: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Test report requires at least one prediction record")

    if "correct" not in frame.columns:
        frame["correct"] = (frame["actual_label"] == frame["predicted_label"]).astype(int)
    else:
        frame["correct"] = frame["correct"].astype(int)

    predictions_path = output_dir / "predictions.csv"
    frame.to_csv(predictions_path, index=False)

    actual_labels = _unique_preserve_order([str(label) for label in label_names])
    accuracy_labels = list(actual_labels)
    predicted_values = [str(value) for value in frame["predicted_label"].tolist()]
    if UNKNOWN_LABEL in predicted_values and UNKNOWN_LABEL not in actual_labels:
        actual_labels = [*actual_labels, UNKNOWN_LABEL]

    confusion = confusion_matrix(
        frame["actual_label"].astype(str),
        frame["predicted_label"].astype(str),
        labels=actual_labels,
    )
    confusion_frame = pd.DataFrame(confusion, index=actual_labels, columns=actual_labels)
    confusion_path = output_dir / "confusion_matrix.csv"
    confusion_frame.to_csv(confusion_path)
    _save_confusion_plot(confusion_frame, output_dir / "confusion_matrix.png")

    per_label_frame = (
        frame.groupby("actual_label", dropna=False)["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "count"})
        .reindex(accuracy_labels)
        .reset_index()
        .rename(columns={"index": "actual_label"})
    )
    per_label_frame["accuracy"] = per_label_frame["accuracy"].fillna(0.0)
    per_label_frame["count"] = per_label_frame["count"].fillna(0).astype(int)
    per_label_path = output_dir / "accuracy_by_label.csv"
    per_label_frame.to_csv(per_label_path, index=False)
    _save_accuracy_plot(per_label_frame, output_dir / "accuracy_by_label.png")

    summary = {
        "num_samples": int(len(frame)),
        "num_correct": int(frame["correct"].sum()),
        "overall_accuracy": float(frame["correct"].mean()),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                frame["actual_label"].astype(str),
                frame["predicted_label"].astype(str),
            )
        ),
        "mean_label_accuracy": float(per_label_frame["accuracy"].mean()) if len(per_label_frame) else 0.0,
        "label_names": actual_labels,
        "predictions_csv": str(predictions_path),
        "confusion_matrix_csv": str(confusion_path),
        "confusion_matrix_png": str(output_dir / "confusion_matrix.png"),
        "accuracy_by_label_csv": str(per_label_path),
        "accuracy_by_label_png": str(output_dir / "accuracy_by_label.png"),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary
