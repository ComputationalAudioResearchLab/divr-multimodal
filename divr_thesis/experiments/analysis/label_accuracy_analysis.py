from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def analyze_accuracy_by_label(
    frame: pd.DataFrame,
    analysis_dir: Path,
) -> Path:
    per_label = (
        frame.groupby("label", dropna=False)["correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "count"})
    )

    csv_path = analysis_dir / "accuracy_by_label.csv"
    per_label.to_csv(csv_path, index=False)
    _save_accuracy_bar(
        frame=per_label,
        x_key="label",
        output_path=analysis_dir / "accuracy_by_label.png",
        title="Accuracy by Label",
        x_label="Label",
    )
    return csv_path


def _save_accuracy_bar(
    frame: pd.DataFrame,
    x_key: str,
    output_path: Path,
    title: str,
    x_label: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    ax.bar(frame[x_key], frame["accuracy"], color="#1f6f8b")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(output_path)
    plt.close(fig)
