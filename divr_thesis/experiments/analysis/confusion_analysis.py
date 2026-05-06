from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


def analyze_confusion_matrix(
    frame: pd.DataFrame,
    label_names: Sequence[str],
    analysis_dir: Path,
) -> Path:
    confusion = confusion_matrix(
        frame["label"],
        frame["prediction"],
        labels=label_names,
    )
    confusion_frame = pd.DataFrame(
        confusion,
        index=label_names,
        columns=label_names,
    )

    csv_path = analysis_dir / "confusion_matrix.csv"
    confusion_frame.to_csv(csv_path)
    _save_confusion_plot(confusion_frame, analysis_dir)
    return csv_path


def _save_confusion_plot(
    confusion_frame: pd.DataFrame,
    analysis_dir: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    image = ax.imshow(confusion_frame.values, cmap=plt.cm.Blues, aspect="auto")
    ax.set_xticks(range(len(confusion_frame.columns)))
    ax.set_yticks(range(len(confusion_frame.index)))
    ax.set_xticklabels(confusion_frame.columns, rotation=45, ha="right")
    ax.set_yticklabels(confusion_frame.index)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
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

    fig.savefig(analysis_dir / "confusion_matrix.png")
    plt.close(fig)
