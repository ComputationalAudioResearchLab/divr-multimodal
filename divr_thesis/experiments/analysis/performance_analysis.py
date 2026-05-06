from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from experiments.analysis.confusion_analysis import analyze_confusion_matrix
from experiments.analysis.label_accuracy_analysis import (
    analyze_accuracy_by_label,
)


def analyze_predictions_csv(
    csv_path: Path,
    label_names: Sequence[str],
    analysis_dir: Path,
) -> dict[str, float | int]:
    frame = pd.read_csv(csv_path)
    frame["correct"] = frame["correct"].astype(int)

    overall_accuracy = float(frame["correct"].mean()) if len(frame) else 0.0
    overall_balanced_accuracy = (
        float(
            balanced_accuracy_score(
                frame["label"],
                frame["prediction"],
            )
        )
        if len(frame)
        else 0.0
    )
    overall_macro_f1 = (
        float(
            f1_score(
                frame["label"],
                frame["prediction"],
                labels=label_names,
                average="macro",
                zero_division=0,
            )
        )
        if len(frame)
        else 0.0
    )

    analyze_confusion_matrix(
        frame=frame,
        label_names=label_names,
        analysis_dir=analysis_dir,
    )
    analyze_accuracy_by_label(frame=frame, analysis_dir=analysis_dir)

    return {
        "overall_accuracy": overall_accuracy,
        "overall_balanced_accuracy": overall_balanced_accuracy,
        "overall_macro_f1": overall_macro_f1,
        "num_samples": int(len(frame)),
    }
