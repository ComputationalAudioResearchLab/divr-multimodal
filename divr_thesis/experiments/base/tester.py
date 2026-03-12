from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm

from experiments.base.hparams import HParams


class Tester:
    def __init__(self, hparams: HParams) -> None:
        self.hparams = hparams
        self.data_loader = hparams.data_loader
        self.model = hparams.model.to(hparams.device)
        self.feature = hparams.feature
        self.label_names = self.data_loader.label_names
        hparams.results_dir.mkdir(parents=True, exist_ok=True)
        hparams.analysis_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def run(
        self,
        checkpoint_name: str = "best.pt",
    ) -> dict[str, str | float | int]:
        checkpoint_metadata = self.model.load(
            checkpoint_name,
            map_location=self.hparams.device,
        )
        self.model.eval()
        rows: list[dict[str, object]] = []
        for batch in tqdm(
            self.data_loader.test(),
            desc="Testing",
            leave=False,
        ):
            labels = batch.labels.to(self.hparams.device)
            logits = self._forward_batch(batch)
            predictions = logits.argmax(dim=1).cpu().tolist()
            actuals = labels.cpu().tolist()
            for index, prediction in enumerate(predictions):
                actual = actuals[index]
                row: dict[str, object] = {
                    "sample_id": batch.sample_ids[index],
                    "label": self.label_names[actual],
                    "prediction": self.label_names[prediction],
                    "correct": int(actual == prediction),
                    "selected_text": batch.selected_texts[index],
                    "audio_paths": "|".join(batch.audio_paths[index]),
                }
                for key, values in batch.metadata.items():
                    row[key] = values[index]
                rows.append(row)

        predictions_path = self.hparams.results_dir / "predictions.csv"
        frame = pd.DataFrame(rows)
        frame.to_csv(predictions_path, index=False)
        summary = self._analyze_from_csv(predictions_path)
        summary["checkpoint"] = checkpoint_name
        summary["checkpoint_epoch"] = checkpoint_metadata.get("epoch")
        summary["checkpoint_eval_accuracy"] = checkpoint_metadata.get(
            "eval_accuracy"
        )
        summary["checkpoint_eval_macro_f1"] = checkpoint_metadata.get(
            "eval_macro_f1"
        )
        summary_path = self.hparams.results_dir / "test_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return {
            **summary,
            "predictions_csv": str(predictions_path),
            "summary_json": str(summary_path),
            "analysis_dir": str(self.hparams.analysis_dir),
        }

    def _forward_batch(self, batch) -> torch.Tensor:
        audio_inputs = batch.audio_inputs
        text_inputs = batch.text_inputs
        if audio_inputs is not None:
            audio_inputs = (
                audio_inputs[0].to(self.hparams.device),
                audio_inputs[1].to(self.hparams.device),
            )
            if self.feature is not None:
                audio_inputs = self.feature(audio_inputs)
        if text_inputs is not None:
            text_inputs = (
                text_inputs[0].to(self.hparams.device),
                text_inputs[1].to(self.hparams.device),
            )

        if audio_inputs is not None and text_inputs is not None:
            return self.model(audio_inputs, text_inputs)
        if audio_inputs is not None:
            return self.model(audio_inputs)
        if text_inputs is not None:
            return self.model(text_inputs)
        raise ValueError("Batch did not contain audio inputs or text inputs")

    def _analyze_from_csv(
        self,
        csv_path: Path,
    ) -> dict[str, float | str | int]:
        frame = pd.read_csv(csv_path)
        frame["correct"] = frame["correct"].astype(int)
        overall_accuracy = (
            float(frame["correct"].mean()) if len(frame) else 0.0
        )
        overall_macro_f1 = (
            float(
                f1_score(
                    frame["label"],
                    frame["prediction"],
                    labels=self.label_names,
                    average="macro",
                    zero_division=0,
                )
            )
            if len(frame)
            else 0.0
        )
        confusion = confusion_matrix(
            frame["label"],
            frame["prediction"],
            labels=self.label_names,
        )
        confusion_frame = pd.DataFrame(
            confusion,
            index=self.label_names,
            columns=self.label_names,
        )
        confusion_frame.to_csv(
            self.hparams.analysis_dir / "confusion_matrix.csv"
        )
        self._save_confusion(confusion_frame=confusion_frame)

        per_label = (
            frame.groupby("label", dropna=False)["correct"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "accuracy", "count": "count"})
        )
        per_label.to_csv(
            self.hparams.analysis_dir / "accuracy_by_label.csv",
            index=False,
        )
        self._save_accuracy_bar(
            frame=per_label,
            x_key="label",
            output_path=self.hparams.analysis_dir / "accuracy_by_label.png",
            title="Accuracy by Label",
            x_label="Label",
        )

        summary: dict[str, float | str] = {
            "overall_accuracy": overall_accuracy,
            "overall_macro_f1": overall_macro_f1,
            "num_samples": int(len(frame)),
        }

        if "age" in frame.columns:
            numeric_ages = pd.to_numeric(frame["age"], errors="coerce")
            age_frame = frame.loc[numeric_ages.notna()].copy()
            if len(age_frame) > 0:
                age_frame["age"] = numeric_ages[
                    numeric_ages.notna()
                ].astype(int)
                bucket_size = max(1, self.hparams.age_bucket_size)
                age_frame["age_bucket_start"] = (
                    age_frame["age"] // bucket_size
                ) * bucket_size
                age_frame["age_bucket"] = age_frame["age_bucket_start"].map(
                    lambda value: f"{value}-{value + bucket_size - 1}"
                )
                age_accuracy = (
                    age_frame.groupby(
                        "age_bucket_start", dropna=False
                    )["correct"]
                    .agg(["mean", "count"])
                    .reset_index()
                    .rename(columns={"mean": "accuracy", "count": "count"})
                    .sort_values("age_bucket_start")
                )
                age_accuracy["age_bucket"] = age_accuracy[
                    "age_bucket_start"
                ].map(lambda value: f"{value}-{value + bucket_size - 1}")
                age_accuracy = age_accuracy[
                    ["age_bucket", "accuracy", "count"]
                ]
                age_accuracy.to_csv(
                    self.hparams.analysis_dir / "accuracy_by_age_bucket.csv",
                    index=False,
                )
                self._save_accuracy_bar(
                    frame=age_accuracy,
                    x_key="age_bucket",
                    output_path=(
                        self.hparams.analysis_dir
                        / "accuracy_by_age_bucket.png"
                    ),
                    title="Accuracy by Age Bucket",
                    x_label="Age Bucket",
                )
                summary["age_bucket_accuracy_csv"] = str(
                    self.hparams.analysis_dir / "accuracy_by_age_bucket.csv"
                )

        return summary

    def _save_confusion(self, confusion_frame: pd.DataFrame) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        image = ax.imshow(confusion_frame.values, cmap="magma", aspect="auto")
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
                    color="white",
                )
        fig.savefig(self.hparams.analysis_dir / "confusion_matrix.png")
        plt.close(fig)

    def _save_accuracy_bar(
        self,
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
