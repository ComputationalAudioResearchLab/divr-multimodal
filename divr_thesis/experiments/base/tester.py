from __future__ import annotations

import json

import pandas as pd
import torch
from tqdm import tqdm

from experiments.analysis import (
    analyze_predictions_csv,
    analyze_shap_contributions,
)
from experiments.base.hparams import HParams
from model.output import reduce_sequence_outputs


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
        enable_shap: bool = False,
    ) -> dict[str, str | float | int]:
        checkpoint_metadata = self.model.load(
            checkpoint_name,
            map_location=self.hparams.device,
        )
        self.model.eval()
        rows: list[dict[str, object]] = []
        shap_batches: list[dict[str, object]] = []
        for batch in tqdm(
            self.data_loader.test(),
            desc="Testing",
            leave=False,
        ):
            labels = batch.labels.to(self.hparams.device)
            audio_inputs, demographic_inputs = self._prepare_batch_inputs(
                batch
            )
            logits = self._forward_from_inputs(
                audio_inputs,
                demographic_inputs,
            )
            logits = self._sample_logits(batch, logits)
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

            if enable_shap and audio_inputs is not None:
                shap_batches.append(
                    {
                        "audio_features": audio_inputs[0].detach().cpu(),
                        "audio_lens": audio_inputs[1].detach().cpu(),
                        "ages": (
                            None
                            if demographic_inputs is None
                            else demographic_inputs[0].detach().cpu()
                        ),
                        "gender_ids": (
                            None
                            if demographic_inputs is None
                            else demographic_inputs[1].detach().cpu()
                        ),
                        "smoking_ids": (
                            None
                            if demographic_inputs is None
                            else demographic_inputs[2].detach().cpu()
                        ),
                        "drinking_ids": (
                            None
                            if demographic_inputs is None
                            else demographic_inputs[3].detach().cpu()
                        ),
                        "labels": actuals,
                    }
                )

        predictions_path = self.hparams.results_dir / "predictions.csv"
        frame = pd.DataFrame(rows)
        frame.to_csv(predictions_path, index=False)

        summary = analyze_predictions_csv(
            csv_path=predictions_path,
            label_names=self.label_names,
            analysis_dir=self.hparams.analysis_dir,
        )
        if enable_shap:
            summary.update(
                analyze_shap_contributions(
                    batch_records=shap_batches,
                    model=self.model,
                    label_names=self.label_names,
                    device=self.hparams.device,
                    analysis_dir=self.hparams.analysis_dir,
                )
            )
        else:
            summary.update(
                {
                    "shap_status": "skipped_disabled",
                    "shap_hint": (
                        "Run with --enable-shap to generate SHAP outputs."
                    ),
                }
            )

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

    def _prepare_batch_inputs(self, batch):
        audio_inputs = batch.audio_inputs
        demographic_inputs = batch.demographic_inputs
        if audio_inputs is not None:
            audio_inputs = (
                audio_inputs[0].to(self.hparams.device),
                audio_inputs[1].to(self.hparams.device),
            )
            if self.feature is not None:
                audio_inputs = self.feature(audio_inputs)
        if demographic_inputs is not None:
            demographic_inputs = tuple(
                value.to(self.hparams.device)
                for value in demographic_inputs
            )

        return audio_inputs, demographic_inputs

    def _forward_from_inputs(
        self,
        audio_inputs,
        demographic_inputs,
    ) -> torch.Tensor:

        if audio_inputs is not None and demographic_inputs is not None:
            return self.model(audio_inputs, demographic_inputs)
        if audio_inputs is not None:
            return self.model(audio_inputs)
        raise ValueError(
            "Pure-text mode is disabled; audio inputs are required"
        )

    def _sample_logits(self, batch, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2:
            return logits
        if logits.dim() != 3:
            raise ValueError(
                f"Unsupported logits shape: {tuple(logits.shape)}"
            )
        if batch.audio_inputs is None:
            raise ValueError("Sequence logits require audio lengths")
        audio_lens = batch.audio_inputs[1].to(logits.device)
        return reduce_sequence_outputs(logits, audio_lens)
