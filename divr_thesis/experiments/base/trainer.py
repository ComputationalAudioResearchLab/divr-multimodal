from __future__ import annotations

import csv
import json

import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

from experiments.base.hparams import HParams
from experiments.base.tboard import MockBoard, TBoard


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                "SupervisedContrastiveLoss expects [batch, dim] features"
            )
        batch_size = features.size(0)
        if batch_size < 2:
            return features.new_tensor(0.0)

        normalized_features = F.normalize(features, dim=-1)
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).to(
            device=features.device,
            dtype=features.dtype,
        )
        logits = torch.matmul(normalized_features, normalized_features.T)
        logits = logits / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        logits_mask = torch.ones_like(positive_mask) - torch.eye(
            batch_size,
            device=features.device,
            dtype=features.dtype,
        )
        positive_mask = positive_mask * logits_mask
        positive_counts = positive_mask.sum(dim=1)
        valid_anchors = positive_counts > 0
        if not torch.any(valid_anchors):
            return features.new_tensor(0.0)

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        )
        mean_log_prob_pos = (
            positive_mask[valid_anchors] * log_prob[valid_anchors]
        ).sum(dim=1) / positive_counts[valid_anchors]
        return -mean_log_prob_pos.mean()


class Trainer:
    def __init__(self, hparams: HParams) -> None:
        tensorboard_path = (
            hparams.cache_path
            / "tboard"
            / hparams.model_name
            / hparams.task_key
        )
        self.hparams = hparams
        self.data_loader = hparams.data_loader
        self.model = hparams.model.to(hparams.device)
        self.feature = hparams.feature
        self.criterion = hparams.criterion
        self.num_epochs = hparams.num_epochs
        self.optimizer = hparams.optimizer
        self.contrastive_enabled = (
            hparams.contrastive_enabled and hparams.contrastive_weight > 0.0
        )
        self.contrastive_weight = hparams.contrastive_weight
        self.contrastive_loss = (
            SupervisedContrastiveLoss(
                temperature=hparams.contrastive_temperature,
            )
            if self.contrastive_enabled
            else None
        )
        self.unique_diagnosis = self.data_loader.unique_diagnosis
        self.best_eval_balanced_accuracy = -1.0
        self.best_eval_accuracy = 0.0
        self.best_eval_macro_f1 = 0.0
        self.best_epoch = 0
        self.history: list[dict[str, float | int]] = []
        hparams.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        hparams.results_dir.mkdir(parents=True, exist_ok=True)
        if hparams.tboard_enabled:
            self.tboard = TBoard(tensorboard_path=tensorboard_path)
        else:
            self.tboard = MockBoard()

    def run(self) -> dict[str, int | float | str]:
        for epoch in tqdm(
            range(1, self.num_epochs + 1),
            desc="Epoch",
            position=0,
        ):
            train_loss = self._train_loop()
            (
                eval_loss,
                eval_balanced_accuracy,
                eval_accuracy,
                eval_macro_f1,
            ) = self._eval_loop()
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "eval_balanced_accuracy": eval_balanced_accuracy,
                    "eval_accuracy": eval_accuracy,
                    "eval_macro_f1": eval_macro_f1,
                }
            )
            self.tboard.add_scalars(
                "loss",
                {"train": train_loss, "eval": eval_loss},
                global_step=epoch,
            )
            self.tboard.add_scalar(
                "eval_balanced_accuracy",
                eval_balanced_accuracy,
                global_step=epoch,
            )
            self.tboard.add_scalar(
                "eval_accuracy",
                eval_accuracy,
                global_step=epoch,
            )
            self.tboard.add_scalar(
                "eval_macro_f1",
                eval_macro_f1,
                global_step=epoch,
            )
            self._save(
                epoch=epoch,
                eval_balanced_accuracy=eval_balanced_accuracy,
                eval_accuracy=eval_accuracy,
                eval_macro_f1=eval_macro_f1,
            )
        self._write_history()
        summary = {
            "best_epoch": self.best_epoch,
            "best_eval_balanced_accuracy": self.best_eval_balanced_accuracy,
            "best_eval_accuracy": self.best_eval_accuracy,
            "best_eval_macro_f1": self.best_eval_macro_f1,
            "best_selection_metric": "eval_balanced_accuracy",
            "best_checkpoint": "best.pt",
        }
        with open(
            self.hparams.results_dir / "training_summary.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(summary, handle, indent=2)
        return summary

    def _train_loop(self) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in tqdm(
            self.data_loader.train(),
            desc="Training",
            leave=False,
        ):
            self.optimizer.zero_grad(set_to_none=True)
            labels = batch.labels.to(self.hparams.device)
            if self.contrastive_enabled:
                logits, embeddings = self._forward_batch(
                    batch,
                    return_embeddings=True,
                )
                loss = self.criterion(logits, labels)
                assert self.contrastive_loss is not None
                loss = loss + self.contrastive_weight * self.contrastive_loss(
                    embeddings,
                    labels,
                )
            else:
                logits = self._forward_batch(batch)
                loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        return total_loss / max(1, total_batches)

    @torch.no_grad()
    def _eval_loop(self) -> tuple[float, float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        total_correct = 0
        total_items = 0
        all_labels: list[int] = []
        all_predictions: list[int] = []
        for batch in tqdm(
            self.data_loader.eval(),
            desc="Validating",
            leave=False,
        ):
            labels = batch.labels.to(self.hparams.device)
            logits = self._forward_batch(batch)
            loss = self.criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            total_loss += loss.item()
            total_batches += 1
            total_correct += (predictions == labels).sum().item()
            total_items += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
        accuracy = total_correct / max(1, total_items)
        balanced_accuracy = (
            float(
                balanced_accuracy_score(
                    all_labels,
                    all_predictions,
                )
            )
            if all_labels
            else 0.0
        )
        macro_f1 = (
            float(
                f1_score(
                    all_labels,
                    all_predictions,
                    average="macro",
                    zero_division=0,
                )
            )
            if all_labels
            else 0.0
        )
        return (
            total_loss / max(1, total_batches),
            balanced_accuracy,
            accuracy,
            macro_f1,
        )

    def _forward_batch(
        self,
        batch,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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

        if audio_inputs is not None and demographic_inputs is not None:
            if return_embeddings:
                return self.model(
                    audio_inputs,
                    demographic_inputs,
                    return_embeddings=True,
                )
            return self.model(audio_inputs, demographic_inputs)
        if audio_inputs is not None:
            if return_embeddings:
                return self.model(audio_inputs, return_embeddings=True)
            return self.model(audio_inputs)
        raise ValueError(
            "Pure-text mode is disabled; audio inputs are required"
        )

    def _save(
        self,
        epoch: int,
        eval_balanced_accuracy: float,
        eval_accuracy: float,
        eval_macro_f1: float,
    ) -> None:
        if not self.hparams.save_enabled:
            return
        if epoch % self.hparams.save_every == 0:
            self.model.save(
                epoch,
                extra={
                    "epoch": epoch,
                    "eval_balanced_accuracy": eval_balanced_accuracy,
                    "eval_accuracy": eval_accuracy,
                    "eval_macro_f1": eval_macro_f1,
                },
            )
        if eval_balanced_accuracy >= self.best_eval_balanced_accuracy:
            self.best_eval_balanced_accuracy = eval_balanced_accuracy
            self.best_eval_accuracy = eval_accuracy
            self.best_eval_macro_f1 = eval_macro_f1
            self.best_epoch = epoch
            self.model.save(
                "best.pt",
                extra={
                    "epoch": epoch,
                    "eval_balanced_accuracy": eval_balanced_accuracy,
                    "eval_accuracy": eval_accuracy,
                    "eval_macro_f1": eval_macro_f1,
                },
            )
        self.model.save(
            "last.pt",
            extra={
                "epoch": epoch,
                "eval_balanced_accuracy": eval_balanced_accuracy,
                "eval_accuracy": eval_accuracy,
                "eval_macro_f1": eval_macro_f1,
            },
        )

    def _write_history(self) -> None:
        history_path = self.hparams.results_dir / "history.csv"
        with open(history_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "eval_loss",
                    "eval_balanced_accuracy",
                    "eval_accuracy",
                    "eval_macro_f1",
                ],
            )
            writer.writeheader()
            writer.writerows(self.history)
