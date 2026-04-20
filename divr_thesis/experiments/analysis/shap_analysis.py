from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model.output import reduce_sequence_outputs


class ShapContributionAnalyzer:
    def __init__(
        self,
        model: torch.nn.Module,
        label_names: Sequence[str],
        device: torch.device,
        analysis_dir: Path,
    ) -> None:
        self.model = model
        self.label_names = list(label_names)
        self.device = device
        self.analysis_dir = analysis_dir

    def analyze(
        self,
        batch_records: list[dict[str, object]],
    ) -> dict[str, str]:
        if not batch_records:
            return {"shap_status": "skipped_no_audio_features"}

        all_labels: list[int] = []
        has_text = False
        for record in batch_records:
            batch_labels = record.get("labels")
            if isinstance(batch_labels, list):
                all_labels.extend(int(value) for value in batch_labels)
            if (
                record.get("ages") is not None
                and record.get("gender_ids") is not None
                and record.get("smoking_ids") is not None
                and record.get("drinking_ids") is not None
                and hasattr(self.model, "demographic_encoder")
            ):
                has_text = True

        if len(set(all_labels)) < 2:
            return {"shap_status": "skipped_single_class"}

        try:
            import shap  # type: ignore
        except ModuleNotFoundError:
            return {
                "shap_status": "skipped_missing_dependency",
                "shap_hint": "Install shap to enable SHAP analysis.",
            }

        class_stats: dict[str, dict[str, float | int]] = {
            class_name: {
                "audio_abs_contribution": 0.0,
                "text_abs_contribution": 0.0,
                "age_abs_contribution": 0.0,
                "gender_abs_contribution": 0.0,
                "smoking_abs_contribution": 0.0,
                "drinking_abs_contribution": 0.0,
                "audio_signed_contribution": 0.0,
                "text_signed_contribution": 0.0,
                "age_signed_contribution": 0.0,
                "gender_signed_contribution": 0.0,
                "smoking_signed_contribution": 0.0,
                "drinking_signed_contribution": 0.0,
                "audio_abs_per_dim": 0.0,
                "num_samples": 0,
            }
            for class_name in self.label_names
        }

        any_results = False
        debug_logged = False

        for record in tqdm(
            batch_records,
            desc="SHAP",
            leave=False,
            total=len(batch_records),
            unit="batch",
        ):
            audio_features_tensor = record["audio_features"]
            audio_lens_tensor = record["audio_lens"]
            labels = np.asarray(record["labels"], dtype=np.int64)
            ages_tensor = record.get("ages")
            gender_ids_tensor = record.get("gender_ids")
            smoking_ids_tensor = record.get("smoking_ids")
            drinking_ids_tensor = record.get("drinking_ids")

            if not isinstance(audio_features_tensor, torch.Tensor):
                continue
            if not isinstance(audio_lens_tensor, torch.Tensor):
                continue

            audio_features = audio_features_tensor.to(self.device)
            audio_lens = audio_lens_tensor.to(self.device)
            batch_size, max_len, audio_dim = audio_features.shape
            if batch_size < 2:
                continue

            flat_audio = audio_features.reshape(batch_size, -1)
            audio_flat_dim = flat_audio.size(1)
            background_size = min(64, batch_size)
            if background_size < 2:
                continue

            if (
                isinstance(ages_tensor, torch.Tensor)
                and isinstance(gender_ids_tensor, torch.Tensor)
                and isinstance(smoking_ids_tensor, torch.Tensor)
                and isinstance(drinking_ids_tensor, torch.Tensor)
                and hasattr(self.model, "demographic_encoder")
            ):
                demographic_inputs = (
                    ages_tensor.to(self.device),
                    gender_ids_tensor.to(self.device),
                    smoking_ids_tensor.to(self.device),
                    drinking_ids_tensor.to(self.device),
                )
                text_embedding = self.model.demographic_encoder(
                    *demographic_inputs
                ).detach()
                text_dim = int(text_embedding.size(1))

                age_dim = int(
                    self.model.demographic_encoder.age_embedding_dim
                )
                gender_dim = int(
                    self.model.demographic_encoder.gender_embedding_dim
                )
                smoking_dim = int(
                    self.model.demographic_encoder.smoking_embedding_dim
                )
                drinking_dim = int(
                    self.model.demographic_encoder.drinking_embedding_dim
                )

                class _BatchMultimodalWrapper(torch.nn.Module):
                    def __init__(
                        self,
                        model: torch.nn.Module,
                        audio_seq_len: int,
                        audio_dim: int,
                        batch_audio_lens: torch.Tensor,
                    ) -> None:
                        super().__init__()
                        self.model = model
                        self.audio_seq_len = audio_seq_len
                        self.audio_dim = audio_dim
                        self.batch_audio_lens = batch_audio_lens

                    def _expand_lens(self, batch_size: int) -> torch.Tensor:
                        base_size = int(self.batch_audio_lens.size(0))
                        if base_size == batch_size:
                            return self.batch_audio_lens
                        if batch_size % base_size == 0:
                            repeats = batch_size // base_size
                            return self.batch_audio_lens.repeat(repeats)
                        raise ValueError(
                            "Cannot expand audio lengths to match "
                            "SHAP batch size"
                        )

                    def forward(
                        self,
                        flat_audio_inputs: torch.Tensor,
                        text_embeddings: torch.Tensor,
                    ) -> torch.Tensor:
                        audio_inputs = flat_audio_inputs.view(
                            -1,
                            self.audio_seq_len,
                            self.audio_dim,
                        )
                        audio_lens = self._expand_lens(audio_inputs.size(0))
                        fused_features = self.model.fusion(
                            audio_inputs,
                            text_embeddings,
                            audio_lens,
                        )
                        logits = self.model.head(fused_features, audio_lens)
                        return reduce_sequence_outputs(logits, audio_lens)

                wrapped_model = _BatchMultimodalWrapper(
                    self.model,
                    audio_seq_len=max_len,
                    audio_dim=audio_dim,
                    batch_audio_lens=audio_lens,
                ).to(self.device)
                wrapped_model.eval()
                background = [
                    flat_audio[:background_size],
                    text_embedding[:background_size],
                ]
                inputs = [flat_audio, text_embedding]
                feature_dim = audio_flat_dim + text_dim
            else:
                text_dim = 0
                age_dim = 0
                gender_dim = 0
                smoking_dim = 0
                drinking_dim = 0

                class _BatchAudioWrapper(torch.nn.Module):
                    def __init__(
                        self,
                        model: torch.nn.Module,
                        audio_seq_len: int,
                        audio_dim: int,
                        batch_audio_lens: torch.Tensor,
                    ) -> None:
                        super().__init__()
                        self.model = model
                        self.audio_seq_len = audio_seq_len
                        self.audio_dim = audio_dim
                        self.batch_audio_lens = batch_audio_lens

                    def _expand_lens(self, batch_size: int) -> torch.Tensor:
                        base_size = int(self.batch_audio_lens.size(0))
                        if base_size == batch_size:
                            return self.batch_audio_lens
                        if batch_size % base_size == 0:
                            repeats = batch_size // base_size
                            return self.batch_audio_lens.repeat(repeats)
                        raise ValueError(
                            "Cannot expand audio lengths to match "
                            "SHAP batch size"
                        )

                    def forward(
                        self,
                        flat_audio_inputs: torch.Tensor,
                    ) -> torch.Tensor:
                        audio_inputs = flat_audio_inputs.view(
                            -1,
                            self.audio_seq_len,
                            self.audio_dim,
                        )
                        audio_lens = self._expand_lens(audio_inputs.size(0))
                        logits = self.model((audio_inputs, audio_lens))
                        return reduce_sequence_outputs(logits, audio_lens)

                wrapped_model = _BatchAudioWrapper(
                    self.model,
                    audio_seq_len=max_len,
                    audio_dim=audio_dim,
                    batch_audio_lens=audio_lens,
                ).to(self.device)
                wrapped_model.eval()
                background = flat_audio[:background_size]
                inputs = flat_audio
                feature_dim = audio_flat_dim

            with torch.enable_grad():
                explainer = shap.DeepExplainer(wrapped_model, background)
                shap_values_raw = explainer.shap_values(
                    inputs,
                    check_additivity=False,
                )

            if not debug_logged:
                print(
                    f"SHAP raw output type={type(shap_values_raw).__name__}",
                    flush=True,
                )
                if isinstance(shap_values_raw, (list, tuple)):
                    print(
                        f"SHAP raw outer len={len(shap_values_raw)}",
                        flush=True,
                    )
                    if len(shap_values_raw) > 0:
                        first_item = shap_values_raw[0]
                        print(
                            f"SHAP raw first item type="
                            f"{type(first_item).__name__}",
                            flush=True,
                        )
                        if isinstance(first_item, np.ndarray):
                            print(
                                f"SHAP raw first item shape="
                                f"{first_item.shape}",
                                flush=True,
                            )
                            if len(shap_values_raw) > 1:
                                second_item = shap_values_raw[1]
                                if isinstance(second_item, np.ndarray):
                                    print(
                                        f"SHAP raw second item shape="
                                        f"{second_item.shape}",
                                        flush=True,
                                    )
                        if isinstance(first_item, (list, tuple)):
                            print(
                                f"SHAP raw first item inner len="
                                f"{len(first_item)}",
                                flush=True,
                            )
                            for inner_index, inner_item in enumerate(
                                first_item
                            ):
                                if isinstance(inner_item, np.ndarray):
                                    print(
                                        f"SHAP raw inner[{inner_index}] "
                                        f"shape={inner_item.shape}",
                                        flush=True,
                                    )
                debug_logged = True

            class_indices = list(range(len(self.label_names)))
            class_shap_values = self._resolve_class_shap_values(
                shap_values_raw=shap_values_raw,
                class_indices=class_indices,
                feature_dim=feature_dim,
                audio_feature_dim=audio_flat_dim,
                text_feature_dim=text_dim,
            )
            if not class_shap_values:
                continue

            any_results = True
            for class_index, class_name in enumerate(self.label_names):
                class_values = class_shap_values.get(class_index)
                if class_values is None:
                    class_values = np.zeros(
                        (batch_size, feature_dim),
                        dtype=np.float32,
                    )
                if (
                    class_values.ndim != 2
                    or class_values.shape[1] != feature_dim
                ):
                    return {"shap_status": "skipped_unexpected_shape"}

                class_mask = labels == class_index
                class_count = int(class_mask.sum())
                if class_count == 0:
                    continue

                class_subset = class_values[class_mask]
                audio_abs_contribution = float(
                    np.mean(
                        np.sum(
                            np.abs(class_subset[:, :audio_flat_dim]),
                            axis=1,
                        )
                    )
                )
                audio_signed_contribution = float(
                    np.mean(np.sum(class_subset[:, :audio_flat_dim], axis=1))
                )

                if text_dim > 0:
                    demo_subset = class_subset[:, audio_flat_dim:]
                    age_start = 0
                    gender_start = age_start + age_dim
                    smoking_start = gender_start + gender_dim
                    drinking_start = smoking_start + smoking_dim

                    age_slice = demo_subset[:, age_start:gender_start]
                    gender_slice = demo_subset[:, gender_start:smoking_start]
                    smoking_slice = demo_subset[
                        :, smoking_start:drinking_start
                    ]
                    drinking_slice = demo_subset[
                        :, drinking_start:drinking_start + drinking_dim
                    ]

                    age_abs_contribution = float(
                        np.mean(np.sum(np.abs(age_slice), axis=1))
                    )
                    gender_abs_contribution = float(
                        np.mean(np.sum(np.abs(gender_slice), axis=1))
                    )
                    smoking_abs_contribution = float(
                        np.mean(np.sum(np.abs(smoking_slice), axis=1))
                    )
                    drinking_abs_contribution = float(
                        np.mean(np.sum(np.abs(drinking_slice), axis=1))
                    )

                    age_signed_contribution = float(
                        np.mean(np.sum(age_slice, axis=1))
                    )
                    gender_signed_contribution = float(
                        np.mean(np.sum(gender_slice, axis=1))
                    )
                    smoking_signed_contribution = float(
                        np.mean(np.sum(smoking_slice, axis=1))
                    )
                    drinking_signed_contribution = float(
                        np.mean(np.sum(drinking_slice, axis=1))
                    )

                    text_abs_contribution = (
                        age_abs_contribution
                        + gender_abs_contribution
                        + smoking_abs_contribution
                        + drinking_abs_contribution
                    )
                    text_signed_contribution = (
                        age_signed_contribution
                        + gender_signed_contribution
                        + smoking_signed_contribution
                        + drinking_signed_contribution
                    )
                else:
                    text_abs_contribution = 0.0
                    text_signed_contribution = 0.0
                    age_abs_contribution = 0.0
                    gender_abs_contribution = 0.0
                    smoking_abs_contribution = 0.0
                    drinking_abs_contribution = 0.0
                    age_signed_contribution = 0.0
                    gender_signed_contribution = 0.0
                    smoking_signed_contribution = 0.0
                    drinking_signed_contribution = 0.0

                audio_abs_per_dim = audio_abs_contribution / max(
                    1,
                    audio_flat_dim,
                )
                stat = class_stats[class_name]
                stat["audio_abs_contribution"] = float(
                    stat["audio_abs_contribution"]
                ) + audio_abs_contribution * class_count
                stat["text_abs_contribution"] = float(
                    stat["text_abs_contribution"]
                ) + text_abs_contribution * class_count
                stat["age_abs_contribution"] = float(
                    stat["age_abs_contribution"]
                ) + age_abs_contribution * class_count
                stat["gender_abs_contribution"] = float(
                    stat["gender_abs_contribution"]
                ) + gender_abs_contribution * class_count
                stat["smoking_abs_contribution"] = float(
                    stat["smoking_abs_contribution"]
                ) + smoking_abs_contribution * class_count
                stat["drinking_abs_contribution"] = float(
                    stat["drinking_abs_contribution"]
                ) + drinking_abs_contribution * class_count
                stat["audio_signed_contribution"] = float(
                    stat["audio_signed_contribution"]
                ) + audio_signed_contribution * class_count
                stat["text_signed_contribution"] = float(
                    stat["text_signed_contribution"]
                ) + text_signed_contribution * class_count
                stat["age_signed_contribution"] = float(
                    stat["age_signed_contribution"]
                ) + age_signed_contribution * class_count
                stat["gender_signed_contribution"] = float(
                    stat["gender_signed_contribution"]
                ) + gender_signed_contribution * class_count
                stat["smoking_signed_contribution"] = float(
                    stat["smoking_signed_contribution"]
                ) + smoking_signed_contribution * class_count
                stat["drinking_signed_contribution"] = float(
                    stat["drinking_signed_contribution"]
                ) + drinking_signed_contribution * class_count
                stat["audio_abs_per_dim"] = float(
                    stat["audio_abs_per_dim"]
                ) + audio_abs_per_dim * class_count
                stat["num_samples"] = int(stat["num_samples"]) + class_count

        if not any_results:
            return {"shap_status": "skipped_empty_results"}

        rows: list[dict[str, object]] = []
        for class_name, stat in class_stats.items():
            num_samples = int(stat["num_samples"])
            if num_samples == 0:
                rows.append(
                    {
                        "label": class_name,
                        "audio_abs_contribution": 0.0,
                        "text_abs_contribution": 0.0,
                        "age_abs_contribution": 0.0,
                        "gender_abs_contribution": 0.0,
                        "smoking_abs_contribution": 0.0,
                        "drinking_abs_contribution": 0.0,
                        "audio_signed_contribution": 0.0,
                        "text_signed_contribution": 0.0,
                        "age_signed_contribution": 0.0,
                        "gender_signed_contribution": 0.0,
                        "smoking_signed_contribution": 0.0,
                        "drinking_signed_contribution": 0.0,
                        "audio_abs_per_dim": 0.0,
                        "audio_contribution": 0.0,
                        "age_contribution": 0.0,
                        "gender_contribution": 0.0,
                        "audio_ratio": 0.0,
                        "text_ratio": 0.0,
                        "age_ratio": 0.0,
                        "gender_ratio": 0.0,
                        "smoking_ratio": 0.0,
                        "drinking_ratio": 0.0,
                        "num_samples": 0,
                    }
                )
                continue

            audio_abs_contribution = (
                float(stat["audio_abs_contribution"]) / num_samples
            )
            text_abs_contribution = (
                float(stat["text_abs_contribution"]) / num_samples
            )
            age_abs_contribution = (
                float(stat["age_abs_contribution"]) / num_samples
            )
            gender_abs_contribution = (
                float(stat["gender_abs_contribution"]) / num_samples
            )
            smoking_abs_contribution = (
                float(stat["smoking_abs_contribution"]) / num_samples
            )
            drinking_abs_contribution = (
                float(stat["drinking_abs_contribution"]) / num_samples
            )
            audio_signed_contribution = (
                float(stat["audio_signed_contribution"]) / num_samples
            )
            text_signed_contribution = (
                float(stat["text_signed_contribution"]) / num_samples
            )
            age_signed_contribution = (
                float(stat["age_signed_contribution"]) / num_samples
            )
            gender_signed_contribution = (
                float(stat["gender_signed_contribution"]) / num_samples
            )
            smoking_signed_contribution = (
                float(stat["smoking_signed_contribution"]) / num_samples
            )
            drinking_signed_contribution = (
                float(stat["drinking_signed_contribution"]) / num_samples
            )
            audio_abs_per_dim = float(stat["audio_abs_per_dim"]) / num_samples
            total = audio_abs_contribution + text_abs_contribution
            rows.append(
                {
                    "label": class_name,
                    "audio_abs_contribution": audio_abs_contribution,
                    "text_abs_contribution": text_abs_contribution,
                    "age_abs_contribution": age_abs_contribution,
                    "gender_abs_contribution": gender_abs_contribution,
                    "smoking_abs_contribution": smoking_abs_contribution,
                    "drinking_abs_contribution": drinking_abs_contribution,
                    "audio_signed_contribution": audio_signed_contribution,
                    "text_signed_contribution": text_signed_contribution,
                    "age_signed_contribution": age_signed_contribution,
                    "gender_signed_contribution": gender_signed_contribution,
                    "smoking_signed_contribution": smoking_signed_contribution,
                    "drinking_signed_contribution": (
                        drinking_signed_contribution
                    ),
                    "audio_abs_per_dim": audio_abs_per_dim,
                    "audio_contribution": audio_abs_contribution,
                    "age_contribution": age_abs_contribution,
                    "gender_contribution": gender_abs_contribution,
                    "audio_ratio": (
                        audio_abs_contribution / total if total > 0 else 0.0
                    ),
                    "text_ratio": (
                        text_abs_contribution / total if total > 0 else 0.0
                    ),
                    "age_ratio": (
                        age_abs_contribution / total if total > 0 else 0.0
                    ),
                    "gender_ratio": (
                        gender_abs_contribution / total
                        if total > 0
                        else 0.0
                    ),
                    "smoking_ratio": (
                        smoking_abs_contribution / total
                        if total > 0
                        else 0.0
                    ),
                    "drinking_ratio": (
                        drinking_abs_contribution / total
                        if total > 0
                        else 0.0
                    ),
                    "num_samples": num_samples,
                }
            )

        contribution_frame = pd.DataFrame(rows)
        contribution_csv = self.analysis_dir / "shap_contribution_by_class.csv"
        contribution_frame.to_csv(contribution_csv, index=False)
        self._save_shap_abs_audio_text_bar(contribution_frame)
        self._save_shap_abs_demographic_bar(contribution_frame)
        self._save_shap_signed_audio_text_bar(contribution_frame)
        self._save_shap_signed_demographic_bar(contribution_frame)
        return {
            "shap_status": "ok",
            "shap_method": (
                "DeepExplainer_full_model_audio_demographic"
                if has_text
                else "DeepExplainer_full_model_audio_only"
            ),
            "shap_contribution_csv": str(contribution_csv),
            "shap_abs_audio_demographic_plot": str(
                self.analysis_dir / "shap_abs_audio_demographic_by_class.png"
            ),
            "shap_abs_demographic_plot": str(
                self.analysis_dir / "shap_abs_demographic_by_class.png"
            ),
            "shap_signed_audio_demographic_plot": str(
                self.analysis_dir
                / "shap_signed_audio_demographic_by_class.png"
            ),
            "shap_signed_demographic_plot": str(
                self.analysis_dir / "shap_signed_demographic_by_class.png"
            ),
        }

    def _resolve_class_shap_values(
        self,
        shap_values_raw: object,
        class_indices: list[int],
        feature_dim: int,
        audio_feature_dim: int,
        text_feature_dim: int,
    ) -> dict[int, np.ndarray]:
        class_to_values: dict[int, np.ndarray] = {}
        if isinstance(shap_values_raw, list):
            if self._looks_like_per_input_class_shap(
                shap_values_raw,
                len(class_indices),
            ):
                for class_index in class_indices:
                    combined = self._combine_shap_inputs_for_class(
                        shap_values_raw,
                        class_index,
                        len(class_indices),
                    )
                    if combined is not None:
                        class_to_values[class_index] = combined
                return class_to_values

            if shap_values_raw and isinstance(
                shap_values_raw[0],
                (list, tuple),
            ):
                for index, values in enumerate(shap_values_raw):
                    if index >= len(class_indices):
                        break
                    combined = self._combine_shap_inputs(values)
                    if (
                        combined is not None
                        and combined.ndim == 2
                        and combined.shape[1] == feature_dim
                    ):
                        class_to_values[class_indices[index]] = combined
                return class_to_values

            if self._looks_like_per_input_shap(
                shap_values_raw,
                audio_feature_dim,
                text_feature_dim,
            ):
                combined = self._combine_shap_inputs(shap_values_raw)
                if combined is not None:
                    if len(class_indices) == 2:
                        class_to_values[class_indices[1]] = combined
                        class_to_values[class_indices[0]] = -combined
                    else:
                        class_to_values[class_indices[0]] = combined
                    return class_to_values

            for index, values in enumerate(shap_values_raw):
                if index >= len(class_indices):
                    break
                values_array = self._normalize_shap_array(values, feature_dim)
                if values_array is not None:
                    class_to_values[class_indices[index]] = values_array
            return class_to_values

        if not isinstance(shap_values_raw, np.ndarray):
            return class_to_values

        values_array = self._normalize_shap_array(shap_values_raw, feature_dim)
        if values_array is None:
            return class_to_values

        if values_array.ndim == 3:
            if (
                values_array.shape[1] == len(class_indices)
                and values_array.shape[2] == feature_dim
            ):
                for index, class_index in enumerate(class_indices):
                    class_to_values[class_index] = values_array[:, index, :]
                return class_to_values
            if (
                values_array.shape[2] == len(class_indices)
                and values_array.shape[1] == feature_dim
            ):
                for index, class_index in enumerate(class_indices):
                    class_to_values[class_index] = values_array[:, :, index]
                return class_to_values
            return class_to_values

        if len(class_indices) == 1:
            class_to_values[class_indices[0]] = values_array
            return class_to_values

        if len(class_indices) == 2:
            class_to_values[class_indices[1]] = values_array
            class_to_values[class_indices[0]] = -values_array
            return class_to_values

        class_to_values[class_indices[0]] = values_array
        return class_to_values

    def _normalize_shap_array(
        self,
        values: object,
        feature_dim: int,
    ) -> np.ndarray | None:
        values_array = np.asarray(values)
        if values_array.ndim == 0:
            return None
        if values_array.ndim == 1:
            return values_array.reshape(1, -1)
        if values_array.ndim > 2:
            values_array = values_array.reshape(values_array.shape[0], -1)

        if values_array.shape[1] == feature_dim:
            return values_array
        if values_array.shape[0] == feature_dim:
            return values_array.T
        return values_array if values_array.ndim == 2 else None

    def _looks_like_per_input_shap(
        self,
        shap_values_raw: list[object],
        audio_feature_dim: int,
        text_feature_dim: int,
    ) -> bool:
        if len(shap_values_raw) != 2:
            return False

        first_array = np.asarray(shap_values_raw[0])
        second_array = np.asarray(shap_values_raw[1])
        if first_array.ndim != 2 or second_array.ndim != 2:
            return False

        valid_audio = first_array.shape[1] == audio_feature_dim
        valid_text = second_array.shape[1] == text_feature_dim
        return valid_audio and valid_text

    def _looks_like_per_input_class_shap(
        self,
        shap_values_raw: list[object],
        num_classes: int,
    ) -> bool:
        if not shap_values_raw:
            return False

        for component in shap_values_raw:
            component_array = np.asarray(component)
            if component_array.ndim != 3:
                return False
            if component_array.shape[-1] != num_classes and (
                component_array.shape[1] != num_classes
            ):
                return False
        return True

    def _combine_shap_inputs_for_class(
        self,
        shap_inputs: object,
        class_index: int,
        num_classes: int,
    ) -> np.ndarray | None:
        if not isinstance(shap_inputs, (list, tuple)):
            return None

        selected_arrays: list[np.ndarray] = []
        for component in shap_inputs:
            component_array = np.asarray(component)
            if component_array.ndim == 3:
                if component_array.shape[-1] == num_classes:
                    component_array = component_array[:, :, class_index]
                elif component_array.shape[1] == num_classes:
                    component_array = component_array[:, class_index, :]
                else:
                    return None
            elif component_array.ndim == 2:
                component_array = self._normalize_shap_array(
                    component_array,
                    component_array.shape[1],
                )
            else:
                return None

            if component_array is None:
                return None
            if component_array.ndim == 1:
                component_array = component_array.reshape(1, -1)
            selected_arrays.append(component_array)

        if not selected_arrays:
            return None

        batch_sizes = {array.shape[0] for array in selected_arrays}
        if len(batch_sizes) != 1:
            return None

        return np.concatenate(selected_arrays, axis=1)

    def _combine_shap_inputs(
        self,
        shap_inputs: object,
    ) -> np.ndarray | None:
        if not isinstance(shap_inputs, (list, tuple)):
            return None

        flattened_arrays: list[np.ndarray] = []
        for component in shap_inputs:
            component_array = np.asarray(component)
            if component_array.ndim == 0:
                return None
            if component_array.ndim == 1:
                component_array = component_array.reshape(1, -1)
            elif component_array.ndim > 2:
                component_array = component_array.reshape(
                    component_array.shape[0],
                    -1,
                )
            flattened_arrays.append(component_array)

        if not flattened_arrays:
            return None

        batch_sizes = {array.shape[0] for array in flattened_arrays}
        if len(batch_sizes) != 1:
            return None

        return np.concatenate(flattened_arrays, axis=1)

    def _save_shap_abs_audio_text_bar(self, frame: pd.DataFrame) -> None:
        ordered = frame.sort_values("label").reset_index(drop=True)
        if "text_abs_contribution" not in ordered.columns:
            ordered["text_abs_contribution"] = 0.0
        y_axis = np.arange(len(ordered))
        bar_height = 0.36
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        ax.barh(
            y_axis - 0.5 * bar_height,
            ordered["audio_abs_contribution"],
            height=bar_height,
            label="Audio (abs)",
            color="#24577A",
        )
        ax.barh(
            y_axis + 0.5 * bar_height,
            ordered["text_abs_contribution"],
            height=bar_height,
            label="Text (abs)",
            color="#2E7D32",
        )
        ax.set_yticks(y_axis)
        ax.set_yticklabels(ordered["label"])
        ax.set_xlabel("Mean |SHAP value| (class subset)")
        ax.set_title("SHAP Absolute Contribution (Audio + Demographic)")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.legend(loc="lower right")
        fig.savefig(
            self.analysis_dir / "shap_abs_audio_demographic_by_class.png"
        )
        plt.close(fig)

    def _save_shap_abs_demographic_bar(self, frame: pd.DataFrame) -> None:
        ordered = frame.sort_values("label").reset_index(drop=True)
        y_axis = np.arange(len(ordered))
        bar_height = 0.15
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        ax.barh(
            y_axis - 2 * bar_height,
            ordered["audio_abs_contribution"],
            height=bar_height,
            label="Audio (abs)",
            color="#24577A",
        )
        ax.barh(
            y_axis - bar_height,
            ordered["age_abs_contribution"],
            height=bar_height,
            label="Age (abs)",
            color="#4D9F8A",
        )
        ax.barh(
            y_axis,
            ordered["gender_abs_contribution"],
            height=bar_height,
            label="Gender (abs)",
            color="#E67E22",
        )
        ax.barh(
            y_axis + bar_height,
            ordered["smoking_abs_contribution"],
            height=bar_height,
            label="Smoking (abs)",
            color="#8E6C8A",
        )
        ax.barh(
            y_axis + 2 * bar_height,
            ordered["drinking_abs_contribution"],
            height=bar_height,
            label="Drinking (abs)",
            color="#C26A3F",
        )
        ax.set_yticks(y_axis)
        ax.set_yticklabels(ordered["label"])
        ax.set_xlabel("Mean |SHAP value| (class subset)")
        ax.set_title("SHAP Absolute Contribution (Audio + Demographics)")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.legend(loc="lower right")
        fig.savefig(self.analysis_dir / "shap_abs_demographic_by_class.png")
        plt.close(fig)

    def _save_shap_signed_audio_text_bar(
        self,
        frame: pd.DataFrame,
    ) -> None:
        ordered = frame.sort_values("label").reset_index(drop=True)
        if "text_signed_contribution" not in ordered.columns:
            ordered["text_signed_contribution"] = 0.0
        y_axis = np.arange(len(ordered))
        bar_height = 0.36
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        ax.barh(
            y_axis - 0.5 * bar_height,
            ordered["audio_signed_contribution"],
            height=bar_height,
            label="Audio (signed)",
            color="#24577A",
        )
        ax.barh(
            y_axis + 0.5 * bar_height,
            ordered["text_signed_contribution"],
            height=bar_height,
            label="Text (signed)",
            color="#2E7D32",
        )
        ax.axvline(0.0, color="#222222", linewidth=1.0, alpha=0.8)
        ax.set_yticks(y_axis)
        ax.set_yticklabels(ordered["label"])
        ax.set_xlabel("Mean signed SHAP value (class subset)")
        ax.set_title("SHAP Signed Contribution (Audio + Demographic)")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.legend(loc="lower right")
        fig.savefig(
            self.analysis_dir
            / "shap_signed_audio_demographic_by_class.png"
        )
        plt.close(fig)

    def _save_shap_signed_demographic_bar(
        self,
        frame: pd.DataFrame,
    ) -> None:
        ordered = frame.sort_values("label").reset_index(drop=True)
        y_axis = np.arange(len(ordered))
        bar_height = 0.15
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        ax.barh(
            y_axis - 2 * bar_height,
            ordered["audio_signed_contribution"],
            height=bar_height,
            label="Audio (signed)",
            color="#24577A",
        )
        ax.barh(
            y_axis - bar_height,
            ordered["age_signed_contribution"],
            height=bar_height,
            label="Age (signed)",
            color="#4D9F8A",
        )
        ax.barh(
            y_axis,
            ordered["gender_signed_contribution"],
            height=bar_height,
            label="Gender (signed)",
            color="#E67E22",
        )
        ax.barh(
            y_axis + bar_height,
            ordered["smoking_signed_contribution"],
            height=bar_height,
            label="Smoking (signed)",
            color="#8E6C8A",
        )
        ax.barh(
            y_axis + 2 * bar_height,
            ordered["drinking_signed_contribution"],
            height=bar_height,
            label="Drinking (signed)",
            color="#C26A3F",
        )
        ax.axvline(0.0, color="#222222", linewidth=1.0, alpha=0.8)
        ax.set_yticks(y_axis)
        ax.set_yticklabels(ordered["label"])
        ax.set_xlabel("Mean signed SHAP value (class subset)")
        ax.set_title("SHAP Signed Contribution (Audio + Demographics)")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.legend(loc="lower right")
        fig.savefig(
            self.analysis_dir / "shap_signed_demographic_by_class.png"
        )
        plt.close(fig)


def analyze_shap_contributions(
    batch_records: list[dict[str, object]],
    model: torch.nn.Module,
    label_names: Sequence[str],
    device: torch.device,
    analysis_dir: Path,
) -> dict[str, str]:
    analyzer = ShapContributionAnalyzer(
        model=model,
        label_names=label_names,
        device=device,
        analysis_dir=analysis_dir,
    )
    return analyzer.analyze(batch_records=batch_records)
