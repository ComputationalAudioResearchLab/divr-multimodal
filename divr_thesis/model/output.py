from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from data_loader import DemographicTensors, InputTensors
from model.classification_attention import build_classification_attention
from model.demographic_encoder import DemographicEncoder
from model.fusion import (
    ConcatenationFusion,
    CrossAttentionFusion,
    FiLMFusion,
    GatedFusion,
)
from model.savable_module import SavableModule


def _mean_pool_sequence(
    sequence: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    max_len = sequence.size(1)
    mask = (
        torch.arange(max_len, device=sequence.device)
        < sequence_lengths.unsqueeze(1)
    ).unsqueeze(-1)
    masked_sequence = sequence * mask
    return masked_sequence.sum(dim=1) / sequence_lengths.clamp_min(1).unsqueeze(1)


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        attention_type: str = "none",
        attention_num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.attention = build_classification_attention(
            attention_type=attention_type,
            feature_dim=input_size,
            num_heads=attention_num_heads,
        )
        self.hidden_size = 1024
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def encode(
        self,
        inputs: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.attention is not None:
            inputs = self.attention(inputs, sequence_lengths)
        return self.feature_extractor(inputs)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(
        self,
        inputs: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.encode(inputs, sequence_lengths)
        return self.classify(features)


class AudioClassifier(SavableModule):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        checkpoint_path: Path,
        head_attention_type: str = "none",
        head_attention_num_heads: int = 4,
        contrastive_projection_dim: int | None = None,
    ):
        super().__init__(checkpoint_path)
        self.head = ClassificationHead(
            input_size=input_size,
            num_classes=num_classes,
            attention_type=head_attention_type,
            attention_num_heads=head_attention_num_heads,
        )
        self.contrastive_projector = self._build_contrastive_projector(
            contrastive_projection_dim
        )

    def _build_contrastive_projector(
        self,
        contrastive_projection_dim: int | None,
    ) -> nn.Module | None:
        if contrastive_projection_dim is None:
            return None
        if contrastive_projection_dim <= 0:
            raise ValueError("contrastive_projection_dim must be positive")
        return nn.Sequential(
            nn.Linear(self.head.hidden_size, self.head.hidden_size),
            nn.ReLU(inplace=False),
            nn.Linear(self.head.hidden_size, contrastive_projection_dim),
        )

    def _contrastive_embedding(self, features: torch.Tensor) -> torch.Tensor:
        if self.contrastive_projector is not None:
            features = self.contrastive_projector(features)
        return F.normalize(features, dim=-1)

    def forward(
        self,
        audio_inputs: InputTensors,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        audio_features, audio_lens = audio_inputs
        hidden_features = self.head.encode(audio_features, audio_lens)
        per_frame_labels = self.head.classify(hidden_features)
        per_audio_labels = _mean_pool_sequence(per_frame_labels, audio_lens)

        if return_embeddings:
            pooled_features = _mean_pool_sequence(hidden_features, audio_lens)
            return per_audio_labels, self._contrastive_embedding(
                pooled_features
            )

        return per_audio_labels


class AudioTextClassifier(SavableModule):
    def __init__(
        self,
        input_size: int,
        demographic_embedding_dim: int,
        num_classes: int,
        checkpoint_path: Path,
        fusion_type: str = "concatenation",
        head_attention_type: str = "none",
        head_attention_num_heads: int = 4,
        contrastive_projection_dim: int | None = None,
    ) -> None:
        super().__init__(checkpoint_path)
        self.demographic_encoder = DemographicEncoder(
            age_embedding_dim=demographic_embedding_dim,
            gender_embedding_dim=demographic_embedding_dim,
            smoking_embedding_dim=demographic_embedding_dim,
            drinking_embedding_dim=demographic_embedding_dim,
        )
        demographic_dim = self.demographic_encoder.demographic_dim
        if fusion_type == "concatenation":
            self.fusion = ConcatenationFusion(input_size, demographic_dim)
        elif fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(input_size, demographic_dim)
        elif fusion_type == "gated":
            self.fusion = GatedFusion(input_size, demographic_dim)
        elif fusion_type == "film":
            self.fusion = FiLMFusion(input_size, demographic_dim)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
        self.head = ClassificationHead(
            input_size=self.fusion.output_dim,
            num_classes=num_classes,
            attention_type=head_attention_type,
            attention_num_heads=head_attention_num_heads,
        )
        self.contrastive_projector = self._build_contrastive_projector(
            contrastive_projection_dim
        )

    def _build_contrastive_projector(
        self,
        contrastive_projection_dim: int | None,
    ) -> nn.Module | None:
        if contrastive_projection_dim is None:
            return None
        if contrastive_projection_dim <= 0:
            raise ValueError("contrastive_projection_dim must be positive")
        return nn.Sequential(
            nn.Linear(self.head.hidden_size, self.head.hidden_size),
            nn.ReLU(inplace=False),
            nn.Linear(self.head.hidden_size, contrastive_projection_dim),
        )

    def _contrastive_embedding(self, features: torch.Tensor) -> torch.Tensor:
        if self.contrastive_projector is not None:
            features = self.contrastive_projector(features)
        return F.normalize(features, dim=-1)

    def forward(
        self,
        audio_inputs: InputTensors,
        demographic_inputs: DemographicTensors,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        audio_features, audio_lens = audio_inputs
        age, gender, smoking, drinking = demographic_inputs
        demographic_embedding = self.demographic_encoder(
            age,
            gender,
            smoking,
            drinking,
        )
        fused_features = self.fusion(
            audio_features,
            demographic_embedding,
            audio_lens,
        )
        hidden_features = self.head.encode(fused_features, audio_lens)
        logits = self.head.classify(hidden_features)
        per_audio_labels = _mean_pool_sequence(logits, audio_lens)

        if return_embeddings:
            pooled_features = _mean_pool_sequence(hidden_features, audio_lens)
            return per_audio_labels, self._contrastive_embedding(
                pooled_features
            )

        return per_audio_labels
