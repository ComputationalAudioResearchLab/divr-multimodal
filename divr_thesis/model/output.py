from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from data_loader import DemographicTensors, InputTensors
from model.demographic_encoder import DemographicEncoder
from model.fusion import (
    ConcatenationFusion,
    CrossAttentionFusion,
    FiLMFusion,
    GatedFusion,
)
from model.savable_module import SavableModule


class ClassificationHead(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        hidden_size = 1024
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class AudioClassifier(SavableModule):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        checkpoint_path: Path,
    ):
        super().__init__(checkpoint_path)
        self.head = ClassificationHead(
            input_size=input_size,
            num_classes=num_classes,
        )

    def forward(self, audio_inputs: InputTensors) -> torch.Tensor:
        audio_features, audio_lens = audio_inputs
        pooled = self._mean_pool(audio_features, audio_lens)
        return self.head(pooled)

    def _mean_pool(
        self,
        audio_features: torch.Tensor,
        audio_lens: torch.Tensor,
    ) -> torch.Tensor:
        max_len = audio_features.size(1)
        mask = (
            torch.arange(max_len, device=audio_features.device)
            < audio_lens.unsqueeze(1)
        ).unsqueeze(-1)
        masked = audio_features * mask
        return masked.sum(dim=1) / audio_lens.clamp_min(1).unsqueeze(1)


class AudioTextClassifier(SavableModule):
    def __init__(
        self,
        input_size: int,
        text_embedding_dim: int,
        num_classes: int,
        checkpoint_path: Path,
        fusion_type: str = "concatenation",
    ) -> None:
        super().__init__(checkpoint_path)
        self.demographic_encoder = DemographicEncoder(
            age_embedding_dim=text_embedding_dim,
            gender_embedding_dim=text_embedding_dim,
            smoking_embedding_dim=text_embedding_dim,
            drinking_embedding_dim=text_embedding_dim,
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
        )

    def forward(
        self,
        audio_inputs: InputTensors,
        demographic_inputs: DemographicTensors,
    ) -> torch.Tensor:
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
        logits = self.head(fused_features)
        return self._mean_pool_logits(logits, audio_lens)

    def _mean_pool_logits(
        self,
        logits: torch.Tensor,
        audio_lens: torch.Tensor,
    ) -> torch.Tensor:
        max_len = logits.size(1)
        mask = (
            torch.arange(max_len, device=logits.device)
            < audio_lens.unsqueeze(1)
        ).unsqueeze(-1)
        masked_logits = logits * mask
        return masked_logits.sum(dim=1) / audio_lens.clamp_min(1).unsqueeze(1)
