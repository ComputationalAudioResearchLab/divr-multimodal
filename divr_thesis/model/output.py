from __future__ import annotations

from pathlib import Path

import torch
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


def mask_sequence_outputs(
    per_frame_logits: torch.Tensor,
    input_lens: torch.Tensor,
) -> torch.Tensor:
    if per_frame_logits.dim() != 3:
        return per_frame_logits

    input_lens = input_lens.to(
        device=per_frame_logits.device,
        dtype=torch.long,
    )
    batch_size, max_len, _ = per_frame_logits.shape
    mask = (
        torch.arange(max_len, device=per_frame_logits.device)
        .expand(batch_size, max_len)
        < input_lens.unsqueeze(1)
    )
    return per_frame_logits * mask.unsqueeze(-1)


def reduce_sequence_outputs(
    per_frame_logits: torch.Tensor,
    input_lens: torch.Tensor,
) -> torch.Tensor:
    if per_frame_logits.dim() != 3:
        return per_frame_logits

    masked_logits = mask_sequence_outputs(per_frame_logits, input_lens)
    return masked_logits.sum(dim=1) / input_lens.to(
        device=per_frame_logits.device,
        dtype=per_frame_logits.dtype,
    ).clamp_min(1).unsqueeze(1)


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

    def forward(
        self,
        inputs: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.attention is not None:
            inputs = self.attention(inputs, sequence_lengths)
        outputs = self.layers(inputs)
        if sequence_lengths is not None:
            outputs = mask_sequence_outputs(outputs, sequence_lengths)
        return outputs


class AudioClassifier(SavableModule):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        checkpoint_path: Path,
        head_attention_type: str = "none",
        head_attention_num_heads: int = 4,
    ):
        super().__init__(checkpoint_path)
        self.head = ClassificationHead(
            input_size=input_size,
            num_classes=num_classes,
            attention_type=head_attention_type,
            attention_num_heads=head_attention_num_heads,
        )

    def forward(self, audio_inputs: InputTensors) -> torch.Tensor:
        audio_features, audio_lens = audio_inputs
        return self.head(audio_features, audio_lens)


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
        return self.head(fused_features, audio_lens)
