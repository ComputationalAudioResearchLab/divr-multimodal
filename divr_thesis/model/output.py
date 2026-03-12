from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from data_loader import InputTensors, TextTensors
from model.fusion import (
    ConcatenationFusion,
    CrossAttentionFusion,
    FiLMFusion,
    GatedFusion,
)
from model.savable_module import SavableModule


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        pad_index: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_inputs: TextTensors) -> torch.Tensor:
        token_ids, token_lens = text_inputs
        embedded = self.embedding(token_ids)
        mask = (
            torch.arange(token_ids.size(1), device=token_ids.device)
            < token_lens.unsqueeze(1)
        ).unsqueeze(-1)
        embedded = embedded * mask
        pooled = embedded.sum(dim=1) / token_lens.clamp_min(1).unsqueeze(1)
        return self.dropout(pooled)


class ClassificationHead(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        hidden_size = 1024
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
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


class TextClassifier(SavableModule):
    def __init__(
        self,
        vocab_size: int,
        text_embedding_dim: int,
        num_classes: int,
        checkpoint_path: Path,
    ) -> None:
        super().__init__(checkpoint_path)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
        )
        self.head = ClassificationHead(
            input_size=text_embedding_dim,
            num_classes=num_classes,
        )

    def forward(self, text_inputs: TextTensors) -> torch.Tensor:
        text_embedding = self.text_encoder(text_inputs)
        return self.head(text_embedding)


class AudioTextClassifier(SavableModule):
    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        text_embedding_dim: int,
        num_classes: int,
        checkpoint_path: Path,
        fusion_type: str = "concatenation",
    ) -> None:
        super().__init__(checkpoint_path)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
        )
        if fusion_type == "concatenation":
            self.fusion = ConcatenationFusion(input_size, text_embedding_dim)
        elif fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(input_size, text_embedding_dim)
        elif fusion_type == "gated":
            self.fusion = GatedFusion(input_size, text_embedding_dim)
        elif fusion_type == "film":
            self.fusion = FiLMFusion(input_size, text_embedding_dim)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
        self.head = ClassificationHead(
            input_size=self.fusion.output_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        audio_inputs: InputTensors,
        text_inputs: TextTensors,
    ) -> torch.Tensor:
        audio_features, audio_lens = audio_inputs
        text_embedding = self.text_encoder(text_inputs)
        fused_features = self.fusion(
            audio_features,
            text_embedding,
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
