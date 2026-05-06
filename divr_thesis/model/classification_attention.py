from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


AttentionType = Literal["none", "cbam", "multi_head_attention"]


def _ensure_sequence(inputs: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if inputs.dim() == 2:
        return inputs.unsqueeze(1), True
    if inputs.dim() != 3:
        raise ValueError(
            "Classification head attention expects a 2D or 3D tensor"
        )
    return inputs, False


def _build_sequence_mask(
    sequence_lengths: torch.Tensor,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    sequence_lengths = sequence_lengths.to(device=device, dtype=torch.long)
    return (
        torch.arange(max_len, device=device).unsqueeze(0)
        < sequence_lengths.unsqueeze(1)
    )


class CBAM1DAttention(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        hidden_dim = max(feature_dim // reduction_ratio, 1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, feature_dim, bias=False),
        )
        padding = kernel_size // 2
        self.spatial_conv = nn.Conv1d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs, squeeze_sequence = _ensure_sequence(inputs)
        _, max_len, _ = inputs.shape

        sequence_mask = None
        if sequence_lengths is not None:
            sequence_mask = _build_sequence_mask(
                sequence_lengths=sequence_lengths,
                max_len=max_len,
                device=inputs.device,
            ).unsqueeze(-1)
            valid_lengths = sequence_lengths.to(
                device=inputs.device,
                dtype=inputs.dtype,
            ).clamp_min(1).unsqueeze(1)
            masked_inputs = inputs.masked_fill(~sequence_mask, 0.0)
            average_pool = masked_inputs.sum(dim=1) / valid_lengths
            min_value = torch.finfo(inputs.dtype).min
            max_pool = inputs.masked_fill(
                ~sequence_mask,
                min_value,
            ).amax(dim=1)
        else:
            average_pool = inputs.mean(dim=1)
            max_pool = inputs.amax(dim=1)

        channel_attention = torch.sigmoid(
            self.channel_mlp(average_pool) + self.channel_mlp(max_pool)
        )
        channel_refined = inputs * channel_attention.unsqueeze(1)
        if sequence_mask is not None:
            channel_refined = channel_refined.masked_fill(~sequence_mask, 0.0)

        temporal_average = channel_refined.mean(dim=-1, keepdim=True)
        temporal_max = channel_refined.amax(dim=-1, keepdim=True)
        spatial_input = torch.cat(
            [temporal_average, temporal_max],
            dim=-1,
        ).transpose(1, 2)
        spatial_attention = torch.sigmoid(
            self.spatial_conv(spatial_input)
        ).transpose(1, 2)
        if sequence_mask is not None:
            spatial_attention = spatial_attention.masked_fill(
                ~sequence_mask,
                0.0,
            )

        refined = channel_refined * spatial_attention
        if squeeze_sequence:
            refined = refined.squeeze(1)
        return refined


class MultiHeadSequenceAttention(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        requested_num_heads = max(1, num_heads)
        effective_num_heads = math.gcd(feature_dim, requested_num_heads) or 1
        self.num_heads = effective_num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=effective_num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs, squeeze_sequence = _ensure_sequence(inputs)

        key_padding_mask = None
        if sequence_lengths is not None:
            key_padding_mask = _build_sequence_mask(
                sequence_lengths=sequence_lengths,
                max_len=inputs.size(1),
                device=inputs.device,
            )

        attended, _ = self.attention(
            query=inputs,
            key=inputs,
            value=inputs,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        refined = self.norm(inputs + self.dropout(attended))
        if key_padding_mask is not None:
            refined = refined.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        if squeeze_sequence:
            refined = refined.squeeze(1)
        return refined


def build_classification_attention(
    attention_type: AttentionType,
    feature_dim: int,
    num_heads: int = 4,
) -> nn.Module | None:
    if attention_type == "none":
        return None
    if attention_type == "cbam":
        return CBAM1DAttention(feature_dim=feature_dim)
    if attention_type == "multi_head_attention":
        return MultiHeadSequenceAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
        )
    raise ValueError(
        f"Unsupported classification head attention: {attention_type}"
    )
