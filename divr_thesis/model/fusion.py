"""Fusion layers for combining audio frames with context vectors."""

import torch
from torch import nn


class ConcatenationFusion(nn.Module):
    """
    Direct concatenation fusion strategy.

    Concatenates audio features with expanded context features across time.
    """

    def __init__(self, audio_feature_dim: int, demographic_dim: int):
        """
        Args:
            audio_feature_dim: Dimension of audio features [B, T, audio_dim]
            demographic_dim: Dimension of context features [B, demo_dim]
        """
        super().__init__()
        self.audio_feature_dim = audio_feature_dim
        self.demographic_dim = demographic_dim
        self.output_dim = audio_feature_dim + demographic_dim

    def forward(
        self,
        audio_features: torch.Tensor,
        demographic_features: torch.Tensor,
        audio_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse audio and context features via concatenation.

        Args:
            audio_features: [B, T, audio_dim] audio feature frames
            demographic_features: [B, demo_dim] context embeddings
            audio_lens: [B] sequence lengths

        Returns:
            fused_features: [B, T, audio_dim + demo_dim]
        """
        batch_size, max_len, _ = audio_features.shape

        # Expand demographic features across temporal dimension
        # [B, demo_dim] -> [B, T, demo_dim]
        expanded_demo = demographic_features.unsqueeze(1).expand(
            batch_size, max_len, self.demographic_dim
        )

        # Concatenate along feature dimension
        fused_features = torch.cat([audio_features, expanded_demo], dim=-1)

        return fused_features


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion strategy.

    Uses multi-head cross-attention to attend audio frames to context features.
    """

    def __init__(
        self,
        audio_feature_dim: int,
        demographic_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            audio_feature_dim: Dimension of audio features
            demographic_dim: Dimension of context features
            num_heads: Number of attention heads
            dropout: Dropout rate for attention
        """
        super().__init__()
        self.audio_feature_dim = audio_feature_dim
        self.demographic_dim = demographic_dim
        self.num_heads = num_heads
        self.output_dim = audio_feature_dim

        # Project context features to query dimension
        self.demo_to_query = nn.Linear(demographic_dim, audio_feature_dim)

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=audio_feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Residual connection and layer norm
        self.norm = nn.LayerNorm(audio_feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        audio_features: torch.Tensor,
        demographic_features: torch.Tensor,
        audio_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse audio and context features via cross-attention.

        Args:
            audio_features: [B, T, audio_dim] audio feature frames
            demographic_features: [B, demo_dim] context embeddings
            audio_lens: [B] sequence lengths

        Returns:
            fused_features: [B, T, audio_dim]
        """
        batch_size, max_len, audio_dim = audio_features.shape

        # Project context features to query
        # [B, demo_dim] -> [B, 1, audio_dim]
        query = self.demo_to_query(demographic_features).unsqueeze(1)

        # Use audio features as both key and value
        key = audio_features
        value = audio_features

        # Create attention mask based on sequence lengths
        key_padding_mask = self._create_padding_mask(
            audio_lens,
            batch_size,
            max_len,
        )

        # Apply cross-attention
        # query: [B, 1, audio_dim]
        # key, value: [B, T, audio_dim]
        attended_features, _ = self.cross_attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
        )

        # Broadcast attended features across all time steps
        # [B, 1, audio_dim] -> [B, T, audio_dim]
        attended_features = attended_features.expand(
            batch_size,
            max_len,
            audio_dim,
        )

        # Residual connection
        fused_features = self.norm(
            audio_features + self.dropout(attended_features)
        )

        return fused_features

    @staticmethod
    def _create_padding_mask(
        lens: torch.Tensor, batch_size: int, max_len: int
    ) -> torch.Tensor:
        """Create attention padding mask from sequence lengths."""
        mask = torch.arange(max_len, device=lens.device).expand(
            batch_size, max_len
        ) >= lens.unsqueeze(1)
        return mask


class GatedFusion(nn.Module):
    """
    Gated fusion strategy.

    Uses learnable gates to weight the contribution of audio and context
    features.
    """

    def __init__(
        self,
        audio_feature_dim: int,
        demographic_dim: int,
        hidden_dim: int = 256,
    ):
        """
        Args:
            audio_feature_dim: Dimension of audio features
            demographic_dim: Dimension of context features
            hidden_dim: Hidden dimension for gate computation
        """
        super().__init__()
        self.audio_feature_dim = audio_feature_dim
        self.demographic_dim = demographic_dim
        self.output_dim = audio_feature_dim

        # Gate network: computes a scalar gate from context features
        self.gate_network = nn.Sequential(
            nn.Linear(demographic_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Optional: residual projection for context features
        self.demo_projection = nn.Linear(demographic_dim, audio_feature_dim)

    def forward(
        self,
        audio_features: torch.Tensor,
        demographic_features: torch.Tensor,
        audio_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse audio and context features via gating mechanism.

        Args:
            audio_features: [B, T, audio_dim] audio feature frames
            demographic_features: [B, demo_dim] context embeddings
            audio_lens: [B] sequence lengths

        Returns:
            fused_features: [B, T, audio_dim]
        """
        batch_size, max_len, audio_dim = audio_features.shape

        # Compute gate values from context features
        # [B, demo_dim] -> [B, 1]
        gate = self.gate_network(demographic_features)  # [B, 1]

        # Project context features to audio dimension
        demo_projected = self.demo_projection(
            demographic_features
        )  # [B, audio_dim]

        # Expand context features across time dimension
        # [B, audio_dim] -> [B, T, audio_dim]
        demo_expanded = demo_projected.unsqueeze(1).expand(
            batch_size, max_len, audio_dim
        )

        # Apply gated fusion: audio * gate + demo * (1 - gate)
        # This allows the model to learn how much to weight each modality
        gate_expanded = gate.unsqueeze(1)  # [B, 1, 1]
        fused_features = audio_features * gate_expanded + demo_expanded * (
            1 - gate_expanded
        )

        return fused_features


class FiLMFusion(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) fusion strategy.

    Uses context features to generate learnable affine transformation
    parameters (scale and shift) that modulate audio features element-wise.
    """

    def __init__(
        self,
        audio_feature_dim: int,
        demographic_dim: int,
        hidden_dim: int = 256,
    ):
        """
        Args:
            audio_feature_dim: Dimension of audio features
            demographic_dim: Dimension of context features
            hidden_dim: Hidden dimension for parameter generation network
        """
        super().__init__()
        self.audio_feature_dim = audio_feature_dim
        self.demographic_dim = demographic_dim
        self.output_dim = audio_feature_dim

        # Network to generate scale parameters
        self.scale_network = nn.Sequential(
            nn.Linear(demographic_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, audio_feature_dim),
        )

        # Network to generate shift parameters
        self.shift_network = nn.Sequential(
            nn.Linear(demographic_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, audio_feature_dim),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        demographic_features: torch.Tensor,
        audio_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse audio and context features via FiLM modulation.

        Args:
            audio_features: [B, T, audio_dim] audio feature frames
            demographic_features: [B, demo_dim] context embeddings
            audio_lens: [B] sequence lengths

        Returns:
            fused_features: [B, T, audio_dim]
        """
        batch_size, max_len, audio_dim = audio_features.shape

        # Generate scale and shift parameters from context features
        # [B, demo_dim] -> [B, audio_dim]
        scale = self.scale_network(demographic_features)  # [B, audio_dim]
        shift = self.shift_network(demographic_features)  # [B, audio_dim]

        # Expand parameters across temporal dimension
        # [B, audio_dim] -> [B, T, audio_dim]
        scale_expanded = scale.unsqueeze(1).expand(
            batch_size,
            max_len,
            audio_dim,
        )
        shift_expanded = shift.unsqueeze(1).expand(
            batch_size,
            max_len,
            audio_dim,
        )

        # Apply FiLM: y = scale * x + shift
        fused_features = scale_expanded * audio_features + shift_expanded

        return fused_features
