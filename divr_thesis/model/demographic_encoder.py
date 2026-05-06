"""Demographic information encoder."""

import torch
from torch import nn


class DemographicEncoder(nn.Module):
    """
    Encodes demographic information into embedding vectors.

    Args:
        age_embedding_dim: Dimension for age embedding
        gender_embedding_dim: Dimension for gender embedding
        smoking_embedding_dim: Dimension for smoking embedding
        drinking_embedding_dim: Dimension for drinking embedding
        max_age: Maximum age value for age bucketing
        gender_vocab: List of gender categories
            (default: ["male", "female", "unknown"])
    """

    def __init__(
        self,
        age_embedding_dim: int = 256,
        gender_embedding_dim: int = 256,
        smoking_embedding_dim: int = 256,
        drinking_embedding_dim: int = 256,
        max_age: int = 100,
        gender_vocab: list[str] | None = None,
    ):
        super().__init__()
        self.age_embedding_dim = age_embedding_dim
        self.gender_embedding_dim = gender_embedding_dim
        self.smoking_embedding_dim = smoking_embedding_dim
        self.drinking_embedding_dim = drinking_embedding_dim
        self.max_age = max_age

        if gender_vocab is None:
            gender_vocab = ["male", "female", "unknown"]
        self.gender_vocab = gender_vocab

        # Age embedding: normalize age to [0, 1] and pass through linear layer
        self.age_linear = nn.Linear(1, age_embedding_dim)

        # Gender embedding: learnable embeddings for each gender
        self.gender_embedding = nn.Embedding(
            len(gender_vocab),
            gender_embedding_dim,
        )

        # Lifestyle embeddings use strict string vocab from FEMH exports.
        self.smoking_vocab = [
            "never",
            "past",
            "active",
            "e-cigarette",
            "unknown",
        ]
        self.smoking_embedding = nn.Embedding(
            len(self.smoking_vocab),
            smoking_embedding_dim,
        )

        self.drinking_vocab = ["never", "past", "active", "unknown"]
        self.drinking_embedding = nn.Embedding(
            len(self.drinking_vocab),
            drinking_embedding_dim,
        )

        # Combined demographic dimension
        self.demographic_dim = (
            age_embedding_dim
            + gender_embedding_dim
            + smoking_embedding_dim
            + drinking_embedding_dim
        )

    def encode_age(self, age: torch.Tensor) -> torch.Tensor:
        """
        Encode age tensor to embedding.

        Args:
            age: Tensor of shape [B] with age values, use -1 for unknown

        Returns:
            Tensor of shape [B, age_embedding_dim]
        """
        if age.numel() == 0:
            return torch.zeros(
                0,
                self.age_embedding_dim,
                device=self.age_linear.weight.device,
            )

        # Keep missing ages (encoded as -1) at zero contribution.
        known_mask = (age >= 0).float().unsqueeze(-1)
        age = age.float().clamp(min=0).unsqueeze(-1) / self.max_age
        age = torch.clamp(age, 0, 1)

        # Pass through linear layer
        age_embedding = self.age_linear(age) * known_mask
        return age_embedding

    def encode_gender(self, gender: torch.Tensor) -> torch.Tensor:
        """
        Encode gender strings to embedding.

        Args:
            gender: Tensor of gender category ids

        Returns:
            Tensor of shape [B, gender_embedding_dim]
        """
        gender_indices = gender.to(
            self.gender_embedding.weight.device,
            dtype=torch.long,
        ).clamp(min=0, max=len(self.gender_vocab) - 1)
        gender_embedding = self.gender_embedding(gender_indices)
        return gender_embedding

    def encode_categorical_ids(
        self,
        values: torch.Tensor,
        embedding_layer: nn.Embedding,
    ) -> torch.Tensor:
        indices = values.to(
            embedding_layer.weight.device,
            dtype=torch.long,
        ).clamp(min=0, max=embedding_layer.num_embeddings - 1)
        return embedding_layer(indices)

    def forward(
        self,
        age: torch.Tensor,
        gender: torch.Tensor,
        smoking: torch.Tensor,
        drinking: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode demographic information.

        Args:
            age: Tensor of shape [B] with age values, use -1 for unknown
            gender: Tensor of shape [B] with gender ids
            smoking: Tensor of shape [B] with smoking ids
            drinking: Tensor of shape [B] with drinking ids

        Returns:
            Tensor of shape [B, demographic_dim]
        """
        age_emb = self.encode_age(age)
        gender_emb = self.encode_gender(gender)
        smoking_emb = self.encode_categorical_ids(
            smoking,
            self.smoking_embedding,
        )
        drinking_emb = self.encode_categorical_ids(
            drinking,
            self.drinking_embedding,
        )

        # Concatenate all available demographic embeddings
        demographic_embedding = torch.cat(
            [age_emb, gender_emb, smoking_emb, drinking_emb],
            dim=-1,
        )
        return demographic_embedding
