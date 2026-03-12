"""
Demographic Information Encoder
Encodes age and gender information as embeddings
"""

import torch
from torch import nn
from typing import Optional, Tuple


class DemographicEncoder(nn.Module):
    """
    Encodes age and gender information into embedding vectors.

    Args:
        age_embedding_dim: Dimension for age embedding
        gender_embedding_dim: Dimension for gender embedding
        max_age: Maximum age value for age bucketing
        gender_vocab: List of gender categories (default: ["male", "female", "unknown"])
    """

    def __init__(
        self,
        age_embedding_dim: int = 64,
        gender_embedding_dim: int = 64,
        max_age: int = 100,
        gender_vocab: Optional[list] = None,
    ):
        super().__init__()
        self.age_embedding_dim = age_embedding_dim
        self.gender_embedding_dim = gender_embedding_dim
        self.max_age = max_age

        if gender_vocab is None:
            gender_vocab = ["male", "female", "unknown"]
        self.gender_vocab = gender_vocab
        self.gender_to_idx = {g: i for i, g in enumerate(gender_vocab)}

        # Age embedding: normalize age to [0, 1] and pass through linear layer
        self.age_linear = nn.Linear(1, age_embedding_dim)

        # Gender embedding: learnable embeddings for each gender
        self.gender_embedding = nn.Embedding(len(gender_vocab), gender_embedding_dim)

        # Combined demographic dimension
        self.demographic_dim = age_embedding_dim + gender_embedding_dim

    def encode_age(self, age: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Encode age tensor to embedding.

        Args:
            age: Tensor of shape [B] with age values or None values

        Returns:
            Tensor of shape [B, age_embedding_dim]
        """
        if age is None or (isinstance(age, torch.Tensor) and age.numel() == 0):
            # If age is not available, use zero embedding
            return torch.zeros(
                1, self.age_embedding_dim, device=self.age_linear.weight.device
            )

        # Normalize age to [0, 1]
        age = age.float().unsqueeze(-1) / self.max_age
        age = torch.clamp(age, 0, 1)

        # Pass through linear layer
        age_embedding = self.age_linear(age)
        return age_embedding

    def encode_gender(self, gender: list) -> torch.Tensor:
        """
        Encode gender strings to embedding.

        Args:
            gender: List of gender strings (e.g., ["male", "female", "male"])

        Returns:
            Tensor of shape [B, gender_embedding_dim]
        """
        gender_indices = torch.tensor(
            [self.gender_to_idx.get(g, self.gender_to_idx["unknown"]) for g in gender],
            device=self.gender_embedding.weight.device,
            dtype=torch.long,
        )
        gender_embedding = self.gender_embedding(gender_indices)
        return gender_embedding

    def forward(self, age: Optional[torch.Tensor], gender: list) -> torch.Tensor:
        """
        Encode demographic information.

        Args:
            age: Tensor of shape [B] with age values, or None
            gender: List of gender strings with length B

        Returns:
            Tensor of shape [B, demographic_dim]
        """
        age_emb = self.encode_age(age)
        gender_emb = self.encode_gender(gender)

        # Concatenate age and gender embeddings
        demographic_embedding = torch.cat([age_emb, gender_emb], dim=-1)
        return demographic_embedding
