from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class SavableModule(nn.Module):
    def __init__(self, checkpoint_path: Path):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)

    def save(
        self,
        checkpoint_name: str | int,
        extra: dict | None = None,
    ) -> Path:
        checkpoint_path = self._weight_file_name(checkpoint_name)
        payload = {
            "model_state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def load(
        self,
        checkpoint_name: str | int = "best.pt",
        map_location: str | torch.device | None = None,
    ) -> dict:
        checkpoint = torch.load(
            self._weight_file_name(checkpoint_name),
            map_location=map_location,
        )
        state_dict = checkpoint
        extra: dict = {}
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            extra = checkpoint.get("extra", {})
        self.load_state_dict(state_dict)
        return extra

    def _weight_file_name(self, checkpoint_name: str | int) -> Path:
        if isinstance(checkpoint_name, int):
            file_name = f"epoch_{checkpoint_name:04d}.pt"
        else:
            file_name = checkpoint_name
        return self.checkpoint_path / file_name

    def init_orthogonal_weights(self):
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(init)

    def to(self, device: torch.device) -> SavableModule:
        super().to(device)
        self.device = device
        return self
