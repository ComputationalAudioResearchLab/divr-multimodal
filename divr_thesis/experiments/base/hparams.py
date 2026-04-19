from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from data_loader import TaskDataModule
from model import AudioEncoder, SavableModule

project_root = Path(__file__).parent.parent.parent.resolve()


@dataclass
class HParams:
    criterion: nn.Module
    data_loader: TaskDataModule
    device: torch.device
    model: SavableModule
    model_name: str
    task_key: str
    num_classes: int
    num_epochs: int
    optimizer: Optimizer
    run_dir: Path
    save_every: int
    task_dir: Path

    feature: AudioEncoder | None = None
    text_fields: Sequence[str] | None = None
    text_equals: Sequence[str] | None = None
    num_workers: int = 0

    cache_path: Path = Path(f"{project_root}/.cache")
    sample_rate: int = 16000
    random_seed: int = 42
    save_enabled: bool = True
    tboard_enabled: bool = True

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def results_dir(self) -> Path:
        return self.run_dir / "results"

    @property
    def analysis_dir(self) -> Path:
        return self.results_dir / "analysis"
