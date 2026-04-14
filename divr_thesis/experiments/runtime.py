from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from data_loader import TaskDataModule
from experiments.base.hparams import HParams
from experiments.base.tester import Tester
from experiments.base.trainer import Trainer
from model import (
    AudioEncoder,
    AudioClassifier,
    AudioTextClassifier,
)


@dataclass(slots=True)
class RunConfig:
    task_dir: Path
    feature_model: str | None
    combine_mode: str
    epochs: int
    batch_size: int
    learning_rate: float
    save_every: int
    sample_rate: int
    random_seed: int
    text_fields: Sequence[str] | None
    text_equals: Sequence[str] | None
    text_embedding_dim: int
    age_bucket_size: int
    num_workers: int
    tboard_enabled: bool
    device: torch.device
    run_dir: Path


def run_experiment(config: RunConfig) -> dict[str, object]:
    include_audio = True
    include_text = config.combine_mode != "audio"
    text_fields = config.text_fields if include_text else None
    text_equals = config.text_equals if include_text else None
    data_module = TaskDataModule(
        task_dir=config.task_dir,
        sample_rate=config.sample_rate,
        batch_size=config.batch_size,
        random_seed=config.random_seed,
        include_audio=include_audio,
        include_text=include_text,
        text_fields=text_fields,
        text_equals=text_equals,
        num_workers=config.num_workers,
    )

    if not config.feature_model:
        raise ValueError("This experiment always requires --feature-model")
    feature = AudioEncoder(
        model_name=config.feature_model,
        device=config.device,
    )
    audio_feature_size = infer_feature_size(
        feature,
        data_module,
        config.device,
    )

    checkpoints_dir = config.run_dir / "checkpoints"
    if config.combine_mode == "audio":
        assert audio_feature_size is not None
        model = AudioClassifier(
            input_size=audio_feature_size,
            num_classes=len(data_module.label_names),
            checkpoint_path=checkpoints_dir,
        )
        model_name = f"audio_{config.feature_model}"
    elif config.combine_mode in {
        "concatenation",
        "cross_attention",
        "gated",
        "film",
    }:
        assert audio_feature_size is not None
        model = AudioTextClassifier(
            input_size=audio_feature_size,
            vocab_size=data_module.text_vocab_size,
            text_embedding_dim=config.text_embedding_dim,
            num_classes=len(data_module.label_names),
            checkpoint_path=checkpoints_dir,
            fusion_type=config.combine_mode,
        )
        model_name = f"{config.feature_model}_{config.combine_mode}"
    else:
        raise ValueError(f"Unsupported combine_mode: {config.combine_mode}")

    class_weights = (
        data_module.class_counts.sum()
        / data_module.class_counts.clamp_min(1)
    )
    hparams = HParams(
        criterion=nn.CrossEntropyLoss(weight=class_weights.to(config.device)),
        data_loader=data_module,
        device=config.device,
        feature=feature,
        model=model,
        model_name=model_name,
        task_key=config.task_dir.name,
        num_classes=len(data_module.label_names),
        num_epochs=config.epochs,
        optimizer=torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
        ),
        run_dir=config.run_dir,
        save_every=config.save_every,
        task_dir=config.task_dir,
        sample_rate=config.sample_rate,
        random_seed=config.random_seed,
        tboard_enabled=config.tboard_enabled,
        text_fields=text_fields,
        text_equals=text_equals,
        age_bucket_size=config.age_bucket_size,
        num_workers=config.num_workers,
    )

    write_config_file(config=config)
    trainer = Trainer(hparams=hparams)
    training_summary = trainer.run()
    tester = Tester(hparams=hparams)
    test_summary = tester.run(checkpoint_name="best.pt")
    return {
        "training": training_summary,
        "testing": test_summary,
    }


def build_run_dir(
    project_root: Path,
    task_name: str,
    feature_model: str | None,
    combine_mode: str,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_key = feature_model or "audio"
    return (
        project_root
        / ".cache"
        / "runs"
        / f"{task_name}_{model_key}_{combine_mode}_{timestamp}"
    )


def infer_feature_size(
    feature: AudioEncoder,
    data_module: TaskDataModule,
    device: torch.device,
) -> int:
    batch = next(iter(data_module.train()))
    assert batch.audio_inputs is not None
    audio_inputs = (
        batch.audio_inputs[0].to(device),
        batch.audio_inputs[1].to(device),
    )
    audio_features, _ = feature(audio_inputs)
    return int(audio_features.size(-1))


def write_config_file(config: RunConfig) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.run_dir / "config.json"
    serializable = asdict(config)
    serializable["task_dir"] = str(config.task_dir)
    serializable["device"] = str(config.device)
    serializable["run_dir"] = str(config.run_dir)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
