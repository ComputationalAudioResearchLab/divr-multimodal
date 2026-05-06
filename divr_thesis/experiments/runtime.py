from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
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
    classification_head_attention: str
    epochs: int
    batch_size: int
    learning_rate: float
    save_every: int
    sample_rate: int
    random_seed: int
    text_fields: Sequence[str] | None
    text_equals: Sequence[str] | None
    demographic_embedding_dim: int
    num_workers: int
    tboard_enabled: bool
    shap_enabled: bool
    device: torch.device
    run_dir: Path
    contrastive_enabled: bool = False
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.07
    contrastive_projection_dim: int | None = None
    evaluation_mode: bool = False
    source_run_dir: Path | None = None
    source_task_dir: Path | None = None
    evaluation_task_dir: Path | None = None
    test_split_names: Sequence[str] | None = None
    checkpoint_name: str = "best.pt"
    evaluation_shap_enabled: bool = False


def load_run_config(run_dir: Path) -> RunConfig:
    config_path = run_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    return RunConfig(
        task_dir=Path(raw_config["task_dir"]),
        feature_model=raw_config.get("feature_model"),
        combine_mode=raw_config["combine_mode"],
        classification_head_attention=raw_config[
            "classification_head_attention"
        ],
        epochs=int(raw_config["epochs"]),
        batch_size=int(raw_config["batch_size"]),
        learning_rate=float(raw_config["learning_rate"]),
        save_every=int(raw_config["save_every"]),
        sample_rate=int(raw_config["sample_rate"]),
        random_seed=int(raw_config["random_seed"]),
        text_fields=raw_config.get("text_fields"),
        text_equals=raw_config.get("text_equals"),
        demographic_embedding_dim=int(
            raw_config["demographic_embedding_dim"]
        ),
        num_workers=int(raw_config["num_workers"]),
        tboard_enabled=bool(raw_config["tboard_enabled"]),
        shap_enabled=bool(raw_config["shap_enabled"]),
        device=torch.device(str(raw_config["device"])),
        run_dir=Path(raw_config["run_dir"]),
        contrastive_enabled=bool(raw_config.get("contrastive_enabled", False)),
        contrastive_weight=float(raw_config.get("contrastive_weight", 0.0)),
        contrastive_temperature=float(
            raw_config.get("contrastive_temperature", 0.07)
        ),
        contrastive_projection_dim=(
            None
            if raw_config.get("contrastive_projection_dim") is None
            else int(raw_config["contrastive_projection_dim"])
        ),
        evaluation_mode=bool(raw_config.get("evaluation_mode", False)),
        source_run_dir=_load_optional_path(raw_config, "source_run_dir"),
        source_task_dir=_load_optional_path(raw_config, "source_task_dir"),
        evaluation_task_dir=_load_optional_path(
            raw_config, "evaluation_task_dir"
        ),
        test_split_names=_load_optional_sequence(
            raw_config, "test_split_names"
        ),
        checkpoint_name=str(raw_config.get("checkpoint_name", "best.pt")),
        evaluation_shap_enabled=bool(
            raw_config.get("evaluation_shap_enabled", False)
        ),
    )


def run_evaluation(
    *,
    project_root: Path,
    source_run_dir: Path,
    evaluation_task_dirs: Sequence[Path],
    test_split_names: Sequence[str] | None,
    device: torch.device,
    checkpoint_name: str = "best.pt",
    enable_shap: bool = False,
) -> dict[str, object]:
    source_config = load_run_config(source_run_dir)
    source_config = replace(source_config, device=device)
    source_data_module = TaskDataModule(
        task_dir=source_config.task_dir,
        sample_rate=source_config.sample_rate,
        batch_size=source_config.batch_size,
        random_seed=source_config.random_seed,
        include_audio=True,
        include_text=source_config.combine_mode != "audio",
        text_fields=(
            source_config.text_fields
            if source_config.combine_mode != "audio"
            else None
        ),
        text_equals=(
            source_config.text_equals
            if source_config.combine_mode != "audio"
            else None
        ),
        num_workers=source_config.num_workers,
    )
    feature, model, base_hparams = build_model_runtime(
        config=source_config,
        data_module=source_data_module,
        checkpoint_root=source_run_dir,
        run_dir=source_run_dir,
    )

    if not evaluation_task_dirs:
        evaluation_task_dirs = [source_config.task_dir]

    requested_splits = (
        list(test_split_names) if test_split_names is not None else ["test"]
    )
    evaluation_results: dict[str, dict[str, object]] = {}
    for evaluation_task_dir in evaluation_task_dirs:
        if not evaluation_task_dir.is_absolute():
            evaluation_task_dir = (
                project_root / evaluation_task_dir
            ).resolve()
        if not evaluation_task_dir.exists():
            raise FileNotFoundError(
                f"Evaluation task directory does not exist: {evaluation_task_dir}"
            )

        eval_run_dir = build_evaluation_run_dir(
            project_root=project_root,
            evaluation_task_name=evaluation_task_dir.name,
            source_run_name=source_run_dir.name,
        )
        eval_config = replace(
            source_config,
            task_dir=evaluation_task_dir,
            run_dir=eval_run_dir,
            evaluation_mode=True,
            source_run_dir=source_run_dir,
            source_task_dir=source_config.task_dir,
            evaluation_task_dir=evaluation_task_dir,
            test_split_names=requested_splits,
            checkpoint_name=checkpoint_name,
            evaluation_shap_enabled=enable_shap,
        )
        write_config_file(eval_config)

        target_data_module = TaskDataModule(
            task_dir=evaluation_task_dir,
            sample_rate=source_config.sample_rate,
            batch_size=source_config.batch_size,
            random_seed=source_config.random_seed,
            include_audio=True,
            include_text=source_config.combine_mode != "audio",
            text_fields=(
                source_config.text_fields
                if source_config.combine_mode != "audio"
                else None
            ),
            text_equals=(
                source_config.text_equals
                if source_config.combine_mode != "audio"
                else None
            ),
            test_split_names=test_split_names,
            num_workers=source_config.num_workers,
        )
        eval_hparams = replace(
            base_hparams,
            data_loader=target_data_module,
            run_dir=eval_run_dir,
            task_dir=evaluation_task_dir,
            task_key=evaluation_task_dir.name,
        )
        tester = Tester(hparams=eval_hparams)
        summary = tester.run(
            checkpoint_name=checkpoint_name,
            enable_shap=enable_shap,
        )
        summary.update(
            {
                "source_run_dir": str(source_run_dir),
                "source_task_dir": str(source_config.task_dir),
                "evaluation_task_dir": str(evaluation_task_dir),
                "evaluation_splits": requested_splits,
            }
        )
        summary_path = Path(str(summary["summary_json"]))
        summary_payload = {
            key: value
            for key, value in summary.items()
            if key not in {"predictions_csv", "summary_json", "analysis_dir"}
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2)
        evaluation_results[str(evaluation_task_dir)] = summary

    if len(evaluation_results) == 1:
        return next(iter(evaluation_results.values()))
    return {"evaluations": evaluation_results}


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
    feature, model, hparams = build_model_runtime(
        config=config,
        data_module=data_module,
        checkpoint_root=config.run_dir,
        run_dir=config.run_dir,
    )

    write_config_file(config=config)
    trainer = Trainer(hparams=hparams)
    training_summary = trainer.run()
    tester = Tester(hparams=hparams)
    test_summary = tester.run(
        checkpoint_name="best.pt",
        enable_shap=config.shap_enabled,
    )
    return {
        "training": training_summary,
        "testing": test_summary,
    }


def build_model_runtime(
    config: RunConfig,
    data_module: TaskDataModule,
    checkpoint_root: Path,
    run_dir: Path,
) -> tuple[AudioEncoder, nn.Module, HParams]:
    if not config.feature_model:
        raise ValueError("This experiment always requires --feature-model")

    include_text = config.combine_mode != "audio"
    text_fields = config.text_fields if include_text else None
    text_equals = config.text_equals if include_text else None
    contrastive_projection_dim = (
        config.contrastive_projection_dim
        if config.contrastive_projection_dim is not None
        else 128
    )
    model_contrastive_projection_dim = (
        contrastive_projection_dim if config.contrastive_enabled else None
    )

    feature = AudioEncoder(
        model_name=config.feature_model,
        device=config.device,
    )
    audio_feature_size = infer_feature_size(
        feature,
        data_module,
        config.device,
    )
    checkpoints_dir = checkpoint_root / "checkpoints"
    if config.combine_mode == "audio":
        assert audio_feature_size is not None
        model = AudioClassifier(
            input_size=audio_feature_size,
            num_classes=len(data_module.label_names),
            checkpoint_path=checkpoints_dir,
            head_attention_type=config.classification_head_attention,
            contrastive_projection_dim=model_contrastive_projection_dim,
        )
        model_name = (
            f"audio_{config.feature_model}_"
            f"{config.classification_head_attention}"
        )
    elif config.combine_mode in {
        "concatenation",
        "cross_attention",
        "gated",
        "film",
    }:
        assert audio_feature_size is not None
        model = AudioTextClassifier(
            input_size=audio_feature_size,
            demographic_embedding_dim=config.demographic_embedding_dim,
            num_classes=len(data_module.label_names),
            checkpoint_path=checkpoints_dir,
            fusion_type=config.combine_mode,
            head_attention_type=config.classification_head_attention,
            contrastive_projection_dim=model_contrastive_projection_dim,
        )
        model_name = (
            f"{config.feature_model}_{config.combine_mode}_"
            f"{config.classification_head_attention}"
        )
    else:
        raise ValueError(f"Unsupported combine_mode: {config.combine_mode}")

    if config.contrastive_enabled:
        model_name = f"{model_name}_supcon{contrastive_projection_dim}"

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
        run_dir=run_dir,
        save_every=config.save_every,
        task_dir=config.task_dir,
        sample_rate=config.sample_rate,
        random_seed=config.random_seed,
        tboard_enabled=config.tboard_enabled,
        text_fields=text_fields,
        text_equals=text_equals,
        num_workers=config.num_workers,
        contrastive_enabled=config.contrastive_enabled,
        contrastive_weight=config.contrastive_weight,
        contrastive_temperature=config.contrastive_temperature,
        contrastive_projection_dim=config.contrastive_projection_dim,
    )
    return feature, model, hparams


def build_run_dir(
    project_root: Path,
    task_name: str,
    feature_model: str | None,
    combine_mode: str,
    classification_head_attention: str,
    contrastive_enabled: bool = False,
    contrastive_projection_dim: int | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_key = feature_model or "audio"
    contrastive_key = ""
    if contrastive_enabled:
        if contrastive_projection_dim is None:
            contrastive_key = "_supcon"
        else:
            contrastive_key = f"_supcon{contrastive_projection_dim}"
    return (
        project_root
        / ".cache"
        / "runs"
        / (
            f"{task_name}_{model_key}_{combine_mode}_"
            f"{classification_head_attention}{contrastive_key}_{timestamp}"
        )
    )


def build_evaluation_run_dir(
    project_root: Path,
    evaluation_task_name: str,
    source_run_name: str,
) -> Path:
    return (
        project_root
        / ".cache"
        / "runs"
        / f"{evaluation_task_name}__on__{source_run_name}"
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
    if config.source_run_dir is not None:
        serializable["source_run_dir"] = str(config.source_run_dir)
    else:
        serializable.pop("source_run_dir", None)
    if config.source_task_dir is not None:
        serializable["source_task_dir"] = str(config.source_task_dir)
    else:
        serializable.pop("source_task_dir", None)
    if config.evaluation_task_dir is not None:
        serializable["evaluation_task_dir"] = str(config.evaluation_task_dir)
    else:
        serializable.pop("evaluation_task_dir", None)
    if config.test_split_names is None:
        serializable.pop("test_split_names", None)
    if not config.evaluation_mode:
        serializable.pop("evaluation_mode", None)
        serializable.pop("checkpoint_name", None)
        serializable.pop("evaluation_shap_enabled", None)
    else:
        serializable["evaluation_mode"] = True
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def _load_optional_path(
    raw_config: dict[str, object],
    key: str,
) -> Path | None:
    value = raw_config.get(key)
    if value is None:
        return None
    return Path(str(value))


def _load_optional_sequence(
    raw_config: dict[str, object],
    key: str,
) -> Sequence[str] | None:
    value = raw_config.get(key)
    if value is None:
        return None
    return [str(item) for item in value]
