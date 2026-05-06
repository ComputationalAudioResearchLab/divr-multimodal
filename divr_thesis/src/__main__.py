from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DIVR multimodal experiments"
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=PROJECT_ROOT / "tasks" / "femh",
        help=(
            "Path to a task folder containing train.yml, val.yml, "
            "and test.yml"
        ),
    )
    parser.add_argument(
        "--feature-model",
        type=str,
        default="wavlm_base",
        help=(
            "Audio pretrained model name. Supports S3PRL names "
            "or HuggingFace model IDs."
        ),
    )
    parser.add_argument(
        "--combine-mode",
        type=str,
        default="concatenation",
        choices=[
            "audio",
            "concatenation",
            "cross_attention",
            "gated",
            "film",
        ],
        help=(
            "Use audio only or a specific audio-text fusion method. "
            "Audio is always required."
        ),
    )
    parser.add_argument(
        "--classification-head-attention",
        type=str,
        default="none",
        choices=[
            "none",
            "cbam",
            "multi_head_attention",
        ],
        help=(
            "Optional attention block inside ClassificationHead. "
            "Use none for the baseline."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text-fields",
        nargs="*",
        default=["all"],
        help=(
            "Selected key=value fields from texts payloads, for example: "
            "age gender smoking. Use all to keep the entire payload. "
            "Ignored when --combine-mode=audio."
        ),
    )
    parser.add_argument(
        "--text-equals",
        nargs="*",
        default=None,
        help=(
            "Optional payload filters, for example: gender=female "
            "dataset=femh femh.smoking=yes. Ignored when "
            "--combine-mode=audio"
        ),
    )
    parser.add_argument(
        "--demographic-embedding-dim",
        type=int,
        default=256,
        help=(
            "Per-field embedding dimension for age, gender, smoking, "
            "and drinking. Total demographic dimension is 4x this value."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, or cuda",
    )
    parser.add_argument(
        "--disable-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    parser.add_argument(
        "--enable-shap",
        action="store_true",
        help="Enable SHAP analysis during testing (disabled by default)",
    )
    parser.add_argument(
        "--enable-supcon",
        action="store_true",
        help=(
            "Enable supervised contrastive learning (SupCon) during "
            "training"
        ),
    )
    parser.add_argument(
        "--supcon-weight",
        type=float,
        default=0.5,
        help="Weight applied to the supervised contrastive loss",
    )
    parser.add_argument(
        "--supcon-temperature",
        type=float,
        default=0.05,
        help="Temperature used by the supervised contrastive loss",
    )
    parser.add_argument(
        "--supcon-projection-dim",
        type=int,
        default=512,
        help="Projection dimension for the supervised contrastive head",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available task folders under tasks/ and exit",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate a saved checkpoint on one or more task folders",
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help=(
            "Path to a trained run directory that contains config.json and "
            "checkpoints"
        ),
    )
    parser.add_argument(
        "--eval-task-dirs",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "One or more task folders to evaluate on. If omitted, the source "
            "task folder from the run config is used."
        ),
    )
    parser.add_argument(
        "--test-splits",
        nargs="*",
        default=None,
        help=(
            "Split names to evaluate. Use all to combine train, val, and "
            "test into one evaluation set."
        ),
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best.pt",
        help="Checkpoint file name to load during evaluation",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def list_task_dirs() -> list[str]:
    tasks_root = PROJECT_ROOT / "tasks"
    if not tasks_root.exists():
        return []
    return sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in tasks_root.iterdir()
        if path.is_dir()
    )


def main() -> None:
    from experiments import (
        RunConfig,
        build_run_dir,
        run_evaluation,
        run_experiment,
    )

    args = parse_args()
    if args.list_tasks:
        print(json.dumps(list_task_dirs(), indent=2))
        return

    if args.evaluate:
        if args.source_run_dir is None:
            raise ValueError("--source-run-dir is required when using --evaluate")
        source_run_dir = args.source_run_dir
        if not source_run_dir.is_absolute():
            source_run_dir = (PROJECT_ROOT / source_run_dir).resolve()
        if not source_run_dir.exists():
            raise FileNotFoundError(
                f"Source run directory does not exist: {source_run_dir}"
            )
        evaluation_task_dirs = args.eval_task_dirs or []
        result = run_evaluation(
            project_root=PROJECT_ROOT,
            source_run_dir=source_run_dir,
            evaluation_task_dirs=evaluation_task_dirs,
            test_split_names=args.test_splits,
            device=torch.device(resolve_device(args.device)),
            checkpoint_name=args.checkpoint_name,
            enable_shap=args.enable_shap,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    task_dir = args.task_dir
    if not task_dir.is_absolute():
        task_dir = (PROJECT_ROOT / task_dir).resolve()
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory does not exist: {task_dir}")

    feature_model = args.feature_model

    text_fields = args.text_fields
    text_equals = args.text_equals
    if args.combine_mode == "audio":
        text_fields = None
        text_equals = None

    device = resolve_device(args.device)
    run_dir = build_run_dir(
        project_root=PROJECT_ROOT,
        task_name=task_dir.name,
        feature_model=feature_model,
        combine_mode=args.combine_mode,
        classification_head_attention=args.classification_head_attention,
        contrastive_enabled=args.enable_supcon,
        contrastive_projection_dim=(
            args.supcon_projection_dim if args.enable_supcon else None
        ),
    )
    config = RunConfig(
        task_dir=task_dir,
        feature_model=feature_model,
        combine_mode=args.combine_mode,
        classification_head_attention=args.classification_head_attention,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        sample_rate=args.sample_rate,
        random_seed=args.seed,
        text_fields=text_fields,
        text_equals=text_equals,
        demographic_embedding_dim=args.demographic_embedding_dim,
        num_workers=args.num_workers,
        tboard_enabled=not args.disable_tensorboard,
        shap_enabled=args.enable_shap,
        device=torch.device(device),
        run_dir=run_dir,
        contrastive_enabled=args.enable_supcon,
        contrastive_weight=(
            args.supcon_weight if args.enable_supcon else 0.0
        ),
        contrastive_temperature=args.supcon_temperature,
        contrastive_projection_dim=(
            args.supcon_projection_dim if args.enable_supcon else None
        ),
    )
    result = run_experiment(config=config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
