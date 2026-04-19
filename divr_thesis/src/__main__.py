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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
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
    parser.add_argument("--text-embedding-dim", type=int, default=128)
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
        "--list-tasks",
        action="store_true",
        help="List available task folders under tasks/ and exit",
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
    from experiments import RunConfig, build_run_dir, run_experiment

    args = parse_args()
    if args.list_tasks:
        print(json.dumps(list_task_dirs(), indent=2))
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
    )
    config = RunConfig(
        task_dir=task_dir,
        feature_model=feature_model,
        combine_mode=args.combine_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        sample_rate=args.sample_rate,
        random_seed=args.seed,
        text_fields=text_fields,
        text_equals=text_equals,
        text_embedding_dim=args.text_embedding_dim,
        num_workers=args.num_workers,
        tboard_enabled=not args.disable_tensorboard,
        device=torch.device(device),
        run_dir=run_dir,
    )
    result = run_experiment(config=config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
