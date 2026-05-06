from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    config_parent = argparse.ArgumentParser(add_help=False)
    _add_config_argument(config_parent, default=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(
        prog="divr-llm",
        description="Run voice disorder classification experiments with Qwen2Audio QLoRA.",
    )
    _add_config_argument(parser, default=str(PROJECT_ROOT / "Qlora" / "qwen2audio_lora.yml"))

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "check-data",
        parents=[config_parent],
        help="Validate YAML splits and audio paths.",
    )
    subparsers.add_parser(
        "train",
        parents=[config_parent],
        help="Fine-tune Qwen2Audio with QLoRA.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        parents=[config_parent],
        help="Evaluate a trained adapter.",
    )
    evaluate_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Adapter checkpoint path. Defaults to output_dir/final_adapter from the config.",
    )

    subparsers.add_parser(
        "run",
        parents=[config_parent],
        help="Train and then evaluate the final adapter.",
    )
    return parser


def _add_config_argument(parser: argparse.ArgumentParser, default: str | object) -> None:
    parser.add_argument(
        "--config",
        default=default,
        help="Path to a QLoRA experiment YAML config.",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from model import check_data, evaluate, run, train

    if args.command == "check-data":
        summary = check_data(args.config)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "train":
        adapter_dir = train(args.config)
        print(f"Saved final adapter to: {adapter_dir}")
        return 0

    if args.command == "evaluate":
        result_dir = evaluate(args.config, checkpoint=args.checkpoint)
        print(f"Wrote analysis outputs to: {result_dir}")
        return 0

    if args.command == "run":
        result_dir = run(args.config)
        print(f"Wrote analysis outputs to: {result_dir}")
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
