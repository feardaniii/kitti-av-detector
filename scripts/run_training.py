"""CLI entrypoint to train the AV detector."""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import ProjectConfig
from training.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AV detector on KITTI.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Path to KITTI root folder containing training/image_2 and training/label_2. "
            "If omitted, uses KITTI_DATASET_ROOT env var or repo datasets/kitti."
        ),
    )
    parser.add_argument("--stage1-epochs", type=int, default=None, help="Override stage 1 epochs.")
    parser.add_argument("--stage2-epochs", type=int, default=None, help="Override stage 2 epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of indexed object samples for faster experiments.",
    )
    parser.add_argument(
        "--skip-fine-tune",
        action="store_true",
        help="Skip stage 2 fine-tuning for faster runs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Data loader worker processes for model.fit.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Square model input size override (example: 160).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast debug preset: stage1=3, stage2=0, max-samples=6000, input-size=160.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config = ProjectConfig(dataset_root_override=args.dataset_root)

    if args.quick:
        config = replace(
            config,
            stage1_epochs=3,
            stage2_epochs=0,
            max_train_samples=6000,
            skip_fine_tune=True,
            input_width=160,
            input_height=160,
        )

    updates = {}
    if args.stage1_epochs is not None:
        updates["stage1_epochs"] = max(0, args.stage1_epochs)
    if args.stage2_epochs is not None:
        updates["stage2_epochs"] = max(0, args.stage2_epochs)
    if args.batch_size is not None:
        updates["batch_size"] = max(1, args.batch_size)
    if args.max_samples is not None:
        updates["max_train_samples"] = max(1, args.max_samples)
    if args.workers is not None:
        updates["train_workers"] = max(1, args.workers)
    if args.input_size is not None:
        updates["input_width"] = max(64, args.input_size)
        updates["input_height"] = max(64, args.input_size)
    if args.skip_fine_tune:
        updates["skip_fine_tune"] = True
    if updates:
        config = replace(config, **updates)

    train_model(config)


if __name__ == "__main__":
    main()
