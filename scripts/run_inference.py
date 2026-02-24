"""CLI entrypoint for video inference."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import ProjectConfig
from inference.video_inference import run_video_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AV detector on an MP4 video.")
    parser.add_argument("--input", type=Path, required=True, help="Path to input MP4 video")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output annotated MP4 video",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to trained .keras model (default: outputs/models/av_perception_final.keras)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence threshold override (0.0-1.0)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config = ProjectConfig()
    output_path = args.output if args.output else config.default_output_video

    run_video_inference(
        config=config,
        input_video=args.input,
        output_video=output_path,
        model_path=args.model,
        confidence_threshold=args.confidence,
    )


if __name__ == "__main__":
    main()
