"""CLI entrypoint for YOLO multi-object video inference."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import ProjectConfig
from inference.yolo_video_inference import run_yolo_video_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretrained YOLO inference and annotate all Car/Pedestrian/Cyclist detections."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to input MP4")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: outputs/videos/annotated_output_yolo.mp4)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics model weights name or path (example: yolov8n.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size passed to YOLO",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Inference device ("cpu", "0", etc.)',
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()

    config = ProjectConfig()
    output = args.output if args.output else config.videos_dir / "annotated_output_yolo.mp4"

    run_yolo_video_inference(
        config=config,
        input_video=args.input,
        output_video=output,
        weights_path=args.weights,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        image_size=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
