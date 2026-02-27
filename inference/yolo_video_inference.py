"""Multi-object video inference using pretrained YOLO models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
from tqdm import tqdm

from config.config import ProjectConfig, ensure_output_dirs


LOGGER = logging.getLogger(__name__)

# COCO -> project label mapping
COCO_TO_PROJECT: Dict[str, str] = {
    "car": "Car",
    "person": "Pedestrian",
    "bicycle": "Cyclist",
}


@dataclass
class Detection:
    """Single detection used for post-processing and drawing."""

    coco_name: str
    mapped_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


class YOLOImportError(RuntimeError):
    """Raised when ultralytics is not installed."""


def run_yolo_video_inference(
    config: ProjectConfig,
    input_video: Path,
    output_video: Path,
    weights_path: str = "yolov8n.pt",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.45,
    image_size: int = 640,
    device: str = "cpu",
) -> Path:
    """Run multi-object detection on an MP4 and save annotated output."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise YOLOImportError(
            "ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    ensure_output_dirs(config)

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    LOGGER.info("Loading YOLO model: %s", weights_path)
    model = YOLO(weights_path)

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video writer: {output_video}")

    LOGGER.info(
        "Running YOLO inference. fps=%.2f, resolution=%dx%d, frames=%d, device=%s",
        fps,
        width,
        height,
        total_frames,
        device,
    )

    try:
        progress = tqdm(total=total_frames if total_frames > 0 else None, desc="YOLO inference")
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = model.predict(
                source=frame,
                verbose=False,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=image_size,
                device=device,
            )

            result = results[0]
            names = result.names
            boxes = result.boxes

            detections: List[Detection] = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    coco_name = names.get(cls_id, str(cls_id)).lower()
                    mapped_name = COCO_TO_PROJECT.get(coco_name)
                    if mapped_name is None:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(
                        Detection(
                            coco_name=coco_name,
                            mapped_name=mapped_name,
                            confidence=conf,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                        )
                    )

            _apply_cyclist_relabel(detections)

            for det in detections:
                x1_i, y1_i, x2_i, y2_i = det.bbox
                color = _label_color(det.mapped_name)
                cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                label = f"{det.mapped_name}: {det.confidence:.2f}"
                (tw, th), base = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                )
                y_top = max(0, y1_i - th - base - 6)
                cv2.rectangle(
                    frame,
                    (x1_i, y_top),
                    (x1_i + tw + 8, y_top + th + base + 6),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1_i + 4, y_top + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            progress.update(1)
        progress.close()
    finally:
        capture.release()
        writer.release()

    LOGGER.info("YOLO annotated video saved to: %s", output_video)
    return output_video


def _apply_cyclist_relabel(detections: List[Detection], iou_threshold: float = 0.1) -> None:
    """Relabel person as Cyclist when overlapping with a bicycle box."""
    bicycle_boxes = [d.bbox for d in detections if d.coco_name == "bicycle"]
    if not bicycle_boxes:
        return

    for det in detections:
        if det.coco_name != "person":
            continue
        for bike_box in bicycle_boxes:
            if _iou(det.bbox, bike_box) >= iou_threshold:
                det.mapped_name = "Cyclist"
                break


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def _label_color(label: str) -> tuple[int, int, int]:
    if label == "Car":
        return (0, 200, 255)
    if label == "Pedestrian":
        return (60, 255, 60)
    if label == "Cyclist":
        return (255, 120, 40)
    return (255, 255, 255)
