"""Frame preprocessing and drawing helpers for inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    """Model detection projected to frame coordinates."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]


def preprocess_frame(frame: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    """Resize and normalize an RGB frame for model inference."""
    input_width, input_height = input_size
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def postprocess_prediction(
    class_probs: np.ndarray,
    bbox_norm: np.ndarray,
    frame_shape: Tuple[int, int, int],
    class_names: Sequence[str],
    confidence_threshold: float,
) -> Optional[DetectionResult]:
    """Convert model outputs into one best detection in original frame coordinates."""
    class_probs = np.asarray(class_probs).reshape(-1)
    bbox_norm = np.asarray(bbox_norm).reshape(4)

    class_id = int(np.argmax(class_probs))
    confidence = float(class_probs[class_id])
    if confidence < confidence_threshold:
        return None

    height, width = frame_shape[:2]
    x1 = int(np.clip(bbox_norm[0], 0.0, 1.0) * width)
    y1 = int(np.clip(bbox_norm[1], 0.0, 1.0) * height)
    x2 = int(np.clip(bbox_norm[2], 0.0, 1.0) * width)
    y2 = int(np.clip(bbox_norm[3], 0.0, 1.0) * height)

    if x2 <= x1 or y2 <= y1:
        return None

    return DetectionResult(
        class_id=class_id,
        class_name=class_names[class_id],
        confidence=confidence,
        bbox=(x1, y1, x2, y2),
    )


def draw_bbox(frame: np.ndarray, detection: DetectionResult) -> np.ndarray:
    """Draw one detection with class label and confidence."""
    x1, y1, x2, y2 = detection.bbox
    color = _class_color(detection.class_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{detection.class_name}: {detection.confidence:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    y_text_top = max(y1 - text_h - baseline - 6, 0)
    cv2.rectangle(
        frame,
        (x1, y_text_top),
        (x1 + text_w + 8, y_text_top + text_h + baseline + 6),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 4, y_text_top + text_h + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def _class_color(class_id: int) -> Tuple[int, int, int]:
    palette = {
        0: (0, 200, 255),   # Car
        1: (60, 255, 60),   # Pedestrian
        2: (255, 120, 40),  # Cyclist
    }
    return palette.get(class_id, (255, 255, 255))
