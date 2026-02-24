"""Video inference pipeline for generating annotated MP4 output."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import tensorflow as tf
from tqdm import tqdm

from config.config import ProjectConfig, ensure_output_dirs
from inference.utils import draw_bbox, postprocess_prediction, preprocess_frame


LOGGER = logging.getLogger(__name__)


def run_video_inference(
    config: ProjectConfig,
    input_video: Path,
    output_video: Path,
    model_path: Optional[Path] = None,
    confidence_threshold: Optional[float] = None,
) -> Path:
    """Run per-frame detection on a video and write an annotated copy."""
    ensure_output_dirs(config)
    model_file = model_path or config.final_model_path
    threshold = (
        config.confidence_threshold
        if confidence_threshold is None
        else float(confidence_threshold)
    )

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found: {model_file}. Train first or pass --model path."
        )

    LOGGER.info("Loading model: %s", model_file)
    model = tf.keras.models.load_model(model_file)

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
        raise RuntimeError(f"Could not open writer for output video: {output_video}")

    LOGGER.info(
        "Running inference. fps=%.2f, resolution=%dx%d, frames=%d",
        fps,
        width,
        height,
        total_frames,
    )

    try:
        progress = tqdm(total=total_frames if total_frames > 0 else None, desc="Inference")
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            model_input = preprocess_frame(
                frame=frame, input_size=(config.input_width, config.input_height)
            )
            class_pred, bbox_pred = model.predict(model_input, verbose=0)

            detection = postprocess_prediction(
                class_probs=class_pred[0],
                bbox_norm=bbox_pred[0],
                frame_shape=frame.shape,
                class_names=config.classes,
                confidence_threshold=threshold,
            )
            if detection is not None:
                frame = draw_bbox(frame, detection)

            writer.write(frame)
            progress.update(1)
        progress.close()
    finally:
        capture.release()
        writer.release()

    LOGGER.info("Annotated video saved to: %s", output_video)
    return output_video
