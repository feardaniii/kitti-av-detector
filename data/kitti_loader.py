"""KITTI parsing and sample indexing utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class KITTILabel:
    """Single parsed KITTI object label line."""

    class_name: str
    truncated: float
    occluded: int
    bbox: Tuple[float, float, float, float]


@dataclass(frozen=True)
class KITTISample:
    """Training sample for a single object instance."""

    image_path: Path
    class_id: int
    bbox: Tuple[float, float, float, float]
    image_width: int
    image_height: int


def parse_kitti_label(line: str) -> Optional[KITTILabel]:
    """
    Parse one KITTI label line.

    Expected format:
    type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom ...
    """
    parts = line.strip().split()
    if len(parts) < 8:
        return None

    try:
        class_name = parts[0]
        truncated = float(parts[1])
        occluded = int(parts[2])
        x1 = float(parts[4])
        y1 = float(parts[5])
        x2 = float(parts[6])
        y2 = float(parts[7])
    except (ValueError, IndexError):
        return None

    return KITTILabel(
        class_name=class_name,
        truncated=truncated,
        occluded=occluded,
        bbox=(x1, y1, x2, y2),
    )


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def is_valid_label(
    label: KITTILabel,
    allowed_classes: Sequence[str],
    max_truncated: float,
    max_occluded: int,
    min_bbox_area: float,
) -> bool:
    """Apply class and quality filtering for KITTI annotations."""
    if label.class_name not in allowed_classes:
        return False
    if label.truncated > max_truncated:
        return False
    if label.occluded > max_occluded:
        return False
    if _bbox_area(label.bbox) < min_bbox_area:
        return False
    return True


def parse_kitti_annotation_file(
    label_path: Path,
    allowed_classes: Sequence[str],
    max_truncated: float,
    max_occluded: int,
    min_bbox_area: float,
) -> List[KITTILabel]:
    """Load and filter all labels from one KITTI annotation file."""
    labels: List[KITTILabel] = []
    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        LOGGER.warning("Could not read label file %s: %s", label_path, exc)
        return labels

    if not text:
        return labels

    for line in text.splitlines():
        parsed = parse_kitti_label(line)
        if parsed is None:
            LOGGER.debug("Skipping malformed line in %s: %s", label_path, line)
            continue
        if is_valid_label(
            parsed,
            allowed_classes=allowed_classes,
            max_truncated=max_truncated,
            max_occluded=max_occluded,
            min_bbox_area=min_bbox_area,
        ):
            labels.append(parsed)
    return labels


def build_kitti_samples(
    image_dir: Path,
    label_dir: Path,
    classes: Sequence[str],
    max_truncated: float,
    max_occluded: int,
    min_bbox_area: float,
) -> List[KITTISample]:
    """Build memory-light sample index of (image, class, bbox) tuples."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    class_to_id: Dict[str, int] = {name: i for i, name in enumerate(classes)}
    label_files = sorted(label_dir.glob("*.txt"))
    samples: List[KITTISample] = []
    skipped_no_objects = 0

    for label_file in tqdm(label_files, desc="Indexing KITTI labels"):
        stem = label_file.stem
        image_path = image_dir / f"{stem}.png"
        if not image_path.exists():
            LOGGER.debug("No matching image for %s", label_file)
            continue

        labels = parse_kitti_annotation_file(
            label_path=label_file,
            allowed_classes=classes,
            max_truncated=max_truncated,
            max_occluded=max_occluded,
            min_bbox_area=min_bbox_area,
        )
        if not labels:
            skipped_no_objects += 1
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except OSError as exc:
            LOGGER.warning("Could not read image size for %s: %s", image_path, exc)
            continue

        for label in labels:
            class_id = class_to_id[label.class_name]
            samples.append(
                KITTISample(
                    image_path=image_path,
                    class_id=class_id,
                    bbox=label.bbox,
                    image_width=width,
                    image_height=height,
                )
            )

    LOGGER.info(
        "Built %d object samples from %d label files (%d images with no valid objects).",
        len(samples),
        len(label_files),
        skipped_no_objects,
    )
    return samples
