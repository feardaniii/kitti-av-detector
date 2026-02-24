"""Keras Sequence generator for KITTI object samples."""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

from data.kitti_loader import KITTISample


LOGGER = logging.getLogger(__name__)


class KITTIDataGenerator(Sequence):
    """Memory-efficient generator yielding image/object batches."""

    def __init__(
        self,
        samples: Sequence[KITTISample],
        input_size: Tuple[int, int],
        batch_size: int,
        num_classes: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.samples: List[KITTISample] = list(samples)
        self.input_width, self.input_height = input_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.indices = np.arange(len(self.samples), dtype=np.int32)
        self.on_epoch_end()

    def __len__(self) -> int:
        """Number of batches per epoch."""
        if len(self.samples) == 0:
            return 0
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Get one batch: image tensor + multi-task targets."""
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_samples = [self.samples[i] for i in batch_indices]

        batch_x = np.zeros(
            (len(batch_samples), self.input_height, self.input_width, 3), dtype=np.float32
        )
        batch_y_class = np.zeros((len(batch_samples),), dtype=np.int32)
        batch_y_bbox = np.zeros((len(batch_samples), 4), dtype=np.float32)

        for i, sample in enumerate(batch_samples):
            image = cv2.imread(str(sample.image_path))
            if image is None:
                LOGGER.warning("Image read failed: %s", sample.image_path)
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(
                image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR
            )
            batch_x[i] = image.astype(np.float32) / 255.0

            batch_y_class[i] = sample.class_id
            batch_y_bbox[i] = self._normalize_bbox(
                bbox=sample.bbox,
                image_width=sample.image_width,
                image_height=sample.image_height,
            )

        targets = {
            "class_output": batch_y_class,
            "bbox_output": batch_y_bbox,
        }
        return batch_x, targets

    def on_epoch_end(self) -> None:
        """Deterministic epoch-based shuffling."""
        if not self.shuffle or len(self.samples) == 0:
            return
        rng = np.random.default_rng(self.seed + self.epoch)
        rng.shuffle(self.indices)
        self.epoch += 1

    @staticmethod
    def _normalize_bbox(
        bbox: Tuple[float, float, float, float], image_width: int, image_height: int
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        norm = np.array(
            [
                x1 / max(float(image_width), 1.0),
                y1 / max(float(image_height), 1.0),
                x2 / max(float(image_width), 1.0),
                y2 / max(float(image_height), 1.0),
            ],
            dtype=np.float32,
        )
        return np.clip(norm, 0.0, 1.0)
