"""Central configuration for training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class ProjectConfig:
    """Container for project-wide paths and hyperparameters."""

    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    # Dataset
    dataset_root_override: Optional[Path] = None
    dataset_root: Path = field(init=False)
    image_dir: Path = field(init=False)
    label_dir: Path = field(init=False)

    # Output paths
    outputs_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    videos_dir: Path = field(init=False)
    final_model_path: Path = field(init=False)
    best_model_path: Path = field(init=False)
    default_output_video: Path = field(init=False)

    # Classes
    classes: Tuple[str, ...] = ("Car", "Pedestrian", "Cyclist")

    # Image / model
    input_height: int = 224
    input_width: int = 224
    channels: int = 3

    # Filtering
    max_truncated: float = 0.5
    max_occluded: int = 1
    min_bbox_area: float = 32.0 * 32.0

    # Training
    seed: int = 42
    batch_size: int = 32
    val_split: float = 0.2
    stage1_epochs: int = 10
    stage2_epochs: int = 10
    stage1_lr: float = 1e-3
    stage2_lr: float = 1e-4
    patience_early_stopping: int = 5
    patience_reduce_lr: int = 3
    min_lr: float = 1e-6
    fine_tune_last_n_layers: int = 30
    max_train_samples: Optional[int] = None
    skip_fine_tune: bool = False
    train_workers: int = 4

    # Inference
    confidence_threshold: float = 0.5

    def __post_init__(self) -> None:
        env_dataset_root = os.getenv("KITTI_DATASET_ROOT")
        dataset_root = (
            Path(self.dataset_root_override).expanduser()
            if self.dataset_root_override is not None
            else (
                Path(env_dataset_root).expanduser()
                if env_dataset_root
                else self.root_dir / "datasets" / "kitti"
            )
        )
        object.__setattr__(self, "dataset_root", dataset_root)
        object.__setattr__(self, "image_dir", dataset_root / "training" / "image_2")
        object.__setattr__(self, "label_dir", dataset_root / "training" / "label_2")

        outputs_dir = self.root_dir / "outputs"
        models_dir = outputs_dir / "models"
        videos_dir = outputs_dir / "videos"
        object.__setattr__(self, "outputs_dir", outputs_dir)
        object.__setattr__(self, "models_dir", models_dir)
        object.__setattr__(self, "videos_dir", videos_dir)
        object.__setattr__(self, "final_model_path", models_dir / "av_perception_final.keras")
        object.__setattr__(self, "best_model_path", models_dir / "av_perception_best.keras")
        object.__setattr__(
            self, "default_output_video", videos_dir / "annotated_output.mp4"
        )

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Return model input shape as (H, W, C)."""
        return (self.input_height, self.input_width, self.channels)


def ensure_output_dirs(config: ProjectConfig) -> None:
    """Ensure output directories exist before writing artifacts."""
    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.videos_dir.mkdir(parents=True, exist_ok=True)
