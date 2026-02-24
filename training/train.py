"""Training pipeline for the AV detector."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config.config import ProjectConfig, ensure_output_dirs
from data.generator import KITTIDataGenerator
from data.kitti_loader import KITTISample, build_kitti_samples
from models.detector import build_detector, set_backbone_trainable


LOGGER = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set all major RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def split_samples(
    samples: Sequence[KITTISample], val_split: float, seed: int
) -> Tuple[List[KITTISample], List[KITTISample]]:
    """Deterministically split samples into train and validation sets."""
    if not samples:
        return [], []

    indices = np.arange(len(samples))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_size = int(len(samples) * val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    return train_samples, val_samples


def build_callbacks(config: ProjectConfig) -> List[tf.keras.callbacks.Callback]:
    """Build standard callbacks for stable training."""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=config.patience_early_stopping,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(config.best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.patience_reduce_lr,
            min_lr=config.min_lr,
            verbose=1,
        ),
    ]


def configure_runtime() -> None:
    """Enable runtime optimizations when supported."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            LOGGER.info("GPU detected (%d). Mixed precision enabled.", len(gpus))
        except ValueError:
            LOGGER.info("GPU detected (%d). Mixed precision not available.", len(gpus))
    else:
        LOGGER.info("No GPU detected by TensorFlow. Training will run on CPU.")


def fit_with_compat(
    model: tf.keras.Model,
    train_gen: KITTIDataGenerator,
    val_gen: KITTIDataGenerator,
    epochs: int,
    fit_kwargs: Dict[str, Any],
) -> None:
    """Fit while handling TensorFlow versions with differing fit kwargs."""
    try:
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            **fit_kwargs,
        )
    except TypeError:
        LOGGER.warning("Falling back to fit() without worker multiprocessing arguments.")
        fallback_kwargs = {k: v for k, v in fit_kwargs.items() if k not in {"workers", "use_multiprocessing"}}
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            **fallback_kwargs,
        )


def train_model(config: ProjectConfig) -> Path:
    """Run full two-stage training and save final model."""
    configure_runtime()
    set_global_seed(config.seed)
    ensure_output_dirs(config)

    LOGGER.info("Building KITTI sample index...")
    samples = build_kitti_samples(
        image_dir=config.image_dir,
        label_dir=config.label_dir,
        classes=config.classes,
        max_truncated=config.max_truncated,
        max_occluded=config.max_occluded,
        min_bbox_area=config.min_bbox_area,
    )
    if not samples:
        raise RuntimeError(
            "No valid training samples found. Verify dataset path and filter thresholds."
        )
    if config.max_train_samples is not None and config.max_train_samples > 0:
        limited_count = min(config.max_train_samples, len(samples))
        samples = samples[:limited_count]
        LOGGER.info("Limiting training samples to %d for faster iteration.", limited_count)

    train_samples, val_samples = split_samples(
        samples=samples, val_split=config.val_split, seed=config.seed
    )
    if not train_samples or not val_samples:
        raise RuntimeError(
            f"Invalid split sizes. train={len(train_samples)}, val={len(val_samples)}"
        )

    LOGGER.info("Train samples: %d, Val samples: %d", len(train_samples), len(val_samples))

    train_gen = KITTIDataGenerator(
        samples=train_samples,
        input_size=(config.input_width, config.input_height),
        batch_size=config.batch_size,
        num_classes=len(config.classes),
        shuffle=True,
        seed=config.seed,
    )
    val_gen = KITTIDataGenerator(
        samples=val_samples,
        input_size=(config.input_width, config.input_height),
        batch_size=config.batch_size,
        num_classes=len(config.classes),
        shuffle=False,
        seed=config.seed,
    )

    model = build_detector(
        input_shape=config.input_shape,
        num_classes=len(config.classes),
        learning_rate=config.stage1_lr,
        freeze_backbone=True,
    )
    callbacks = build_callbacks(config)
    fit_kwargs: Dict[str, Any] = {
        "callbacks": callbacks,
        "verbose": 1,
        "workers": max(1, int(config.train_workers)),
        "use_multiprocessing": True,
    }

    LOGGER.info("Stage 1 training: frozen backbone")
    fit_with_compat(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        epochs=config.stage1_epochs,
        fit_kwargs=fit_kwargs,
    )

    if config.skip_fine_tune or config.stage2_epochs <= 0:
        LOGGER.info("Skipping stage 2 fine-tuning.")
    else:
        LOGGER.info(
            "Stage 2 training: fine-tune last %d backbone layers",
            config.fine_tune_last_n_layers,
        )
        set_backbone_trainable(
            model=model,
            trainable=True,
            last_n_layers=config.fine_tune_last_n_layers,
            learning_rate=config.stage2_lr,
        )
        fit_with_compat(
            model=model,
            train_gen=train_gen,
            val_gen=val_gen,
            epochs=config.stage2_epochs,
            fit_kwargs=fit_kwargs,
        )

    model.save(config.final_model_path)
    LOGGER.info("Final model saved to: %s", config.final_model_path)
    return config.final_model_path
