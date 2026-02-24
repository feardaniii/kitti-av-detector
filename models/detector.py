"""Detector model definition and fine-tuning utilities."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers


def build_detector(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    learning_rate: float,
    freeze_backbone: bool = True,
) -> Model:
    """Build and compile a MobileNetV2-based multi-task detector."""
    inputs = layers.Input(shape=input_shape, name="image_input")

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = not freeze_backbone
    backbone._name = "backbone"

    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dense(256, activation="relu", name="shared_dense_1")(x)
    x = layers.Dropout(0.3, name="shared_dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="shared_dense_2")(x)
    x = layers.Dropout(0.2, name="shared_dropout_2")(x)

    class_output = layers.Dense(
        num_classes, activation="softmax", name="class_output"
    )(x)
    bbox_output = layers.Dense(4, activation="sigmoid", name="bbox_output")(x)

    model = Model(inputs=inputs, outputs=[class_output, bbox_output], name="av_detector")
    _compile_model(model=model, learning_rate=learning_rate)
    return model


def _compile_model(model: Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "class_output": "sparse_categorical_crossentropy",
            "bbox_output": "mean_squared_error",
        },
        loss_weights={"class_output": 1.0, "bbox_output": 5.0},
        metrics={"class_output": ["accuracy"], "bbox_output": ["mse"]},
    )


def set_backbone_trainable(
    model: Model, trainable: bool, last_n_layers: int = 30, learning_rate: float = 1e-4
) -> None:
    """Freeze all backbone layers or unfreeze only the last N layers."""
    backbone = model.get_layer("backbone")

    if not trainable:
        backbone.trainable = False
    else:
        backbone.trainable = True
        for layer in backbone.layers[:-last_n_layers]:
            layer.trainable = False
        for layer in backbone.layers[-last_n_layers:]:
            layer.trainable = True

    _compile_model(model=model, learning_rate=learning_rate)
