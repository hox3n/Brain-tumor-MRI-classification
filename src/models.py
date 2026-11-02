"""CNN architectures.

A single topology is parameterized by activation and batch normalization, with
the learning rate held fixed by `config.yaml`. Varying one thing at a time makes
the comparison a genuine ablation: `relu` vs `relu_bn` isolates normalization,
and `relu_bn` vs `elu_bn` isolates the activation function.
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf
from tensorflow.keras import layers, models

from .config import Config


def _augmentation(cfg: Config) -> list[layers.Layer]:
    """Light augmentation. MRI scans share a canonical orientation, so we use a
    horizontal flip and small rotation/zoom only, and no vertical flip."""
    if not cfg.train.augment:
        return []
    return [
        layers.RandomFlip("horizontal", seed=cfg.data.seed),
        layers.RandomRotation(cfg.train.rotation, seed=cfg.data.seed),
        layers.RandomZoom(cfg.train.zoom, seed=cfg.data.seed),
    ]


def build_cnn(
    cfg: Config,
    *,
    activation: str = "relu",
    batch_norm: bool = False,
    name: str = "cnn",
) -> tf.keras.Model:
    """Build the CNN topology.

    Rescaling is the first layer rather than a `dataset.map(x/255)` step, so the
    normalization travels with the saved model and inference cannot forget it.
    """
    def conv(filters: int, kernel: int | tuple[int, int], strides=1) -> list[layers.Layer]:
        block: list[layers.Layer] = [
            layers.Conv2D(filters, kernel, strides=strides,
                          activation=None, padding="same")
        ]
        # BN sits before the non-linearity.
        if batch_norm:
            block.append(layers.BatchNormalization())
        block.append(layers.Activation(activation))
        return block

    layer_stack: list[layers.Layer] = [
        layers.Input(shape=cfg.data.input_shape),
        *_augmentation(cfg),
        layers.Rescaling(1.0 / 255),

        *conv(16, 5, strides=2),
        *conv(16, 3, strides=2),

        *conv(32, 3),
        layers.MaxPooling2D(),

        *conv(32, 2),
        layers.MaxPooling2D(),

        *conv(64, 2),
        *conv(64, 2),

        layers.Flatten(),
        layers.Dense(256, activation=activation),
        layers.Dropout(cfg.train.dropout),
        layers.Dense(cfg.data.num_classes, activation="softmax"),
    ]

    model = models.Sequential(layer_stack, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# `relu` is the baseline; the other two each change exactly one thing from it.
ARCHITECTURES: dict[str, Callable[[Config], tf.keras.Model]] = {
    "relu": lambda cfg: build_cnn(cfg, activation="relu", batch_norm=False, name="relu"),
    "relu_bn": lambda cfg: build_cnn(cfg, activation="relu", batch_norm=True, name="relu_bn"),
    "elu_bn": lambda cfg: build_cnn(cfg, activation="elu", batch_norm=True, name="elu_bn"),
}


def build(arch: str, cfg: Config) -> tf.keras.Model:
    if arch not in ARCHITECTURES:
        raise KeyError(f"Unknown architecture {arch!r}. Choose from {sorted(ARCHITECTURES)}.")
    return ARCHITECTURES[arch](cfg)
