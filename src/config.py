"""Configuration loading.

Every script reads `config.yaml` through `load_config()` so that a run is fully
described by that file plus whatever the CLI overrode.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class DataConfig:
    kaggle_dataset: str
    classes: list[str]
    image_size: tuple[int, int]
    color_mode: str
    val_split: float
    seed: int

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def channels(self) -> int:
        return 1 if self.color_mode == "grayscale" else 3

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (*self.image_size, self.channels)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    dropout: float
    augment: bool
    rotation: float
    zoom: float
    early_stopping_patience: int
    reduce_lr_patience: int
    reduce_lr_factor: float


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    models_dir: Path
    logs_dir: Path
    figures_dir: Path
    metrics_dir: Path

    def mkdirs(self) -> None:
        for p in (self.data_dir, self.models_dir, self.logs_dir,
                  self.figures_dir, self.metrics_dir):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    data: DataConfig
    train: TrainConfig
    paths: Paths
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load `config.yaml`, applying non-None keyword overrides from the CLI.

    Overrides are matched against the `data` and `train` sections by field name,
    so `load_config(epochs=5)` updates `train.epochs`.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    data = DataConfig(
        kaggle_dataset=raw["data"]["kaggle_dataset"],
        classes=list(raw["data"]["classes"]),
        image_size=tuple(raw["data"]["image_size"]),
        color_mode=raw["data"]["color_mode"],
        val_split=float(raw["data"]["val_split"]),
        seed=int(raw["data"]["seed"]),
    )
    train = TrainConfig(**raw["train"])

    # Paths in the YAML are relative to the repo root, not the caller's cwd.
    paths = Paths(**{k: REPO_ROOT / v for k, v in raw["paths"].items()})

    supplied = {k: v for k, v in overrides.items() if v is not None}
    data_fields = {k: v for k, v in supplied.items() if hasattr(data, k)}
    train_fields = {k: v for k, v in supplied.items() if hasattr(train, k)}

    unknown = set(supplied) - set(data_fields) - set(train_fields)
    if unknown:
        raise ValueError(f"Unknown config override(s): {sorted(unknown)}")

    return Config(
        data=replace(data, **data_fields),
        train=replace(train, **train_fields),
        paths=paths,
        raw=raw,
    )
