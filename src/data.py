"""Dataset download and `tf.data` pipeline.

The dataset is fetched from Kaggle on first use and cached by kagglehub.

Split policy
------------
The source dataset ships `Training/` and `Testing/` directories. Validation is
carved out of `Training/` and drives early stopping, while `Testing/` is kept
untouched so the reported score carries no model-selection signal:

    Training/  ->  train + validation  (val_split, used for early stopping)
    Testing/   ->  test                (held out, touched only by evaluate.py)
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from .config import Config

AUTOTUNE = tf.data.AUTOTUNE


def download_dataset(cfg: Config) -> Path:
    """Download the Kaggle dataset and return the directory holding the splits.

    Requires Kaggle credentials (`~/.kaggle/kaggle.json`, or the KAGGLE_USERNAME
    and KAGGLE_KEY environment variables).
    """
    import kagglehub

    root = Path(kagglehub.dataset_download(cfg.data.kaggle_dataset))

    # kagglehub sometimes returns a wrapper directory containing the real split
    # folders, so locate `Training/` rather than assuming it sits at the root.
    for candidate in (root, *sorted(p for p in root.iterdir() if p.is_dir())):
        if (candidate / "Training").is_dir() and (candidate / "Testing").is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not find Training/ and Testing/ under {root}. "
        "The dataset layout may have changed."
    )


def _load_directory(
    directory: Path,
    cfg: Config,
    *,
    subset: str | None = None,
    shuffle: bool,
) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        # Restricting class_names to the three tumor types is what drops
        # `notumor`; images in other subdirectories are never loaded.
        class_names=cfg.data.classes,
        color_mode=cfg.data.color_mode,
        batch_size=cfg.train.batch_size,
        image_size=cfg.data.image_size,
        shuffle=shuffle,
        seed=cfg.data.seed,
        validation_split=cfg.data.val_split if subset else None,
        subset=subset,
        interpolation="bilinear",
    )


def build_datasets(
    cfg: Config, dataset_dir: Path | None = None
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Return `(train, val, test)` datasets, batched and prefetched.

    Images are left in the raw 0-255 range. Rescaling lives inside the model
    (see `models.py`) so that inference never has to remember to normalize.
    """
    root = Path(dataset_dir) if dataset_dir else download_dataset(cfg)

    train = _load_directory(root / "Training", cfg, subset="training", shuffle=True)
    val = _load_directory(root / "Training", cfg, subset="validation", shuffle=True)
    test = _load_directory(root / "Testing", cfg, shuffle=False)

    # Cache after decoding but before prefetch; these splits fit in memory at
    # 128x128 grayscale. Only the training split is reshuffled each epoch.
    train = train.cache().shuffle(1000, seed=cfg.data.seed).prefetch(AUTOTUNE)
    val = val.cache().prefetch(AUTOTUNE)
    test = test.cache().prefetch(AUTOTUNE)

    return train, val, test


def class_distribution(dataset_dir: Path, cfg: Config) -> dict[str, dict[str, int]]:
    """Count images per class per split, for the dataset table in the README."""
    counts: dict[str, dict[str, int]] = {}
    for split in ("Training", "Testing"):
        counts[split] = {
            name: sum(1 for _ in (dataset_dir / split / name).glob("*"))
            for name in cfg.data.classes
        }
    return counts
