"""Train one architecture.

    python -m src.train --arch relu
    python -m src.train --arch elu_bn --epochs 60 --learning-rate 0.0005
    python -m src.train --all          # train every architecture for comparison

Artifacts per run: `models/<arch>.keras` (best epoch by val_accuracy),
`reports/metrics/<arch>_history.json`, a CSV log, and TensorBoard traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

# Allow `python src/train.py` (e.g. an IDE's Run button) as well as the
# canonical `python -m src.train`. Run as a file there is no parent package, so
# relative imports would fail; put the repo root on sys.path and name the
# package explicitly.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

from .config import Config, load_config
from .data import build_datasets
from .models import ARCHITECTURES, build


def _callbacks(arch: str, cfg: Config) -> list[tf.keras.callbacks.Callback]:
    ckpt = cfg.paths.models_dir / f"{arch}.keras"
    return [
        # Keep the best epoch by validation accuracy, not whichever one happens
        # to run last.
        tf.keras.callbacks.ModelCheckpoint(
            ckpt, monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max",
            patience=cfg.train.early_stopping_patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=cfg.train.reduce_lr_factor,
            patience=cfg.train.reduce_lr_patience, verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(cfg.paths.logs_dir / arch)),
        tf.keras.callbacks.CSVLogger(str(cfg.paths.metrics_dir / f"{arch}_log.csv")),
    ]


def train_one(arch: str, cfg: Config, datasets=None) -> dict:
    train_ds, val_ds, _ = datasets if datasets else build_datasets(cfg)

    model = build(arch, cfg)
    print(f"\n=== {arch} ===")
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.train.epochs,
        callbacks=_callbacks(arch, cfg),
    )

    record = {
        "arch": arch,
        "epochs_run": len(history.history["loss"]),
        "best_val_accuracy": max(history.history["val_accuracy"]),
        "params": model.count_params(),
        "config": {"data": cfg.raw["data"], "train": cfg.raw["train"]},
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }

    out = cfg.paths.metrics_dir / f"{arch}_history.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nBest val_accuracy: {record['best_val_accuracy']:.4f}")
    print(f"Model  -> {cfg.paths.models_dir / f'{arch}.keras'}")
    print(f"History -> {out}")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a brain tumor MRI classifier.")
    ap.add_argument("--arch", choices=sorted(ARCHITECTURES), default="relu")
    ap.add_argument("--all", action="store_true",
                    help="train every architecture (ablation comparison)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", dest="batch_size", type=int)
    ap.add_argument("--learning-rate", dest="learning_rate", type=float)
    ap.add_argument("--no-augment", dest="augment", action="store_false", default=None)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="use an already-downloaded dataset directory")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(
        args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        augment=args.augment,
    )
    cfg.paths.mkdirs()

    # Build the pipeline once and share it across architectures so every model
    # sees an identical split.
    datasets = build_datasets(cfg, args.data_dir)

    archs = sorted(ARCHITECTURES) if args.all else [args.arch]
    results = [train_one(a, cfg, datasets) for a in archs]

    if len(results) > 1:
        print("\n=== comparison (best val_accuracy) ===")
        for r in sorted(results, key=lambda r: -r["best_val_accuracy"]):
            print(f"  {r['arch']:<10} {r['best_val_accuracy']:.4f}"
                  f"   ({r['params']:,} params, {r['epochs_run']} epochs)")


if __name__ == "__main__":
    main()
