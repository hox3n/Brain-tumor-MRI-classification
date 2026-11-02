"""Evaluate a trained model on the held-out test split.

    python -m src.evaluate --arch relu

Writes a confusion matrix and training-curve figure to `reports/figures/`, and
per-class precision/recall/F1 to `reports/metrics/`. Per-class numbers matter
more than an aggregate here, since they show which tumor type a model confuses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

# Allow `python src/evaluate.py` (e.g. an IDE's Run button) as well as the
# canonical `python -m src.evaluate`. See the note in train.py.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

from .config import Config, load_config
from .data import build_datasets
from .models import ARCHITECTURES


def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset):
    """Return `(y_true, y_pred)` as class indices over the whole dataset."""
    y_true, y_pred = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_true.append(np.argmax(labels.numpy(), axis=1))
        y_pred.append(np.argmax(probs, axis=1))
    return np.concatenate(y_true), np.concatenate(y_pred)


def plot_confusion(y_true, y_pred, classes: list[str], out: Path, arch: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, normalize, title in (
        (axes[0], None, "Counts"),
        (axes[1], "true", "Row-normalized (recall)"),
    ):
        cm_n = confusion_matrix(y_true, y_pred, labels=range(len(classes)),
                                normalize=normalize)
        disp = ConfusionMatrixDisplay(cm_n, display_labels=classes)
        disp.plot(ax=ax, cmap="Blues", colorbar=False,
                  values_format="d" if normalize is None else ".2f")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(f"Confusion matrix: {arch} (test split)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix -> {out}")
    return cm


def plot_history(history: dict, out: Path, arch: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (key, val_key, label) in zip(
        axes, [("accuracy", "val_accuracy", "Accuracy"), ("loss", "val_loss", "Loss")]
    ):
        ax.plot(history[key], color="teal", label=f"train {label.lower()}")
        ax.plot(history[val_key], color="crimson", label=f"val {label.lower()}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"Training curves: {arch}")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves -> {out}")


def evaluate(arch: str, cfg: Config, data_dir: Path | None = None) -> dict:
    model_path = cfg.paths.models_dir / f"{arch}.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python -m src.train --arch {arch}` first."
        )

    model = tf.keras.models.load_model(model_path)
    _, _, test_ds = build_datasets(cfg, data_dir)

    loss_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
    y_true, y_pred = collect_predictions(model, test_ds)
    classes = cfg.data.classes

    report = classification_report(
        y_true, y_pred, target_names=classes, digits=4, output_dict=True, zero_division=0
    )
    print(f"\n=== {arch} on held-out test split ===")
    print(classification_report(y_true, y_pred, target_names=classes,
                                digits=4, zero_division=0))

    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    cm = plot_confusion(y_true, y_pred, classes,
                        cfg.paths.figures_dir / f"{arch}_confusion_matrix.png", arch)

    history_path = cfg.paths.metrics_dir / f"{arch}_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
        plot_history(history, cfg.paths.figures_dir / f"{arch}_training_curves.png", arch)

    results = {
        "arch": arch,
        "test_metrics": {k: float(v) for k, v in loss_metrics.items()},
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "n_test": int(len(y_true)),
    }
    out = cfg.paths.metrics_dir / f"{arch}_test_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Metrics -> {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained classifier.")
    ap.add_argument("--arch", choices=sorted(ARCHITECTURES), default="relu")
    ap.add_argument("--all", action="store_true", help="evaluate every trained architecture")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.paths.mkdirs()

    archs = sorted(ARCHITECTURES) if args.all else [args.arch]
    results = []
    for arch in archs:
        try:
            results.append(evaluate(arch, cfg, args.data_dir))
        except FileNotFoundError as exc:
            print(f"skipping {arch}: {exc}")

    if len(results) > 1:
        print("\n=== test accuracy / macro F1 ===")
        for r in sorted(results, key=lambda r: -r["classification_report"]["accuracy"]):
            rep = r["classification_report"]
            print(f"  {r['arch']:<10} acc {rep['accuracy']:.4f}"
                  f"   macro-F1 {rep['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
