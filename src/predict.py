"""Run a trained model on new images.

    python -m src.predict --arch relu --image path/to/scan.jpg
    python -m src.predict --arch relu --image path/to/folder --top 3

Rescaling lives inside the saved model, so no preprocessing is required here
beyond decode and resize.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

# Allow `python src/predict.py` (e.g. an IDE's Run button) as well as the
# canonical `python -m src.predict`. See the note in train.py.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

from .config import Config, load_config
from .models import ARCHITECTURES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image(path: Path, cfg: Config) -> np.ndarray:
    raw = tf.io.read_file(str(path))
    img = tf.io.decode_image(raw, channels=cfg.data.channels, expand_animations=False)
    img = tf.image.resize(img, cfg.data.image_size, method="bilinear")
    return tf.expand_dims(img, 0).numpy()


def predict_paths(model: tf.keras.Model, paths: list[Path], cfg: Config, top: int):
    for path in paths:
        probs = model.predict(load_image(path, cfg), verbose=0)[0]
        order = np.argsort(probs)[::-1][:top]
        best = cfg.data.classes[order[0]]
        print(f"\n{path.name}\n  -> {best}  ({probs[order[0]]:.1%} confidence)")
        for idx in order:
            bar = "#" * int(round(probs[idx] * 30))
            print(f"     {cfg.data.classes[idx]:<12} {probs[idx]:6.2%}  {bar}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify brain MRI scans.")
    ap.add_argument("--arch", choices=sorted(ARCHITECTURES), default="relu")
    ap.add_argument("--image", type=Path, required=True,
                    help="an image file, or a directory of images")
    ap.add_argument("--top", type=int, default=3, help="how many classes to list")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)

    model_path = cfg.paths.models_dir / f"{args.arch}.keras"
    if not model_path.exists():
        raise SystemExit(
            f"No trained model at {model_path}. "
            f"Run `python -m src.train --arch {args.arch}` first."
        )
    model = tf.keras.models.load_model(model_path)

    if args.image.is_dir():
        paths = sorted(p for p in args.image.iterdir()
                       if p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise SystemExit(f"No images found in {args.image}")
    elif args.image.is_file():
        paths = [args.image]
    else:
        raise SystemExit(f"No such file or directory: {args.image}")

    predict_paths(model, paths, cfg, max(1, min(args.top, cfg.data.num_classes)))


if __name__ == "__main__":
    main()
