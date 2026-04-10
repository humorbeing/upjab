"""Create a small static MNIST sample JSON file for the web app.

The preferred path uses torchvision.datasets.MNIST. A local IDX fallback is
included so the script can regenerate samples when torchvision is unavailable
but MNIST raw files already exist under ./data/MNIST/raw.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
OUTPUT_PATH = ROOT / "samples" / "mnist_samples.json"


def normalize_pixel(value: int) -> float:
    return round(value / 255.0, 6)


def samples_from_torchvision() -> list[dict[str, object]]:
    from torchvision import datasets  # type: ignore

    dataset = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True)
    counts = {digit: 0 for digit in range(10)}
    samples: list[dict[str, object]] = []

    for image, label in dataset:
        label = int(label)
        if counts[label] >= 10:
            continue

        grayscale = image.convert("L")
        values = list(grayscale.getdata())
        width, height = grayscale.size
        pixels = [
            [normalize_pixel(values[row * width + col]) for col in range(width)]
            for row in range(height)
        ]
        samples.append(
            {
                "label": label,
                "index_within_class": counts[label],
                "pixels": pixels,
            }
        )
        counts[label] += 1

        if all(count == 10 for count in counts.values()):
            break

    return samples


def read_idx_images(path: Path) -> Iterable[list[list[float]]]:
    with path.open("rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image IDX magic number: {magic}")
        for _ in range(count):
            raw = file.read(rows * cols)
            yield [
                [normalize_pixel(raw[row * cols + col]) for col in range(cols)]
                for row in range(rows)
            ]


def read_idx_labels(path: Path) -> Iterable[int]:
    with path.open("rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label IDX magic number: {magic}")
        for value in file.read(count):
            yield int(value)


def samples_from_idx_files() -> list[dict[str, object]]:
    raw_dir = DATA_ROOT / "MNIST" / "raw"
    images_path = raw_dir / "train-images-idx3-ubyte"
    labels_path = raw_dir / "train-labels-idx1-ubyte"
    if not images_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "MNIST raw IDX files were not found. Install torchvision or place "
            "train-images-idx3-ubyte and train-labels-idx1-ubyte in data/MNIST/raw."
        )

    counts = {digit: 0 for digit in range(10)}
    samples: list[dict[str, object]] = []
    for pixels, label in zip(read_idx_images(images_path), read_idx_labels(labels_path)):
        if counts[label] >= 10:
            continue

        samples.append(
            {
                "label": label,
                "index_within_class": counts[label],
                "pixels": pixels,
            }
        )
        counts[label] += 1

        if all(count == 10 for count in counts.values()):
            break

    return samples


def collect_samples() -> list[dict[str, object]]:
    try:
        return samples_from_torchvision()
    except Exception as exc:
        print(f"torchvision path unavailable ({exc}); using local IDX files.")
        return samples_from_idx_files()


def main() -> None:
    samples = collect_samples()
    if len(samples) != 100:
        raise RuntimeError(f"Expected 100 samples, got {len(samples)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"images": samples}, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
