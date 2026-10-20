#!/usr/bin/env python3
"""Create contiguous train/validation/test splits for the checkered-bottle clip."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/bottle_checkered"))
    args = parser.parse_args()
    root = args.root
    raw_images = sorted(p for p in (root / "images" / "raw").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    raw_labels = root / "labels" / "raw"
    if not raw_images:
        raise SystemExit(f"No raw images found under {root / 'images' / 'raw'}")

    missing = [image.name for image in raw_images if not (raw_labels / f"{image.stem}.txt").exists()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} label file(s), e.g. {missing[:3]}")

    # Keep adjacent frames together so near-identical frames do not cross splits.
    train_end = round(len(raw_images) * 0.65)
    val_end = round(len(raw_images) * 0.825)
    ranges = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, len(raw_images)),
    }
    counts: dict[str, tuple[int, int]] = {}
    for split, (start, end) in ranges.items():
        for kind in ("images", "labels"):
            destination = root / kind / split
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True)
        positive = 0
        for image in raw_images[start:end]:
            label = raw_labels / f"{image.stem}.txt"
            shutil.copy2(image, root / "images" / split / image.name)
            shutil.copy2(label, root / "labels" / split / label.name)
            positive += bool(label.read_text(encoding="utf-8").strip())
        counts[split] = (end - start, positive)

    (root / "data.yaml").write_text(
        "path: data/bottle_checkered\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: my_bottle\n",
        encoding="utf-8",
    )
    for split, (total, positive) in counts.items():
        print(f"{split}: {total} images, {positive} with bottle labels, {total - positive} negatives")
    print(f"[DONE] Dataset config: {root / 'data.yaml'}")


if __name__ == "__main__":
    main()
