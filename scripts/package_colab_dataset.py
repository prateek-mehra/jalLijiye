#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package train/val/test YOLO dataset for Colab training."
    )
    parser.add_argument("--train-images", default="data/bottle/images/train")
    parser.add_argument("--train-labels", default="data/bottle/labels/train")
    parser.add_argument("--val-images", default="data/bottle/images/val")
    parser.add_argument("--val-labels", default="data/bottle/labels/val")
    parser.add_argument("--test-images", default="data/bottle/imports/pink_bottle_v2_yolo/test/images")
    parser.add_argument("--test-labels", default="data/bottle/imports/pink_bottle_v2_yolo/test/labels")
    parser.add_argument("--out-root", default="data/bottle_colab")
    parser.add_argument("--zip", action="store_true")
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_split(src_images: Path, src_labels: Path, dst_images: Path, dst_labels: Path) -> int:
    count = 0
    for img in sorted(src_images.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = src_labels / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        shutil.copy2(img, dst_images / img.name)
        shutil.copy2(lbl, dst_labels / lbl.name)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    reset_dir(out_root)

    split_args = {
        "train": (Path(args.train_images), Path(args.train_labels)),
        "val": (Path(args.val_images), Path(args.val_labels)),
        "test": (Path(args.test_images), Path(args.test_labels)),
    }

    counts: dict[str, int] = {}
    for split, (src_images, src_labels) in split_args.items():
        if not src_images.exists() or not src_labels.exists():
            raise SystemExit(f"[ERROR] missing split source: {split}")
        dst_images = out_root / "images" / split
        dst_labels = out_root / "labels" / split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        counts[split] = copy_split(src_images, src_labels, dst_images, dst_labels)

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                "  0: my_bottle",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if args.zip:
        shutil.make_archive(str(out_root), "zip", root_dir=str(out_root))

    print(
        f"[DONE] train={counts['train']} val={counts['val']} test={counts['test']} "
        f"out={out_root}"
    )


if __name__ == "__main__":
    main()
