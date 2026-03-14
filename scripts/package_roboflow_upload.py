#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package paired YOLO images+labels into a Roboflow upload folder."
    )
    parser.add_argument("--images-dir", default="data/bottle/images/raw", help="Source images dir")
    parser.add_argument("--labels-dir", default="data/bottle/labels/raw", help="Source labels dir")
    parser.add_argument(
        "--out-dir",
        default="data/bottle/roboflow_upload",
        help="Output folder to create/reset",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create a zip archive of out-dir",
    )
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)

    if not images_dir.exists():
        print(f"[ERROR] images dir not found: {images_dir}")
        raise SystemExit(1)
    if not labels_dir.exists():
        print(f"[ERROR] labels dir not found: {labels_dir}")
        raise SystemExit(1)

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    reset_dir(out_images)
    reset_dir(out_labels)

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    paired = 0
    missing_labels = 0

    for img in images:
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            missing_labels += 1
            continue
        shutil.copy2(img, out_images / img.name)
        shutil.copy2(lbl, out_labels / lbl.name)
        paired += 1

    if paired == 0:
        print("[ERROR] no image/label pairs found")
        raise SystemExit(1)

    archive_path: Path | None = None
    if args.zip:
        archive_base = out_dir.parent / out_dir.name
        archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=str(out_dir)))

    print(
        f"[DONE] paired={paired}, missing_labels={missing_labels}, "
        f"out_dir={out_dir}"
    )
    if archive_path is not None:
        print(f"[DONE] zip={archive_path}")


if __name__ == "__main__":
    main()
