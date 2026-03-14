#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split YOLO dataset into train/val.")
    p.add_argument("--images-raw", default="data/bottle/images/raw")
    p.add_argument("--labels-raw", default="data/bottle/labels/raw")
    p.add_argument("--out-root", default="data/bottle")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--require-non-empty-labels",
        action="store_true",
        help="Only include images whose YOLO label file contains at least one box",
    )
    return p.parse_args()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    images_raw = Path(args.images_raw)
    labels_raw = Path(args.labels_raw)
    out_root = Path(args.out_root)

    if not images_raw.exists() or not labels_raw.exists():
        print("[ERROR] raw image/label folders are missing")
        raise SystemExit(1)

    images = sorted([p for p in images_raw.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        print("[ERROR] no images found in raw folder")
        raise SystemExit(1)

    pairs: list[tuple[Path, Path]] = []
    for img in images:
        lbl = labels_raw / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        if args.require_non_empty_labels and not lbl.read_text(encoding="utf-8").strip():
            continue
        pairs.append((img, lbl))

    if not pairs:
        print("[ERROR] no matching labels found for images")
        raise SystemExit(1)

    random.seed(args.seed)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * args.train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    train_img_dir = out_root / "images" / "train"
    val_img_dir = out_root / "images" / "val"
    train_lbl_dir = out_root / "labels" / "train"
    val_lbl_dir = out_root / "labels" / "val"

    for d in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
        reset_dir(d)

    for img, lbl in train_pairs:
        shutil.copy2(img, train_img_dir / img.name)
        shutil.copy2(lbl, train_lbl_dir / lbl.name)

    for img, lbl in val_pairs:
        shutil.copy2(img, val_img_dir / img.name)
        shutil.copy2(lbl, val_lbl_dir / lbl.name)

    print(f"[DONE] train={len(train_pairs)} val={len(val_pairs)}")
    print(f"[INFO] output: {out_root}")


if __name__ == "__main__":
    main()
