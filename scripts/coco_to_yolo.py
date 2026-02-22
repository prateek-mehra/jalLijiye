#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert COCO boxes to YOLO txt labels.")
    p.add_argument("--coco-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out-images", required=True)
    p.add_argument("--out-labels", required=True)
    p.add_argument(
        "--category-id",
        type=int,
        default=1,
        help="COCO category_id to keep and map to YOLO class 0",
    )
    p.add_argument("--copy-images", action="store_true", default=True)
    return p.parse_args()


def coco_bbox_to_yolo(bbox: list[float], width: float, height: float) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox]
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    return cx / width, cy / height, w / width, h / height


def main() -> None:
    args = parse_args()
    coco_json = Path(args.coco_json)
    images_dir = Path(args.images_dir)
    out_images = Path(args.out_images)
    out_labels = Path(args.out_labels)

    if not coco_json.exists():
        raise SystemExit(f"COCO json not found: {coco_json}")
    if not images_dir.exists():
        raise SystemExit(f"Images dir not found: {images_dir}")

    if out_images.exists():
        shutil.rmtree(out_images)
    if out_labels.exists():
        shutil.rmtree(out_labels)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    data = json.loads(coco_json.read_text())
    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)

    for ann in data.get("annotations", []):
        if int(ann.get("category_id", -1)) != args.category_id:
            continue
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        anns_by_image[int(ann["image_id"])].append(ann)

    kept_images = 0
    kept_boxes = 0

    for image_id, image_info in images.items():
        file_name = image_info["file_name"]
        src_img = images_dir / file_name
        if not src_img.exists():
            continue

        width = float(image_info["width"])
        height = float(image_info["height"])
        anns = anns_by_image.get(image_id, [])

        if args.copy_images:
            shutil.copy2(src_img, out_images / file_name)

        yolo_lines: list[str] = []
        for ann in anns:
            x, y, w, h = coco_bbox_to_yolo(ann["bbox"], width, height)
            if w <= 0 or h <= 0:
                continue
            yolo_lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            kept_boxes += 1

        label_path = out_labels / f"{Path(file_name).stem}.txt"
        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        kept_images += 1

    print(f"[DONE] converted images={kept_images} boxes={kept_boxes}")
    print(f"[INFO] images -> {out_images}")
    print(f"[INFO] labels -> {out_labels}")


if __name__ == "__main__":
    main()
