#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label images with a YOLO model and write YOLO txt labels."
    )
    parser.add_argument("--images-dir", required=True, help="Input images directory")
    parser.add_argument("--labels-dir", required=True, help="Output labels directory")
    parser.add_argument(
        "--model-path",
        default="models/bottle_v1/weights/best.pt",
        help="Path to current bottle model",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Target class id from the model",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for pseudo labels",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    parser.add_argument(
        "--write-empty-labels",
        action="store_true",
        help="Also write empty txt files when no detection is found",
    )
    return parser.parse_args()


def to_yolo_line(x1: float, y1: float, x2: float, y2: float, w: int, h: int, cls_id: int) -> str:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + (bw / 2.0)
    cy = y1 + (bh / 2.0)
    return f"{cls_id} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    model_path = Path(args.model_path)

    if not images_dir.exists():
        print(f"[ERROR] images dir not found: {images_dir}")
        raise SystemExit(1)
    if not model_path.exists():
        print(f"[ERROR] model path not found: {model_path}")
        raise SystemExit(1)

    labels_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        print(f"[ERROR] no images found in: {images_dir}")
        raise SystemExit(1)

    total = len(images)
    labeled = 0
    empty = 0
    total_boxes = 0

    for idx, img_path in enumerate(images, start=1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] unreadable image: {img_path}")
            continue
        h, w = image.shape[:2]

        results = model.predict(
            source=image,
            conf=args.conf,
            imgsz=args.imgsz,
            classes=[args.class_id],
            verbose=False,
        )

        out_txt = labels_dir / f"{img_path.stem}.txt"
        lines: list[str] = []
        if results:
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None:
                for box in boxes:
                    try:
                        cls_val = int(float(box.cls[0]))
                        conf_val = float(box.conf[0])
                        if cls_val != args.class_id or conf_val < args.conf:
                            continue
                        xyxy = box.xyxy[0].tolist()
                        if len(xyxy) < 4:
                            continue
                        lines.append(to_yolo_line(xyxy[0], xyxy[1], xyxy[2], xyxy[3], w, h, args.class_id))
                    except Exception:
                        continue

        if lines:
            out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
            labeled += 1
            total_boxes += len(lines)
        elif args.write_empty_labels:
            out_txt.write_text("", encoding="utf-8")
            empty += 1
        else:
            if out_txt.exists():
                out_txt.unlink()
            empty += 1

        if idx % 25 == 0 or idx == total:
            print(f"[INFO] {idx}/{total} processed")

    print(
        f"[DONE] images={total}, labeled={labeled}, empty={empty}, "
        f"boxes={total_boxes}, labels_dir={labels_dir}"
    )


if __name__ == "__main__":
    main()
