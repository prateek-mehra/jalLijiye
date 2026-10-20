#!/usr/bin/env python3
"""Create draft YOLO detection labels with the MLX SAM 3 text predictor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
from mlx_vlm.utils import get_model_path, load_model


DEFAULT_PROMPTS = [
    "black and white checkered bottle",
    "black and white checkered sipper",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/bottle_checkered/images/raw"))
    parser.add_argument("--labels", type=Path, default=Path("data/bottle_checkered/labels/raw"))
    parser.add_argument("--previews", type=Path, default=Path("data/bottle_checkered/previews"))
    parser.add_argument("--model", default="mlx-community/sam3-4bit")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the sorted image list")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images for a short, low-memory smoke run")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images that already have a YOLO label file")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    return parser.parse_args()


def as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "tolist"):
        return np.asarray(value.tolist())
    return np.asarray(value)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def suppress_duplicates(
    detections: list[tuple[np.ndarray, float, str]], threshold: float
) -> list[tuple[np.ndarray, float, str]]:
    kept: list[tuple[np.ndarray, float, str]] = []
    for candidate in sorted(detections, key=lambda item: item[1], reverse=True):
        if all(iou_xyxy(candidate[0], chosen[0]) < threshold for chosen in kept):
            kept.append(candidate)
    return kept


def main() -> None:
    args = parse_args()
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    start_index = max(0, args.start_index)
    images = images[start_index : start_index + args.max_images if args.max_images > 0 else None]
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    model_path = get_model_path(args.model)
    model = load_model(model_path)
    processor = Sam3Processor.from_pretrained(str(model_path))
    predictor = Sam3Predictor(model, processor, score_threshold=args.threshold)
    args.labels.mkdir(parents=True, exist_ok=True)
    args.previews.mkdir(parents=True, exist_ok=True)

    summary_path = args.previews / "annotations.json"
    if summary_path.exists():
        summary: list[dict[str, object]] = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = []
    processed = 0
    for image_path in images:
        label_path = args.labels / f"{image_path.stem}.txt"
        if args.skip_existing and label_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        candidates: list[tuple[np.ndarray, float, str]] = []
        for prompt in args.prompts:
            result = predictor.predict(image, text_prompt=prompt)
            boxes = as_numpy(result.boxes).reshape(-1, 4)
            scores = as_numpy(result.scores).reshape(-1)
            for box, score in zip(boxes, scores):
                box = np.asarray(box, dtype=float)
                score = float(score)
                if score >= args.threshold and box[2] > box[0] and box[3] > box[1]:
                    candidates.append((box, score, prompt))
            del result
            mx.clear_cache()

        detections = suppress_duplicates(candidates, args.nms_iou)
        lines: list[str] = []
        canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        for box, score, prompt in detections:
            x1, y1, x2, y2 = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
            cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(canvas, pt1, pt2, (0, 220, 0), 3)
            cv2.putText(
                canvas, f"{score:.2f}: {prompt}", (pt1[0], max(24, pt1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2, cv2.LINE_AA,
            )

        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        cv2.imwrite(str(args.previews / f"{image_path.stem}.jpg"), canvas)
        summary.append({
            "image": str(image_path),
            "prompts": args.prompts,
            "detections": [
                {"box_xyxy": box.tolist(), "score": score, "prompt": prompt}
                for box, score, prompt in detections
            ],
        })
        processed += 1
        print(f"[{processed}/{len(images)}] {image_path.name}: {len(detections)} box(es)", flush=True)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    positive = sum(bool(row["detections"]) for row in summary)
    print(f"[DONE] processed {len(images)} image(s); {positive} total images with detections; labels: {args.labels}; previews: {args.previews}")


if __name__ == "__main__":
    main()
