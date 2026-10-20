#!/usr/bin/env python3
"""Compare isolated inference resource use for the two trained bottle models."""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def current_rss_mb() -> float:
    result = subprocess.run(
        ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip()) / 1024


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def mps_memory_mb(torch: Any, which: str) -> float | None:
    if not torch.backends.mps.is_available():
        return None
    fn = getattr(torch.mps, which, None)
    if fn is None:
        return None
    return float(fn()) / (1024 * 1024)


def worker(model_path: Path, images_dir: Path, result_path: Path, confidence: float) -> dict[str, Any]:
    import cv2
    import torch
    from ultralytics import YOLO

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rss_after_import = current_rss_mb()
    model = YOLO(str(model_path))
    rss_after_load = current_rss_mb()

    def sync() -> None:
        if device == "mps":
            torch.mps.synchronize()

    def predict(path: Path) -> int:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Cannot read {path}")
        results = model.predict(
            frame,
            imgsz=640,
            conf=confidence,
            classes=[0],
            device=device,
            verbose=False,
        )
        return len(results[0].boxes)

    for path in image_paths[:5]:
        predict(path)
    sync()
    rss_after_warmup = current_rss_mb()
    mps_after_warmup = mps_memory_mb(torch, "current_allocated_memory")
    mps_driver_after_warmup = mps_memory_mb(torch, "driver_allocated_memory")

    wall_seconds = 0.0
    cpu_seconds = 0.0
    detections = 0
    positive_frames = 0
    positive_frames_with_detection = 0
    negative_frames = 0
    negative_frames_with_detection = 0
    labels_dir = images_dir.parent.parent / "labels" / images_dir.name
    for path in image_paths:
        sync()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        frame_detections = predict(path)
        detections += frame_detections
        label_path = labels_dir / f"{path.stem}.txt"
        is_positive = label_path.exists() and bool(label_path.read_text(encoding="utf-8").strip())
        if is_positive:
            positive_frames += 1
            positive_frames_with_detection += frame_detections > 0
        else:
            negative_frames += 1
            negative_frames_with_detection += frame_detections > 0
        sync()
        wall_seconds += time.perf_counter() - wall_start
        cpu_seconds += time.process_time() - cpu_start

    result = {
        "model": model_path.parent.parent.name,
        "device": device,
        "frames": len(image_paths),
        "confidence_threshold": confidence,
        "detections_at_threshold": detections,
        "positive_frames": positive_frames,
        "positive_frames_with_detection": positive_frames_with_detection,
        "negative_frames": negative_frames,
        "negative_frames_with_detection": negative_frames_with_detection,
        "mean_inference_ms": wall_seconds / len(image_paths) * 1000,
        "cpu_seconds_total": cpu_seconds,
        "cpu_seconds_per_frame": cpu_seconds / len(image_paths),
        "mean_cpu_percent_of_one_core": cpu_seconds / wall_seconds * 100 if wall_seconds else 0,
        "rss_after_import_mb": rss_after_import,
        "rss_after_load_mb": rss_after_load,
        "rss_after_warmup_mb": rss_after_warmup,
        "peak_rss_mb": peak_rss_mb(),
        "mps_allocated_after_warmup_mb": mps_after_warmup,
        "mps_driver_allocated_after_warmup_mb": mps_driver_after_warmup,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--images", type=Path, default=Path("data/bottle_checkered/images/test"))
    parser.add_argument("--result", type=Path)
    parser.add_argument("--conf", type=float, default=0.80, help="App bottle confidence threshold")
    args = parser.parse_args()

    if args.worker:
        if args.model is None or args.result is None:
            parser.error("--worker requires --model and --result")
        worker(args.model, args.images, args.result, args.conf)
        return 0

    model_paths = [
        Path("models/checkered_yolov8n_batch4/weights/best.pt"),
        Path("models/checkered_yolo26n_batch4/weights/best.pt"),
    ]
    results = []
    result_dir = (
        ROOT
        / "runs"
        / "detect"
        / "checkered_bottle_comparison"
        / f"resource_profile_conf_{args.conf:.2f}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    for model_path in model_paths:
        if not model_path.exists():
            raise SystemExit(f"Missing trained model: {model_path}")
        result_path = result_dir / f"{model_path.parent.parent.name}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--model",
            str(model_path.resolve()),
            "--images",
            str(args.images.resolve()),
            "--result",
            str(result_path),
            "--conf",
            str(args.conf),
        ]
        print(f"\nProfiling {model_path.parent.parent.name} in an isolated process...", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        results.append(json.loads(result_path.read_text(encoding="utf-8")))

    print("\nSummary: model | bottle frames hit | negative frames hit | mean ms | CPU % core | peak RSS MB | MPS MB")
    for result in results:
        print(
            f"{result['model']} | {result['positive_frames_with_detection']}/{result['positive_frames']} | "
            f"{result['negative_frames_with_detection']}/{result['negative_frames']} | "
            f"{result['mean_inference_ms']:.2f} | {result['mean_cpu_percent_of_one_core']:.1f}% | "
            f"{result['peak_rss_mb']:.1f} | {result['mps_allocated_after_warmup_mb'] or 0:.1f}"
        )
    print(f"Results saved in {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
