#!/usr/bin/env python3
"""Profile the person and bottle models in separate processes on identical frames."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def _current_rss_mb() -> float:
    result = subprocess.run(
        ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip()) / 1024


def _capture_frames(frame_dir: Path, camera_index: int, duration: float, fps: int) -> list[Path]:
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    frame_paths: list[Path] = []
    interval = 1.0 / max(1, fps)
    deadline = time.monotonic() + duration
    next_capture = time.monotonic()
    print(f"Capturing for {duration:g}s at up to {fps} FPS. Perform the drinking action now.", flush=True)
    try:
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.monotonic()
            if now >= next_capture:
                path = frame_dir / f"frame_{len(frame_paths):05d}.jpg"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Could not save captured frame to {path}")
                frame_paths.append(path)
                next_capture = now + interval
    finally:
        cap.release()

    if len(frame_paths) < 6:
        raise RuntimeError(f"Captured only {len(frame_paths)} frames; need at least 6")
    print(f"Captured {len(frame_paths)} frames. Profiling each model in a separate process.", flush=True)
    return frame_paths


def _run_worker(model_name: str, frame_paths: list[Path]) -> dict[str, Any]:
    import cv2
    import torch
    from ultralytics import YOLO

    from app.config import load_config

    config = load_config()
    if model_name == "person":
        model_path = config.person_model_path or config.model_path
        confidence = config.object_confidence
        classes = [0]
    else:
        model_path = config.bottle_model_path
        confidence = config.bottle_confidence
        classes = [config.bottle_class_id]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rss_after_import_mb = _current_rss_mb()
    model = YOLO(model_path)
    rss_after_load_mb = _current_rss_mb()

    def sync_device() -> None:
        if device == "mps":
            torch.mps.synchronize()

    def predict(frame: Any) -> None:
        model.predict(
            frame,
            verbose=False,
            device=device,
            conf=confidence,
            classes=classes,
        )

    warmup_paths = frame_paths[:5]
    for path in warmup_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read captured frame {path}")
        predict(frame)
    sync_device()
    rss_after_warmup_mb = _current_rss_mb()

    measured_paths = frame_paths[5:] or frame_paths
    cpu_seconds = 0.0
    wall_seconds = 0.0
    for path in measured_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read captured frame {path}")
        sync_device()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        predict(frame)
        sync_device()
        wall_seconds += time.perf_counter() - wall_start
        cpu_seconds += time.process_time() - cpu_start

    rss_after_measurement_mb = _current_rss_mb()
    result = {
        "model": model_name,
        "pid": __import__("os").getpid(),
        "device": device,
        "measured_inferences": len(measured_paths),
        "mean_inference_ms": wall_seconds / len(measured_paths) * 1000,
        "wall_seconds_total": wall_seconds,
        "cpu_seconds_total": cpu_seconds,
        "cpu_seconds_per_inference": cpu_seconds / len(measured_paths),
        "mean_cpu_percent_of_one_core": cpu_seconds / wall_seconds * 100 if wall_seconds else 0,
        "rss_after_import_mb": rss_after_import_mb,
        "rss_after_model_load_mb": rss_after_load_mb,
        "rss_after_warmup_mb": rss_after_warmup_mb,
        "rss_after_measurement_mb": rss_after_measurement_mb,
        "estimated_model_load_rss_increase_mb": max(0.0, rss_after_load_mb - rss_after_import_mb),
        "peak_rss_mb": _peak_rss_mb(),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=30)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--worker", choices=("person", "bottle"))
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--result-path", type=Path)
    args = parser.parse_args()

    if args.worker:
        if args.frames_dir is None:
            parser.error("--worker requires --frames-dir")
        frame_paths = sorted(args.frames_dir.glob("frame_*.jpg"))
        result = _run_worker(args.worker, frame_paths)
        if args.result_path:
            args.result_path.write_text(json.dumps(result), encoding="utf-8")
        return 0

    import cv2  # noqa: F401 - validate OpenCV before opening the camera

    with tempfile.TemporaryDirectory(prefix="jallijiye-profile-") as temp_dir:
        frame_dir = Path(temp_dir)
        frame_paths = _capture_frames(frame_dir, args.camera, args.duration_seconds, args.fps)
        results: list[dict[str, Any]] = []
        for model_name in ("person", "bottle"):
            result_path = frame_dir / f"{model_name}_result.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                model_name,
                "--frames-dir",
                str(frame_dir),
                "--result-path",
                str(result_path),
            ]
            print(f"\nProfiling {model_name} model...", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            results.append(json.loads(result_path.read_text(encoding="utf-8")))

        cpu_total = sum(result["cpu_seconds_total"] for result in results)
        wall_total = sum(result["wall_seconds_total"] for result in results)
        model_rss_total = sum(result["estimated_model_load_rss_increase_mb"] for result in results)
        print("\nResource comparison (isolated model processes)")
        print("model | device | mean inference | mean CPU / inference | mean CPU % | RSS after load | peak RSS")
        for result in results:
            print(
                f"{result['model']} | {result['device']} | {result['mean_inference_ms']:.1f} ms | "
                f"{result['cpu_seconds_per_inference'] * 1000:.1f} ms | "
                f"{result['mean_cpu_percent_of_one_core']:.1f}% | "
                f"{result['rss_after_model_load_mb']:.1f} MB | {result['peak_rss_mb']:.1f} MB"
            )
        print("\nShare of measured model work")
        for result in results:
            cpu_share = result["cpu_seconds_total"] / cpu_total * 100 if cpu_total else 0.0
            wall_share = result["wall_seconds_total"] / wall_total * 100 if wall_total else 0.0
            memory_share = (
                result["estimated_model_load_rss_increase_mb"] / model_rss_total * 100
                if model_rss_total
                else 0.0
            )
            print(
                f"{result['model']}: CPU time {cpu_share:.1f}%, inference time {wall_share:.1f}%, "
                f"estimated added RSS {memory_share:.1f}%"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
