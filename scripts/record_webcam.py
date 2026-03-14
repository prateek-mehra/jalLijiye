#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record webcam video clips for bottle dataset collection.")
    parser.add_argument(
        "--output-dir",
        default="data/raw_video",
        help="Directory where videos will be saved",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=180,
        help="Duration of recording in seconds",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for OpenCV VideoCapture",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Target output FPS",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Target frame width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Target frame height",
    )
    parser.add_argument(
        "--prefix",
        default="session",
        help="Filename prefix",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview window during recording (press q to stop)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        raise SystemExit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{args.prefix}_{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        cap.release()
        print("[ERROR] Could not open output writer.")
        raise SystemExit(1)

    start = time.time()
    frames = 0
    max_duration = max(1, args.seconds)
    print(f"[INFO] Recording for up to {max_duration}s -> {out_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Camera read failed; stopping.")
            break

        frame_resized = cv2.resize(frame, (args.width, args.height))
        writer.write(frame_resized)
        frames += 1

        elapsed = time.time() - start
        if args.preview:
            overlay = frame_resized.copy()
            cv2.putText(
                overlay,
                f"REC {elapsed:05.1f}s / {max_duration}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Recording (press q to stop)", overlay)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                print("[INFO] Stopped by user.")
                break

        if elapsed >= max_duration:
            break

    cap.release()
    writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    print(f"[DONE] Saved {frames} frames to {out_path}")


if __name__ == "__main__":
    main()
