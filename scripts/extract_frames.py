#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract sampled frames from videos for bottle annotation.")
    p.add_argument("--input-dir", default="data/raw_videos", help="Folder containing raw videos")
    p.add_argument("--output-dir", default="data/bottle/images/raw", help="Folder to save frames")
    p.add_argument(
        "--sample-every-seconds",
        type=float,
        default=1.0,
        help="Sampling interval in seconds",
    )
    p.add_argument(
        "--max-frames-per-video",
        type=int,
        default=80,
        help="Hard cap per video to avoid near-duplicate frames",
    )
    return p.parse_args()


def extract_from_video(
    video_path: Path,
    output_dir: Path,
    sample_every_seconds: float,
    max_frames_per_video: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] cannot open: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_step = max(1, int(round(fps * sample_every_seconds)))
    saved = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            output_name = f"{video_path.stem}_{frame_idx:06d}.jpg"
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), frame)
            saved += 1
            if saved >= max_frames_per_video:
                break

        frame_idx += 1

    cap.release()
    print(f"[INFO] {video_path.name}: saved {saved} frames")
    return saved


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] input dir not found: {input_dir}")
        raise SystemExit(1)

    videos = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS])
    if not videos:
        print(f"[ERROR] no videos found in: {input_dir}")
        raise SystemExit(1)

    total = 0
    for video in videos:
        total += extract_from_video(
            video,
            output_dir,
            sample_every_seconds=args.sample_every_seconds,
            max_frames_per_video=args.max_frames_per_video,
        )

    print(f"[DONE] saved {total} total frames to {output_dir}")


if __name__ == "__main__":
    main()
