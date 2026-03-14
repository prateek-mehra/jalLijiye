# JalLijiye

Hydration reminder app for macOS that runs as a menu bar process, detects drinking events using on-device computer vision, and alerts when the no-drink threshold is crossed.

## Features

- Menu bar app with status and quick actions.
- Hybrid trigger model:
  - Vision-based drink detection (`person/face + bottle near mouth`).
  - Manual fallback (`Mark Drink Now`).
- Presence-aware timer:
  - Auto-pauses when you are away.
  - Resumes when you return.
- Escalating alerts:
  - Beep every 30s in escalation window.
  - Continuous beeping after 3 minutes unresolved.
- Live debug stream showing:
  - Person/face/mouth/bottle boxes.
  - Bottle source (`custom` vs `none`).
  - Drink-decision telemetry (`contact_seconds`, `drink_detected`, cooldown).
- Custom bottle model support for your bottle.

## Setup

```bash
cd /Users/prateek/Downloads/_Projects/Personal/codex/jalLijiye
python3 -m venv .venv
source .venv/bin/activate
pip install --no-build-isolation -e '.[dev]'
```

## Run

```bash
cd /Users/prateek/Downloads/_Projects/Personal/codex/jalLijiye
source .venv/bin/activate
python -m app.main
```

Grant Camera permission on first launch.

## Build macOS App

Build a proper `.app` bundle for menu bar startup on macOS:

```bash
cd /Users/prateek/Downloads/_Projects/Personal/codex/jalLijiye
chmod +x scripts/build_macos_app.sh
./scripts/build_macos_app.sh
```

Expected output:
- `dist/JalLijiye.app`

## Config

Edit `/Users/prateek/Downloads/_Projects/Personal/codex/jalLijiye/config/defaults.yaml`.

Key values:

- `alert_after_minutes: 1` (testing profile)
- `presence_absent_after_seconds: 10.0`
- `presence_person_min_area_ratio: 0.06`
- `presence_person_center_margin: 0.20`
- `person_model_path: yolov8n.pt`
- `bottle_model_path: models/bottle_v3/weights/best.pt`
- `bottle_class_id: 0`
- `bottle_confidence: 0.80`
- `mouth_expand_ratio: 0.15`
- `mouth_memory_seconds: 2.5`
- `drink_hold_seconds: 1.0`
- `drink_cooldown_minutes: 0.0833` (5 seconds, testing)
- `show_debug_window: true`

## Fast Bottle Training Workflow

### 1) Collect videos

Put your webcam videos in:
- `data/raw_videos/`

### 2) Extract frames

```bash
source .venv/bin/activate
python scripts/extract_frames.py --input-dir data/raw_videos --output-dir data/bottle/images/raw --sample-every-seconds 1.0
```

### 3) Annotate

- Annotate only bottle boxes.
- Save YOLO `.txt` labels in `data/bottle/labels/raw/`.
- Class id mapping: `0 -> my_bottle`.

### 4) Split dataset

```bash
source .venv/bin/activate
python scripts/split_dataset.py --images-raw data/bottle/images/raw --labels-raw data/bottle/labels/raw --out-root data/bottle --train-ratio 0.8
```

### 5) Train + validate

```bash
bash scripts/train_bottle_model.sh
```

Expected output model:
- `models/bottle_v1/weights/best.pt`

## Tests

```bash
source .venv/bin/activate
pytest -q
```

## Launch at Login (Recommended)

Use the macOS autostart script.

Preferred flow:
- build `dist/JalLijiye.app`
- autostart launches the app bundle at login

Fallback flow:
- if the app bundle does not exist yet, the script falls back to the NoSlouchBench-style hidden Terminal runner

Install and start:

```bash
cd /Users/prateek/Downloads/_Projects/Personal/codex/jalLijiye
chmod +x scripts/autostart_macos.sh
./scripts/autostart_macos.sh install
```

Check status:

```bash
./scripts/autostart_macos.sh status
```

Stop:

```bash
./scripts/autostart_macos.sh stop
```

Uninstall:

```bash
./scripts/autostart_macos.sh uninstall
```
