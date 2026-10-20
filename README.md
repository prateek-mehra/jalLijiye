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

## Resource Optimizations

Activity Monitor observations on 2026-10-20 while camera inference was active on an Apple M1 MacBook Air (macOS 26.3, Python 3.12.6):

| Inference backend | Debug window | CPU (% of one core) | GPU | Real memory | Measurement notes |
| --- | --- | ---: | ---: | ---: | --- |
| CPU | On (`show_debug_window: true`) | 138–149% | Not recorded | 214–218 MB | Point-in-time samples during a drinking-action attempt. |
| CPU | Off (`show_debug_window: false`) | 131–136% | Not recorded | 244–306 MB | Two early samples about 20 seconds apart; memory fell between them and had not settled. |
| MPS | On (`show_debug_window: true`) | 69.7% | 21.6% | 190.3 MB | One Activity Monitor sample during an active camera inference run (Python PID 58140). |

These are preliminary samples from separate runs, not controlled averages or peaks. They do not establish a memory reduction from disabling the debug window or a controlled performance improvement from MPS. CPU readings above 100% mean the process used more than one core. Activity Monitor did not confirm whether the drinking event was detected.

### Isolated per-model profile

On 2026-10-20, the person and bottle models were profiled in separate MPS processes using the same 137 camera frames. Each model had 5 warm-up inferences followed by 132 measured inferences. CPU and inference percentages below are each model's share of the two isolated runs.

| Model | Device | Mean inference time | CPU time per inference | Mean CPU (% of one core) | CPU time share | Inference time share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Person (`yolov8n.pt`) | MPS | 16.5 ms | 8.5 ms | 51.3% | 52.8% | 56.2% |
| Bottle (custom model) | MPS | 12.9 ms | 7.6 ms | 59.0% | 47.2% | 43.8% |

| Model | RSS after imports | RSS after model load | RSS after warm-up | RSS after measurement | Estimated model-load RSS increase | Peak process RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Person (`yolov8n.pt`) | 309.5 MB | 321.1 MB | 117.3 MB | 99.8 MB | 11.6 MB | 429.8 MB |
| Bottle (custom model) | 316.4 MB | 327.5 MB | 409.0 MB | 378.3 MB | 11.2 MB | 457.3 MB |

These are isolated model runs rather than the full app. CPU time is measured around synchronized inference calls; mean CPU is process CPU time divided by inference wall time, expressed as a percentage of one core. RAM includes Python, PyTorch, Ultralytics, and model allocations. The estimated model-load RSS increase is the difference between current RSS after imports and after loading the model, so it is approximate. Current-RSS snapshots varied substantially across checkpoints; peak process RSS is the high-water mark and is more useful for comparing the two runs. The results show a near-even CPU-time split, while the bottle run had a higher peak process RSS.
