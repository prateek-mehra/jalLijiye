# JalLijiye

Hydration reminder app for macOS that runs as a menu bar process, detects drinking events using on-device computer vision, and alerts when no drink is detected for more than 90 minutes.

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
- Privacy by default:
  - On-device inference only.
  - No frame/image/video storage.

## Setup

```bash
cd /Users/prateek/Downloads/_Projects/Personal/codex/jalLijiye
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run

```bash
python3 -m app.main
```

Grant Camera permission on first launch.

## Launch at Login (launchd)

1. Ensure logs directory exists:

```bash
mkdir -p /Users/prateek/Downloads/_Projects/Personal/codex/jalLijiye/logs
```

2. Copy plist and load agent:

```bash
cp /Users/prateek/Downloads/_Projects/Personal/codex/jalLijiye/launchd/com.prateek.jallijiye.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.prateek.jallijiye.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.prateek.jallijiye.plist
```

## Config

Edit `/Users/prateek/Downloads/_Projects/Personal/codex/jalLijiye/config/defaults.yaml`.

Default values:

- `alert_after_minutes: 90`
- `absence_pause_minutes: 2`
- `fps: 5`
- `object_confidence: 0.45`
- `drink_hold_seconds: 2.0`
- `drink_window_seconds: 5.0`
- `drink_cooldown_minutes: 10.0`

## Tests

```bash
pytest
```

## Notes

- If YOLO/OpenCV is unavailable, app stays active in manual mode and still alerts based on timer.
- v1 uses heuristic proximity detection; OBB/pose-angle logic is intentionally deferred to v2.
