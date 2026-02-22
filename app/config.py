from __future__ import annotations

from pathlib import Path
from typing import Any

from app.types import Config

try:
    import yaml
except Exception:  # pragma: no cover - optional runtime dependency
    yaml = None


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"


def _coerce_number(raw: Any, cast: type, default: Any) -> Any:
    if raw is None:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return Config()

    if yaml is not None:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = _parse_simple_yaml(cfg_path)

    return Config(
        alert_after_minutes=_coerce_number(data.get("alert_after_minutes"), int, 90),
        absence_pause_minutes=_coerce_number(data.get("absence_pause_minutes"), int, 2),
        fps=_coerce_number(data.get("fps"), int, 5),
        object_confidence=_coerce_number(data.get("object_confidence"), float, 0.45),
        mouth_expand_ratio=_coerce_number(data.get("mouth_expand_ratio"), float, 0.15),
        drink_hold_seconds=_coerce_number(data.get("drink_hold_seconds"), float, 2.0),
        drink_window_seconds=_coerce_number(data.get("drink_window_seconds"), float, 5.0),
        drink_cooldown_minutes=_coerce_number(data.get("drink_cooldown_minutes"), float, 10.0),
        escalating_minutes=_coerce_number(data.get("escalating_minutes"), float, 3.0),
        model_path=str(data.get("model_path") or "yolov8n.pt"),
    )


def _parse_simple_yaml(path: Path) -> dict[str, str]:
    """
    Minimal parser for flat key: value YAML files.
    Keeps app bootable even if PyYAML is missing.
    """
    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data
