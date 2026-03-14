from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "JalLijiye"


def resource_root() -> Path:
    executable = Path(sys.executable).resolve()
    if executable.parent.name == "MacOS" and executable.parent.parent.name == "Contents":
        return executable.parent.parent / "Resources"
    return Path(__file__).resolve().parent.parent


def resolve_resource_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return resource_root() / candidate


def logs_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_NAME
    return resource_root() / "logs"


def ensure_logs_dir() -> Path:
    path = logs_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def in_app_bundle() -> bool:
    executable = Path(sys.executable).resolve()
    return executable.parent.name == "MacOS" and executable.parent.parent.name == "Contents"


def app_bundle_path() -> Path | None:
    executable = Path(sys.executable).resolve()
    if executable.parent.name == "MacOS" and executable.parent.parent.name == "Contents":
        return executable.parent.parent.parent
    return None
