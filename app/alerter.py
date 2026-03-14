from __future__ import annotations

import subprocess
import threading
import time

from app.types import AppMode


class Alerter:
    def __init__(self, sound_path: str = "/System/Library/Sounds/Bottle.aiff") -> None:
        self.sound_path = sound_path
        self._mode: AppMode | None = None
        self._alert_started_at: float | None = None
        self._last_escalating_beep = 0.0
        self._last_continuous_beep = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._stop.clear()
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def set_mode(self, mode: AppMode, alert_started_at: float | None) -> None:
        with self._lock:
            self._mode = mode
            self._alert_started_at = alert_started_at

    def _beep_once(self) -> None:
        try:
            subprocess.run(
                ["afplay", self.sound_path],
                check=False,
                capture_output=True,
                timeout=2.0,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            print("\a", end="", flush=True)

    def _worker(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            with self._lock:
                mode = self._mode
                alert_started_at = self._alert_started_at

            if mode == "ALERT_ESCALATING":
                if alert_started_at is None:
                    alert_started_at = now
                in_escalation_window = (now - alert_started_at) <= 180
                if in_escalation_window and (now - self._last_escalating_beep) >= 30:
                    self._beep_once()
                    self._last_escalating_beep = now
                time.sleep(0.2)
                continue

            if mode == "ALERT_CONTINUOUS":
                if (now - self._last_continuous_beep) >= 1.2:
                    self._beep_once()
                    self._last_continuous_beep = now
                time.sleep(0.1)
                continue

            time.sleep(0.25)
