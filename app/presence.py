from __future__ import annotations

from collections import deque


class PresenceTracker:
    """Smooths raw person/face detections into stable presence signals."""

    def __init__(
        self,
        required_present_frames: int = 3,
        history_size: int = 5,
        absent_after_seconds: float = 120.0,
    ) -> None:
        self.required_present_frames = required_present_frames
        self.history_size = history_size
        self.absent_after_seconds = absent_after_seconds
        self._history: deque[bool] = deque(maxlen=history_size)
        self._missing_since: float | None = None
        self._stable_present = True

    def update(self, raw_present: bool, now: float) -> bool:
        self._history.append(raw_present)

        if raw_present:
            self._missing_since = None
        elif self._missing_since is None:
            self._missing_since = now

        present_votes = sum(1 for v in self._history if v)
        history_signals_present = present_votes >= self.required_present_frames
        timed_out_absent = (
            self._missing_since is not None and (now - self._missing_since) >= self.absent_after_seconds
        )

        if timed_out_absent:
            self._stable_present = False
        elif history_signals_present:
            self._stable_present = True

        return self._stable_present
