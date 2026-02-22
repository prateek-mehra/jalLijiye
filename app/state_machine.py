from __future__ import annotations

import time

from app.types import AppMode, AppState, Config, DrinkEvent


class HydrationStateMachine:
    def __init__(self, config: Config, initial_now: float | None = None) -> None:
        now = initial_now if initial_now is not None else time.time()
        self.config = config
        self.mode: AppMode = "TRACKING"
        self.is_present: bool = True
        self.last_drink_ts: float = now
        self.alert_started_at: float | None = None
        self.manual_pause_until: float | None = None

    def set_presence(self, present: bool, now: float | None = None) -> AppState:
        now = now if now is not None else time.time()
        self.is_present = present
        if not present:
            self.mode = "PAUSED_ABSENT"
        elif self.manual_pause_until is None and self.mode == "PAUSED_ABSENT":
            self.mode = "TRACKING"
        return self.snapshot(now)

    def pause_for(self, minutes: float, now: float | None = None) -> AppState:
        now = now if now is not None else time.time()
        self.manual_pause_until = now + (minutes * 60)
        self.mode = "PAUSED_ABSENT"
        return self.snapshot(now)

    def resume(self, now: float | None = None) -> AppState:
        now = now if now is not None else time.time()
        self.manual_pause_until = None
        if self.is_present:
            self.mode = "TRACKING"
        else:
            self.mode = "PAUSED_ABSENT"
        return self.snapshot(now)

    def mark_drink(self, event: DrinkEvent, now: float | None = None) -> AppState:
        now = now if now is not None else event.timestamp
        self.last_drink_ts = event.timestamp
        self.alert_started_at = None
        if self.is_present:
            self.mode = "TRACKING"
        else:
            self.mode = "PAUSED_ABSENT"
        return self.snapshot(now)

    def tick(self, now: float | None = None) -> AppState:
        now = now if now is not None else time.time()

        if self.manual_pause_until is not None and now >= self.manual_pause_until:
            self.manual_pause_until = None
            if self.is_present:
                self.mode = "TRACKING"

        if not self.is_present:
            self.mode = "PAUSED_ABSENT"
            self.alert_started_at = None
            return self.snapshot(now)

        if self.manual_pause_until is not None and now < self.manual_pause_until:
            self.mode = "PAUSED_ABSENT"
            self.alert_started_at = None
            return self.snapshot(now)

        elapsed = now - self.last_drink_ts
        alert_after = self.config.alert_after_minutes * 60
        escalating_window = self.config.escalating_minutes * 60

        if elapsed < alert_after:
            self.mode = "TRACKING"
            self.alert_started_at = None
            return self.snapshot(now)

        if self.mode in ("TRACKING", "PAUSED_ABSENT"):
            self.mode = "ALERT_ESCALATING"
            self.alert_started_at = now
        elif self.mode == "ALERT_ESCALATING" and self.alert_started_at is not None:
            if now - self.alert_started_at >= escalating_window:
                self.mode = "ALERT_CONTINUOUS"

        return self.snapshot(now)

    def snapshot(self, now: float | None = None) -> AppState:
        now = now if now is not None else time.time()
        minutes_since_drink = max(0.0, (now - self.last_drink_ts) / 60.0)

        if not self.is_present:
            detail = "Away"
        elif self.manual_pause_until is not None and now < self.manual_pause_until:
            detail = "Manual pause"
        elif self.mode == "ALERT_ESCALATING":
            detail = "Drink water soon"
        elif self.mode == "ALERT_CONTINUOUS":
            detail = "Drink water now"
        else:
            detail = ""

        return AppState(
            mode=self.mode,
            minutes_since_drink=minutes_since_drink,
            is_present=self.is_present,
            status_detail=detail,
        )
