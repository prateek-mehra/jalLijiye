from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from math import ceil

from app.types import Config


ReminderMode = str  # WAITING, MONITORING, or SNOOZED

# Week numbering and team rotations from the user's schedule. ISO Week 43 of
# 2026 is Mod Week 5; calculate from that anchor to keep the cycle continuous
# across ISO years, including years with a Week 53.
SCHEDULE_ANCHOR = date.fromisocalendar(2026, 43, 1)
SCHEDULE_ANCHOR_MOD_WEEK = 5
WFH_DAYS_BY_MOD_WEEK = {
    1: {1, 3},       # Tuesday, Thursday
    2: {0, 2, 4},    # Monday, Wednesday, Friday
    3: {1, 3},
    4: {1},          # Tuesday only
    5: {0, 2, 4},
    6: {0, 2, 4},
}


def is_wfh_day(day: datetime) -> bool:
    """Return whether the schedule says the given local date is WFH."""
    week_monday = day.date() - timedelta(days=day.weekday())
    week_offset = (week_monday - SCHEDULE_ANCHOR).days // 7
    mod_week = (SCHEDULE_ANCHOR_MOD_WEEK - 1 + week_offset) % 6 + 1
    return day.weekday() in WFH_DAYS_BY_MOD_WEEK[mod_week]


def seconds_until_next_wfh_day(day: datetime) -> float:
    """Seconds until midnight at the start of the next scheduled WFH day."""
    next_midnight = datetime.combine(day.date() + timedelta(days=1), time.min, tzinfo=day.tzinfo)
    for offset in range(7):
        candidate = next_midnight + timedelta(days=offset)
        if is_wfh_day(candidate):
            return max(0.0, (candidate - day).total_seconds())
    return 7 * 24 * 60 * 60.0


@dataclass(frozen=True, slots=True)
class ReminderSnapshot:
    mode: ReminderMode
    seconds_until_next_action: float
    bottle_hits_in_window: int
    window_frames_seen: int


class ReminderScheduler:
    """Schedules short bottle-detection sessions between idle intervals."""

    def __init__(self, config: Config, started_at: float) -> None:
        self.config = config
        self.interval_seconds = max(0.0, config.reminder_interval_minutes * 60.0)
        self.window_seconds = max(1.0, config.detection_window_seconds)
        self.snooze_seconds = max(0.0, config.detection_snooze_minutes * 60.0)
        self.window_frames = max(1, config.bottle_detection_window_frames)
        threshold = min(1.0, max(0.0, config.bottle_detection_threshold))
        self.required_hits = max(1, ceil(self.window_frames * threshold))

        self.mode: ReminderMode = "WAITING"
        self.next_action_at = started_at + self.interval_seconds
        self.monitor_deadline: float | None = None
        self._detections: deque[bool] = deque(maxlen=self.window_frames)
        self.last_success_at: float | None = None

    def advance(self, now: float, wall_now: datetime | None = None) -> str | None:
        wall_now = wall_now or datetime.now().astimezone()
        if not is_wfh_day(wall_now):
            was_monitoring = self.mode == "MONITORING"
            if was_monitoring:
                self._detections.clear()
                self.monitor_deadline = None
            self.mode = "WAITING"
            self.next_action_at = now + seconds_until_next_wfh_day(wall_now)
            return "timeout" if was_monitoring else None

        if self.mode == "MONITORING":
            if self.monitor_deadline is not None and now >= self.monitor_deadline:
                self.mode = "SNOOZED"
                self.next_action_at = now + self.snooze_seconds
                self.monitor_deadline = None
                self._detections.clear()
                return "timeout"
            return None

        if self.mode in ("WAITING", "SNOOZED") and now >= self.next_action_at:
            self.mode = "MONITORING"
            self.monitor_deadline = now + self.window_seconds
            self._detections.clear()
            return "start"

        return None

    def observe_bottle(self, detected: bool, now: float) -> str | None:
        if self.mode != "MONITORING":
            return None

        self._detections.append(detected)
        if (
            len(self._detections) == self.window_frames
            and sum(self._detections) >= self.required_hits
        ):
            self.mode = "WAITING"
            self.next_action_at = now + self.interval_seconds
            self.monitor_deadline = None
            self.last_success_at = now
            self._detections.clear()
            return "success"

        return None

    def mark_completed(self, now: float) -> None:
        """Treat a manual confirmation as success and begin the normal interval."""
        self.mode = "WAITING"
        self.next_action_at = now + self.interval_seconds
        self.monitor_deadline = None
        self._detections.clear()
        self.last_success_at = now

    def pause_for(self, minutes: float, now: float) -> None:
        self.mode = "WAITING"
        self.next_action_at = now + max(0.0, minutes * 60.0)
        self.monitor_deadline = None
        self._detections.clear()

    def start_immediately(self, now: float) -> None:
        """Schedule a fresh bottle-check window immediately, such as after wake."""
        self.mode = "WAITING"
        self.next_action_at = now
        self.monitor_deadline = None
        self._detections.clear()

    def snapshot(self, now: float) -> ReminderSnapshot:
        deadline = self.monitor_deadline if self.mode == "MONITORING" else self.next_action_at
        return ReminderSnapshot(
            mode=self.mode,
            seconds_until_next_action=max(0.0, (deadline or now) - now),
            bottle_hits_in_window=sum(self._detections),
            window_frames_seen=len(self._detections),
        )
