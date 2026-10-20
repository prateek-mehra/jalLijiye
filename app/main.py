from __future__ import annotations

import logging
import math
import os
import subprocess
import signal
import threading
import time

from app.alerter import Alerter
from app.config import load_config
from app.menu_bar import JalLijiyeMenuBar
from app.paths import ensure_logs_dir
from app.reminder_scheduler import ReminderScheduler
from app.types import Box, DetectionFrame
from app.wake import SystemWakeObserver


def is_reliable_person_presence(
    frame: DetectionFrame,
    min_area_ratio: float,
    center_margin: float,
) -> bool:
    """
    Treat person detection as presence only when the box is large and near center.
    This filters small/background false positives that keep mode in TRACKING.
    """
    if not frame.person_boxes or frame.frame is None:
        return False

    shape = getattr(frame.frame, "shape", None)
    if not shape or len(shape) < 2:
        return False

    frame_h = float(shape[0])
    frame_w = float(shape[1])
    frame_area = frame_w * frame_h
    if frame_area <= 0:
        return False

    best: Box = max(
        frame.person_boxes,
        key=lambda b: max(0.0, (b.x2 - b.x1) * (b.y2 - b.y1)),
    )
    box_area = max(0.0, (best.x2 - best.x1) * (best.y2 - best.y1))
    if box_area < (frame_area * min_area_ratio):
        return False

    cx, cy = best.center()
    x_margin_px = frame_w * center_margin
    y_margin_top_px = frame_h * 0.05
    y_margin_bottom_px = frame_h * 0.05
    return (
        x_margin_px <= cx <= (frame_w - x_margin_px)
        and y_margin_top_px <= cy <= (frame_h - y_margin_bottom_px)
    )


class JalLijiyeController:
    def __init__(self) -> None:
        from app.detector import BottleOnlyDetector

        self.config = load_config()
        self.scheduler = ReminderScheduler(self.config, started_at=time.monotonic())
        self.detector = BottleOnlyDetector(self.config)
        self.alerter = Alerter()

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._detector_thread = threading.Thread(target=self._detector_loop, daemon=True)
        self._active_alert_started_at: float | None = None

        self.detector_status = "Starting"
        self.hydration_count = 0

    def start(self) -> None:
        log_dir = ensure_logs_dir()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(log_dir / "events.log"), logging.StreamHandler()],
        )

        logging.info("JalLijiye starting")

        self.alerter.start()
        self.detector_status = "Waiting for scheduled bottle check"
        self._detector_thread.start()
        logging.info(
            "Scheduled bottle checks enabled: interval=%.1f min, window=%.1f sec, "
            "snooze=%.1f min, detector=%d fps, threshold=%d/%d frames",
            self.config.reminder_interval_minutes,
            self.config.detection_window_seconds,
            self.config.detection_snooze_minutes,
            self.config.fps,
            self.scheduler.required_hits,
            self.scheduler.window_frames,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._detector_thread.is_alive():
            self._detector_thread.join(timeout=2.0)
        self.alerter.stop()
        self.detector.close()
        logging.info("JalLijiye stopped")

    def quit_app(self) -> None:
        self._stop_launch_agent_for_session()
        self.stop()

    def mark_manual_drink(self) -> None:
        with self._lock:
            self.scheduler.mark_completed(now=time.monotonic())
            self.hydration_count += 1
            self._active_alert_started_at = None
        logging.info("Drink event: source=manual")

    def pause_30_minutes(self) -> None:
        with self._lock:
            self.scheduler.pause_for(30, now=time.monotonic())
            self._active_alert_started_at = None
        logging.info("Manual pause enabled for 30 minutes")

    def on_system_wake(self) -> None:
        with self._lock:
            self.scheduler.start_immediately(now=time.monotonic())
            self._active_alert_started_at = None
        logging.info("System woke from sleep; starting a bottle check immediately")

    def get_status(self) -> tuple[str, str, bool]:
        # rumps calls this from its UI timer on the main thread. OpenCV's
        # macOS window must also be created and pumped from the main thread.
        self.detector.pump_debug_window()
        with self._lock:
            snapshot = self.scheduler.snapshot(now=time.monotonic())
            hydration_count = self.hydration_count
            hydration_goal = self.config.daily_hydration_goal
            alert_started_at = self._active_alert_started_at

        if snapshot.mode in ("MONITORING", "SNOOZED"):
            title = "Drink water"
        else:
            remaining_seconds = snapshot.seconds_until_next_action
            remaining_minutes = math.ceil(remaining_seconds / 60.0)
            if remaining_seconds <= 0:
                title = "Next check starting"
            elif remaining_seconds < 60:
                title = "Next check in: <1m"
            else:
                title = f"Next check in: {remaining_minutes}m"

        alert_mode = "ALERT_ESCALATING" if snapshot.mode == "MONITORING" else "TRACKING"
        self.alerter.set_mode(
            alert_mode,
            alert_started_at if snapshot.mode == "MONITORING" else None,
        )

        progress_title = f"Hydration: {hydration_count}/{hydration_goal}"
        show_menu_bar = snapshot.mode in ("MONITORING", "SNOOZED")
        return title, progress_title, show_menu_bar

    def _detector_loop(self) -> None:
        detector_active = False
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                transition = self.scheduler.advance(now)
                mode = self.scheduler.mode

            if transition == "start":
                self.detector.start()
                detector_active = True
                with self._lock:
                    self._active_alert_started_at = time.time()
                    self.detector_status = self.detector.status.message
                logging.info("Bottle check window started")
                continue

            if transition == "timeout":
                self.detector.close()
                detector_active = False
                with self._lock:
                    self._active_alert_started_at = None
                logging.info(
                    "Bottle threshold not met in check window; snoozing for %.1f minutes",
                    self.config.detection_snooze_minutes,
                )
                continue

            if mode == "MONITORING":
                detected = self.detector.read_bottle_detection()
                if detected is not None:
                    with self._lock:
                        result = self.scheduler.observe_bottle(detected, time.monotonic())
                        if result == "success":
                            self.hydration_count += 1
                            self._active_alert_started_at = None
                        mode = self.scheduler.mode

                    if result == "success":
                        self.detector.close()
                        detector_active = False
                        logging.info("Bottle detection threshold met; next check scheduled")

                if mode != "MONITORING" and detector_active:
                    self.detector.close()
                    detector_active = False
                time.sleep(0.01 if self.detector.status.available else 0.05)
                continue

            if detector_active:
                self.detector.close()
                detector_active = False
            with self._lock:
                self._active_alert_started_at = None
            time.sleep(0.2)

    def _stop_launch_agent_for_session(self) -> None:
        label = os.environ.get("JAL_LAUNCH_LABEL", "com.prateek.jallijiye")
        uid = os.getuid()
        commands = [
            ["launchctl", "bootout", f"gui/{uid}/{label}"],
            ["pkill", "-f", "run_jallijiye.sh"],
        ]
        for command in commands:
            try:
                subprocess.run(command, check=False, capture_output=True)
            except Exception:
                continue



def run() -> None:
    controller = JalLijiyeController()
    controller.start()
    wake_observer = SystemWakeObserver(controller.on_system_wake)

    def _shutdown(*_: object) -> None:
        controller.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    menu = JalLijiyeMenuBar(
        get_status=controller.get_status,
        on_mark_drink=controller.mark_manual_drink,
        on_pause=controller.pause_30_minutes,
        on_quit=controller.quit_app,
    )
    try:
        menu.run()
    finally:
        wake_observer.close()


if __name__ == "__main__":
    run()
