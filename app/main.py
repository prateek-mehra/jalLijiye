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
from app.presence import PresenceTracker
from app.state_machine import HydrationStateMachine
from app.types import Box, DetectionFrame, DrinkEvent


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
        from app.detector import DrinkHeuristic, VisionDetector

        self.config = load_config()
        self.state_machine = HydrationStateMachine(self.config)
        self.presence = PresenceTracker(
            required_present_frames=3,
            history_size=5,
            absent_after_seconds=self.config.presence_absent_after_seconds,
        )
        self.detector = VisionDetector(self.config)
        self.drink_heuristic = DrinkHeuristic(self.config)
        self.alerter = Alerter()

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._detector_thread = threading.Thread(target=self._detector_loop, daemon=True)

        self.detector_status = "Starting"
        self.hydration_count = 0
        self._last_state = self.state_machine.snapshot()

    def start(self) -> None:
        log_dir = ensure_logs_dir()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(log_dir / "events.log"), logging.StreamHandler()],
        )

        logging.info("JalLijiye starting")

        self.detector.start()
        self.detector_status = self.detector.status.message
        self.alerter.start()
        self._detector_thread.start()

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
        now = time.time()
        event = DrinkEvent(timestamp=now, source="manual", confidence=1.0)
        with self._lock:
            state = self.state_machine.mark_drink(event, now=now)
            self._last_state = state
            self.hydration_count += 1
        logging.info("Drink event: source=manual")

    def pause_30_minutes(self) -> None:
        now = time.time()
        with self._lock:
            state = self.state_machine.pause_for(30, now=now)
            self._last_state = state
        logging.info("Manual pause enabled for 30 minutes")

    def get_status(self) -> tuple[str, str]:
        now = time.time()
        with self._lock:
            self._last_state = self.state_machine.tick(now=now)
            state = self._last_state
            hydration_count = self.hydration_count
            hydration_goal = self.config.daily_hydration_goal

        self.detector.pump_debug_window()
        self.alerter.set_mode(state.mode, self.state_machine.alert_started_at)

        progress_title = f"Hydration: {hydration_count}/{hydration_goal}"
        remaining_minutes = max(0.0, self.config.alert_after_minutes - state.minutes_since_drink)
        if remaining_minutes <= 0.0:
            countdown_title = "Drink right now"
        elif remaining_minutes < 1.0:
            countdown_title = "Drink after: <1m"
        else:
            countdown_title = f"Drink after: {math.ceil(remaining_minutes)}m"

        alternating_title = progress_title if int(now) % 6 < 3 else countdown_title
        return alternating_title, progress_title

    def _detector_loop(self) -> None:
        while not self._stop.is_set():
            frame = self.detector.read_frame()

            if frame is None:
                self.detector_status = self.detector.status.message
                time.sleep(0.05)
                continue

            raw_present = is_reliable_person_presence(
                frame=frame,
                min_area_ratio=self.config.presence_person_min_area_ratio,
                center_margin=self.config.presence_person_center_margin,
            )
            stable_present = self.presence.update(raw_present=raw_present, now=frame.timestamp)

            with self._lock:
                self.state_machine.set_presence(stable_present, now=frame.timestamp)
                vision_event = self.drink_heuristic.update(frame)
                if vision_event is not None:
                    self.state_machine.mark_drink(vision_event, now=frame.timestamp)
                    self.hydration_count += 1
                    logging.info(
                        "Drink event: source=vision confidence=%.2f",
                        vision_event.confidence,
                    )
                current_state = self.state_machine.tick(now=frame.timestamp)
                self._last_state = current_state
                heuristic_debug = self.drink_heuristic.debug_snapshot()

            self.detector.show_debug_stream(
                frame=frame,
                present=stable_present,
                mode=current_state.mode,
                debug=heuristic_debug,
            )

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
    menu.run()


if __name__ == "__main__":
    run()
