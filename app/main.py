from __future__ import annotations

import logging
import os
import signal
import threading
import time

from app.alerter import Alerter
from app.config import load_config
from app.detector import DrinkHeuristic, VisionDetector
from app.menu_bar import JalLijiyeMenuBar
from app.presence import PresenceTracker
from app.state_machine import HydrationStateMachine
from app.types import DrinkEvent


class JalLijiyeController:
    def __init__(self) -> None:
        self.config = load_config()
        self.state_machine = HydrationStateMachine(self.config)
        self.presence = PresenceTracker(
            required_present_frames=3,
            history_size=5,
            absent_after_seconds=self.config.absence_pause_minutes * 60,
        )
        self.detector = VisionDetector(self.config)
        self.drink_heuristic = DrinkHeuristic(self.config)
        self.alerter = Alerter()

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._detector_thread = threading.Thread(target=self._detector_loop, daemon=True)

        self.detector_status = "Starting"
        self._last_state = self.state_machine.snapshot()

    def start(self) -> None:
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler("logs/events.log"), logging.StreamHandler()],
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

    def mark_manual_drink(self) -> None:
        now = time.time()
        event = DrinkEvent(timestamp=now, source="manual", confidence=1.0)
        with self._lock:
            state = self.state_machine.mark_drink(event, now=now)
            self._last_state = state
        logging.info("Drink event: source=manual")

    def pause_30_minutes(self) -> None:
        now = time.time()
        with self._lock:
            state = self.state_machine.pause_for(30, now=now)
            self._last_state = state
        logging.info("Manual pause enabled for 30 minutes")

    def resume(self) -> None:
        now = time.time()
        with self._lock:
            state = self.state_machine.resume(now=now)
            self._last_state = state
        logging.info("Manual pause cleared")

    def get_status(self) -> tuple[str, str, str]:
        now = time.time()
        with self._lock:
            self._last_state = self.state_machine.tick(now=now)
            state = self._last_state

        self.detector.pump_debug_window()
        self.alerter.set_mode(state.mode, self.state_machine.alert_started_at)

        if state.mode == "PAUSED_ABSENT":
            if state.status_detail == "Manual pause":
                title = "Paused (Manual)"
            else:
                title = "Paused (Away)"
        elif state.mode == "ALERT_ESCALATING":
            title = "ALERT: Drink water"
        elif state.mode == "ALERT_CONTINUOUS":
            title = "ALERT: Drink NOW"
        else:
            title = f"Hydration: {int(state.minutes_since_drink)}m"

        detail = state.status_detail or "Tracking"
        return title, detail, self.detector_status

    def _detector_loop(self) -> None:
        while not self._stop.is_set():
            frame = self.detector.read_frame()

            if frame is None:
                self.detector_status = self.detector.status.message
                time.sleep(0.05)
                continue

            raw_present = bool(frame.face_box or frame.person_boxes)
            stable_present = self.presence.update(raw_present=raw_present, now=frame.timestamp)

            with self._lock:
                self.state_machine.set_presence(stable_present, now=frame.timestamp)
                vision_event = self.drink_heuristic.update(frame)
                if vision_event is not None:
                    self.state_machine.mark_drink(vision_event, now=frame.timestamp)
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
        on_resume=controller.resume,
        on_quit=controller.stop,
    )
    menu.run()


if __name__ == "__main__":
    run()
