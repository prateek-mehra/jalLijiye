from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from app.types import Box, Config, DetectionFrame, DrinkEvent

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None


@dataclass(slots=True)
class DetectorStatus:
    available: bool
    message: str


class VisionDetector:
    """Webcam detector for person, bottle, and approximate mouth ROI."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._cap: Any | None = None
        self._model: Any | None = None
        self._face_detector: Any | None = None
        self._last_frame_ts: float = 0.0
        self.status = DetectorStatus(available=False, message="Initializing")

    def start(self) -> None:
        if cv2 is None:
            self.status = DetectorStatus(False, "OpenCV not installed; manual mode only")
            return

        try:
            self._cap = cv2.VideoCapture(0)
            if not self._cap or not self._cap.isOpened():
                self.status = DetectorStatus(False, "Camera unavailable; manual mode only")
                return

            self._face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            if YOLO is not None:
                self._model = YOLO(self.config.model_path)
                self.status = DetectorStatus(True, "Vision active")
            else:
                self.status = DetectorStatus(False, "Ultralytics missing; manual mode only")
                self.close()
        except Exception as exc:  # pragma: no cover - runtime safety path
            self.status = DetectorStatus(False, f"Detector init failed: {exc}")
            self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_frame(self) -> DetectionFrame | None:
        if not self.status.available:
            return None
        if cv2 is None or self._cap is None or self._model is None:
            return None

        now = time.time()
        min_interval = 1.0 / max(1, self.config.fps)
        if now - self._last_frame_ts < min_interval:
            return None

        ok, frame = self._cap.read()
        self._last_frame_ts = now
        if not ok:
            self.status = DetectorStatus(False, "Camera read failed; manual mode only")
            return None

        person_boxes: list[Box] = []
        bottle_boxes: list[Box] = []

        try:
            results = self._model.predict(
                frame,
                verbose=False,
                conf=self.config.object_confidence,
                classes=[0, 39],  # COCO: person, bottle
            )
            if results:
                for box in results[0].boxes:
                    cls_idx = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    parsed = Box(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)
                    if cls_idx == 0:
                        parsed.label = "person"
                        person_boxes.append(parsed)
                    elif cls_idx == 39:
                        parsed.label = "bottle"
                        bottle_boxes.append(parsed)
        except Exception as exc:  # pragma: no cover - runtime safety path
            self.status = DetectorStatus(False, f"Inference failed: {exc}")
            return None

        face_box, mouth_roi = self._detect_face_and_mouth_roi(frame)

        return DetectionFrame(
            timestamp=now,
            person_boxes=person_boxes,
            bottle_boxes=bottle_boxes,
            face_box=face_box,
            mouth_roi=mouth_roi,
        )

    def _detect_face_and_mouth_roi(self, frame: Any) -> tuple[Box | None, Box | None]:
        if cv2 is None or self._face_detector is None:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(faces) == 0:
            return None, None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = Box(float(x), float(y), float(x + w), float(y + h), confidence=1.0, label="face")

        mouth_x1 = x + (0.20 * w)
        mouth_x2 = x + (0.80 * w)
        mouth_y1 = y + (0.58 * h)
        mouth_y2 = y + (0.92 * h)
        mouth = Box(float(mouth_x1), float(mouth_y1), float(mouth_x2), float(mouth_y2), label="mouth")

        return face, mouth


class DrinkHeuristic:
    """Heuristic drink detector using bottle center overlap with mouth ROI over time."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._contact_times: deque[float] = deque()
        self._last_drink_ts: float | None = None

    def update(self, frame: DetectionFrame) -> DrinkEvent | None:
        now = frame.timestamp
        self._prune(now)

        if not (frame.face_box or frame.person_boxes):
            return None
        if frame.mouth_roi is None or not frame.bottle_boxes:
            return None

        expanded_mouth = frame.mouth_roi.expanded(self.config.mouth_expand_ratio)

        bottle_contact = False
        for bottle in frame.bottle_boxes:
            if bottle.confidence < self.config.object_confidence:
                continue
            cx, cy = bottle.center()
            if expanded_mouth.contains_point(cx, cy):
                bottle_contact = True
                break

        if bottle_contact:
            self._contact_times.append(now)
            self._prune(now)

        contact_seconds = len(self._contact_times) / max(1, self.config.fps)
        cooldown_seconds = self.config.drink_cooldown_minutes * 60

        cooldown_elapsed = (
            self._last_drink_ts is None or (now - self._last_drink_ts) >= cooldown_seconds
        )

        if contact_seconds >= self.config.drink_hold_seconds and cooldown_elapsed:
            self._last_drink_ts = now
            self._contact_times.clear()
            confidence = min(1.0, contact_seconds / self.config.drink_hold_seconds)
            return DrinkEvent(timestamp=now, source="vision", confidence=confidence)

        return None

    def _prune(self, now: float) -> None:
        window = self.config.drink_window_seconds
        while self._contact_times and (now - self._contact_times[0]) > window:
            self._contact_times.popleft()
