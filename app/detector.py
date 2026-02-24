from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

from app.types import AppMode, Box, Config, DetectionFrame, DrinkEvent

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


@dataclass(slots=True)
class DrinkHeuristicDebug:
    bottle_contact: bool
    contact_seconds: float
    cooldown_remaining_seconds: float
    using_cached_mouth: bool
    mouth_age_seconds: float
    drink_detected: bool


class VisionDetector:
    """Webcam detector for person, bottle, and approximate mouth ROI."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._cap: Any | None = None
        self._person_model: Any | None = None
        self._bottle_model: Any | None = None
        self._face_detector: Any | None = None
        self._last_frame_ts: float = 0.0
        self._window_name = "JalLijiye Debug Stream"
        self._debug_lock = threading.Lock()
        self._pending_debug_frame: Any | None = None
        self._debug_window_disabled = False
        self.status = DetectorStatus(available=False, message="Initializing")

    def start(self) -> None:
        if cv2 is None:
            self.status = DetectorStatus(False, "OpenCV not installed; manual mode only")
            return
        if YOLO is None:
            self.status = DetectorStatus(False, "Ultralytics missing; manual mode only")
            return

        try:
            self._cap = cv2.VideoCapture(0)
            if not self._cap or not self._cap.isOpened():
                self.status = DetectorStatus(False, "Camera unavailable; manual mode only")
                return

            self._face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            self._person_model = YOLO(self.config.person_model_path or self.config.model_path)

            bottle_model_error = ""
            bottle_path = Path(self.config.bottle_model_path)
            if bottle_path.exists():
                try:
                    self._bottle_model = YOLO(str(bottle_path))
                except Exception as exc:  # pragma: no cover - runtime safety path
                    bottle_model_error = f"custom bottle model load failed: {exc}"
            else:
                bottle_model_error = "custom bottle model missing"

            if self._bottle_model is not None:
                message = "Vision active (custom bottle model)"
            elif self.config.use_coco_bottle_fallback:
                message = "Vision active (COCO bottle fallback)"
            else:
                message = "Vision active (no bottle detector)"

            if bottle_model_error:
                message = f"{message}; {bottle_model_error}"

            self.status = DetectorStatus(True, message)
        except Exception as exc:  # pragma: no cover - runtime safety path
            self.status = DetectorStatus(False, f"Detector init failed: {exc}")
            self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if cv2 is not None:
            try:
                cv2.destroyWindow(self._window_name)
            except Exception:
                pass

    def read_frame(self) -> DetectionFrame | None:
        if not self.status.available:
            return None
        if cv2 is None or self._cap is None or self._person_model is None:
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

        try:
            base_results = self._person_model.predict(
                frame,
                verbose=False,
                conf=min(self.config.object_confidence, self.config.bottle_confidence),
                classes=[0, 39],  # COCO: person, bottle
            )
        except Exception as exc:  # pragma: no cover - runtime safety path
            self.status = DetectorStatus(False, f"Inference failed: {exc}")
            return None

        person_boxes = self._extract_boxes(
            base_results,
            accepted_classes={0},
            min_confidence=self.config.object_confidence,
            label="person",
        )

        coco_bottle_boxes = self._extract_boxes(
            base_results,
            accepted_classes={39},
            min_confidence=self.config.bottle_confidence,
            label="bottle",
        )

        custom_bottle_boxes: list[Box] = []
        if self._bottle_model is not None:
            try:
                custom_results = self._bottle_model.predict(
                    frame,
                    verbose=False,
                    conf=self.config.bottle_confidence,
                    classes=[self.config.bottle_class_id],
                )
                custom_bottle_boxes = self._extract_boxes(
                    custom_results,
                    accepted_classes={self.config.bottle_class_id},
                    min_confidence=self.config.bottle_confidence,
                    label="bottle",
                )
            except Exception:  # pragma: no cover - runtime safety path
                custom_bottle_boxes = []

        bottle_boxes, bottle_source = self._select_bottle_boxes(
            custom_bottle_boxes=custom_bottle_boxes,
            coco_bottle_boxes=coco_bottle_boxes,
        )

        face_box, mouth_roi = self._detect_face_and_mouth_roi(frame, person_boxes)

        return DetectionFrame(
            timestamp=now,
            person_boxes=person_boxes,
            bottle_boxes=bottle_boxes,
            face_box=face_box,
            mouth_roi=mouth_roi,
            bottle_source=bottle_source,
            frame=frame,
        )

    def _extract_boxes(
        self,
        results: Any,
        accepted_classes: set[int] | None,
        min_confidence: float,
        label: str,
    ) -> list[Box]:
        parsed: list[Box] = []
        if not results:
            return parsed

        for result in results:
            raw_boxes = getattr(result, "boxes", None)
            if raw_boxes is None:
                continue

            for raw_box in raw_boxes:
                try:
                    cls_val = raw_box.cls[0]
                    conf_val = raw_box.conf[0]
                    cls_idx = int(cls_val.item() if hasattr(cls_val, "item") else cls_val)
                    conf = float(conf_val.item() if hasattr(conf_val, "item") else conf_val)
                    xyxy = raw_box.xyxy[0]
                    coords = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)
                    if len(coords) < 4:
                        continue
                    x1, y1, x2, y2 = coords[:4]
                except Exception:
                    continue

                if accepted_classes is not None and cls_idx not in accepted_classes:
                    continue
                if conf < min_confidence:
                    continue

                parsed.append(
                    Box(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=conf,
                        label=label,
                    )
                )

        return parsed

    def _select_bottle_boxes(
        self,
        custom_bottle_boxes: list[Box],
        coco_bottle_boxes: list[Box],
    ) -> tuple[list[Box], str]:
        if custom_bottle_boxes:
            return custom_bottle_boxes, "custom"
        if self.config.use_coco_bottle_fallback and coco_bottle_boxes:
            return coco_bottle_boxes, "coco_fallback"
        return [], "none"

    def show_debug_stream(
        self,
        frame: DetectionFrame,
        present: bool,
        mode: AppMode,
        debug: DrinkHeuristicDebug,
    ) -> None:
        if (
            cv2 is None
            or not self.config.show_debug_window
            or frame.frame is None
            or self._debug_window_disabled
        ):
            return

        canvas = frame.frame.copy()

        for person in frame.person_boxes:
            self._draw_box(canvas, person, (0, 255, 0), "person")
        for bottle in frame.bottle_boxes:
            self._draw_box(canvas, bottle, (255, 128, 0), f"bottle {bottle.confidence:.2f}")
            cx, cy = bottle.center()
            cv2.circle(canvas, (int(cx), int(cy)), 4, (0, 255, 255), -1)

        if frame.face_box is not None:
            self._draw_box(canvas, frame.face_box, (255, 0, 255), "face")
        if frame.mouth_roi is not None:
            self._draw_box(canvas, frame.mouth_roi, (0, 0, 255), "mouth")
            expanded = frame.mouth_roi.expanded(self.config.mouth_expand_ratio)
            self._draw_box(canvas, expanded, (0, 165, 255), "mouth+margin")

        lines = [
            f"mode={mode}",
            f"present={present}",
            f"bottle_source={frame.bottle_source}",
            f"bottle_count={len(frame.bottle_boxes)}",
            f"cached_mouth={debug.using_cached_mouth}",
            f"mouth_age={debug.mouth_age_seconds:.2f}s",
            f"contact={debug.bottle_contact}",
            f"contact_seconds={debug.contact_seconds:.2f}/{self.config.drink_hold_seconds:.2f}",
            f"cooldown_remaining={debug.cooldown_remaining_seconds:.1f}s",
            f"drink_detected={debug.drink_detected}",
        ]

        y = 24
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 24

        with self._debug_lock:
            self._pending_debug_frame = canvas

    def pump_debug_window(self) -> None:
        """
        Must run on the main thread on macOS to keep OpenCV HighGUI stable.
        """
        if cv2 is None or not self.config.show_debug_window or self._debug_window_disabled:
            return

        with self._debug_lock:
            canvas = self._pending_debug_frame
            self._pending_debug_frame = None

        if canvas is None:
            return

        try:
            cv2.imshow(self._window_name, canvas)
            cv2.waitKey(1)
        except Exception:
            self._debug_window_disabled = True
            self.status = DetectorStatus(
                self.status.available,
                "Debug window disabled (OpenCV HighGUI error)",
            )

    def _draw_box(self, frame: Any, box: Box, color: tuple[int, int, int], label: str) -> None:
        if cv2 is None:
            return
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    def _detect_face_and_mouth_roi(
        self,
        frame: Any,
        person_boxes: list[Box],
    ) -> tuple[Box | None, Box | None]:
        if cv2 is None or self._face_detector is None:
            return None, None
        if not person_boxes:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(faces) == 0:
            return None, None

        supported_faces: list[Box] = []
        for x, y, w, h in faces:
            candidate = Box(float(x), float(y), float(x + w), float(y + h), confidence=1.0, label="face")
            if self._face_supported_by_person(candidate, person_boxes):
                supported_faces.append(candidate)

        if not supported_faces:
            return None, None

        face = max(supported_faces, key=lambda f: (f.x2 - f.x1) * (f.y2 - f.y1))

        fx, fy, fw, fh = face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1
        mouth_x1 = fx + (0.20 * fw)
        mouth_x2 = fx + (0.80 * fw)
        mouth_y1 = fy + (0.58 * fh)
        mouth_y2 = fy + (0.92 * fh)
        mouth = Box(float(mouth_x1), float(mouth_y1), float(mouth_x2), float(mouth_y2), label="mouth")

        return face, mouth

    def _face_supported_by_person(self, face: Box, person_boxes: list[Box]) -> bool:
        fx, fy = face.center()
        for person in person_boxes:
            if person.contains_point(fx, fy):
                return True

            inter_x1 = max(face.x1, person.x1)
            inter_y1 = max(face.y1, person.y1)
            inter_x2 = min(face.x2, person.x2)
            inter_y2 = min(face.y2, person.y2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            face_area = max(1.0, (face.x2 - face.x1) * (face.y2 - face.y1))
            if (inter_area / face_area) >= 0.25:
                return True

        return False


class DrinkHeuristic:
    """Heuristic drink detector using bottle-mouth contact over time."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._contact_times: deque[float] = deque()
        self._last_drink_ts: float | None = None
        self._last_mouth_roi: Box | None = None
        self._last_mouth_ts: float | None = None
        self._last_debug = DrinkHeuristicDebug(
            bottle_contact=False,
            contact_seconds=0.0,
            cooldown_remaining_seconds=0.0,
            using_cached_mouth=False,
            mouth_age_seconds=0.0,
            drink_detected=False,
        )

    def update(self, frame: DetectionFrame) -> DrinkEvent | None:
        now = frame.timestamp
        self._prune(now)
        cooldown_seconds = self.config.drink_cooldown_minutes * 60
        cooldown_remaining_seconds = 0.0
        if self._last_drink_ts is not None:
            cooldown_remaining_seconds = max(0.0, cooldown_seconds - (now - self._last_drink_ts))
        effective_mouth, using_cached_mouth, mouth_age_seconds = self._resolve_mouth_roi(frame, now)

        if not (frame.face_box or frame.person_boxes):
            self._last_debug = DrinkHeuristicDebug(
                bottle_contact=False,
                contact_seconds=0.0,
                cooldown_remaining_seconds=cooldown_remaining_seconds,
                using_cached_mouth=using_cached_mouth,
                mouth_age_seconds=mouth_age_seconds,
                drink_detected=False,
            )
            return None
        if effective_mouth is None or not frame.bottle_boxes:
            contact_seconds = len(self._contact_times) / max(1, self.config.fps)
            self._last_debug = DrinkHeuristicDebug(
                bottle_contact=False,
                contact_seconds=contact_seconds,
                cooldown_remaining_seconds=cooldown_remaining_seconds,
                using_cached_mouth=using_cached_mouth,
                mouth_age_seconds=mouth_age_seconds,
                drink_detected=False,
            )
            return None

        expanded_mouth = effective_mouth.expanded(self.config.mouth_expand_ratio)

        bottle_contact = False
        for bottle in frame.bottle_boxes:
            if bottle.confidence < self.config.bottle_confidence:
                continue
            if self._is_bottle_near_mouth(bottle, expanded_mouth):
                bottle_contact = True
                break

        if bottle_contact:
            self._contact_times.append(now)
            self._prune(now)

        contact_seconds = len(self._contact_times) / max(1, self.config.fps)
        cooldown_elapsed = (
            self._last_drink_ts is None or (now - self._last_drink_ts) >= cooldown_seconds
        )
        self._last_debug = DrinkHeuristicDebug(
            bottle_contact=bottle_contact,
            contact_seconds=contact_seconds,
            cooldown_remaining_seconds=cooldown_remaining_seconds,
            using_cached_mouth=using_cached_mouth,
            mouth_age_seconds=mouth_age_seconds,
            drink_detected=False,
        )

        if contact_seconds >= self.config.drink_hold_seconds and cooldown_elapsed:
            self._last_drink_ts = now
            self._contact_times.clear()
            confidence = min(1.0, contact_seconds / self.config.drink_hold_seconds)
            self._last_debug = DrinkHeuristicDebug(
                bottle_contact=bottle_contact,
                contact_seconds=contact_seconds,
                cooldown_remaining_seconds=0.0,
                using_cached_mouth=using_cached_mouth,
                mouth_age_seconds=mouth_age_seconds,
                drink_detected=True,
            )
            return DrinkEvent(timestamp=now, source="vision", confidence=confidence)

        return None

    def debug_snapshot(self) -> DrinkHeuristicDebug:
        return self._last_debug

    def _prune(self, now: float) -> None:
        window = self.config.drink_window_seconds
        while self._contact_times and (now - self._contact_times[0]) > window:
            self._contact_times.popleft()

    def _is_bottle_near_mouth(self, bottle: Box, mouth: Box) -> bool:
        # Robust contact logic:
        # 1) Bottle center near mouth.
        # 2) Bottle top-center near mouth (covers tilted bottle while drinking).
        # 3) Any meaningful overlap between bottle box and mouth box.
        cx, cy = bottle.center()
        top_center_x = cx
        top_center_y = bottle.y1

        if mouth.contains_point(cx, cy):
            return True
        if mouth.contains_point(top_center_x, top_center_y):
            return True

        overlap_ratio = self._overlap_ratio(bottle, mouth)
        return overlap_ratio >= 0.03

    def _resolve_mouth_roi(self, frame: DetectionFrame, now: float) -> tuple[Box | None, bool, float]:
        if frame.mouth_roi is not None:
            self._last_mouth_roi = frame.mouth_roi
            self._last_mouth_ts = now
            return frame.mouth_roi, False, 0.0

        if self._last_mouth_roi is None or self._last_mouth_ts is None:
            return None, False, 0.0

        mouth_age = now - self._last_mouth_ts
        if mouth_age <= self.config.mouth_memory_seconds:
            return self._last_mouth_roi, True, mouth_age

        return None, False, mouth_age

    def _overlap_ratio(self, a: Box, b: Box) -> float:
        inter_x1 = max(a.x1, b.x1)
        inter_y1 = max(a.y1, b.y1)
        inter_x2 = min(a.x2, b.x2)
        inter_y2 = min(a.y2, b.y2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        a_area = max(0.0, (a.x2 - a.x1) * (a.y2 - a.y1))
        b_area = max(0.0, (b.x2 - b.x1) * (b.y2 - b.y1))
        denom = min(a_area, b_area)
        if denom <= 0:
            return 0.0
        return inter_area / denom
