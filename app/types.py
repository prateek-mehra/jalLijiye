from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


AppMode = Literal["PAUSED_ABSENT", "TRACKING", "ALERT_ESCALATING", "ALERT_CONTINUOUS"]
DrinkSource = Literal["vision", "manual"]


@dataclass(slots=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    label: str = ""

    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def expanded(self, ratio: float) -> "Box":
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        dx = w * ratio
        dy = h * ratio
        return Box(
            x1=self.x1 - dx,
            y1=self.y1 - dy,
            x2=self.x2 + dx,
            y2=self.y2 + dy,
            confidence=self.confidence,
            label=self.label,
        )


@dataclass(slots=True)
class DetectionFrame:
    timestamp: float
    person_boxes: list[Box] = field(default_factory=list)
    bottle_boxes: list[Box] = field(default_factory=list)
    face_box: Box | None = None
    mouth_roi: Box | None = None
    bottle_source: str = "none"
    frame: Any | None = None


@dataclass(slots=True)
class DrinkEvent:
    timestamp: float
    source: DrinkSource
    confidence: float


@dataclass(slots=True)
class AppState:
    mode: AppMode
    minutes_since_drink: float
    is_present: bool
    status_detail: str = ""


@dataclass(slots=True)
class Config:
    alert_after_minutes: int = 1
    absence_pause_minutes: int = 2
    fps: int = 5
    object_confidence: float = 0.45
    bottle_confidence: float = 0.45
    mouth_expand_ratio: float = 0.15
    mouth_memory_seconds: float = 2.5
    drink_hold_seconds: float = 1.0
    drink_window_seconds: float = 5.0
    drink_cooldown_minutes: float = 10.0
    escalating_minutes: float = 3.0
    model_path: str = "yolov8n.pt"
    person_model_path: str = "yolov8n.pt"
    bottle_model_path: str = "models/bottle_v1/weights/best.pt"
    bottle_class_id: int = 0
    use_coco_bottle_fallback: bool = True
    show_debug_window: bool = True
