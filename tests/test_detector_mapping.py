import numpy as np

from app.detector import VisionDetector
from app.types import Config


class FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy: list[float]) -> None:
        self.cls = np.array([cls_id], dtype=float)
        self.conf = np.array([conf], dtype=float)
        self.xyxy = np.array([xyxy], dtype=float)


class FakeResult:
    def __init__(self, boxes: list[FakeBox]) -> None:
        self.boxes = boxes


def test_custom_class_zero_maps_to_bottle() -> None:
    cfg = Config(bottle_class_id=0, bottle_confidence=0.25)
    detector = VisionDetector(cfg)

    results = [FakeResult([FakeBox(0, 0.9, [1, 2, 3, 4]), FakeBox(1, 0.9, [5, 6, 7, 8])])]
    boxes = detector._extract_boxes(
        results,
        accepted_classes={cfg.bottle_class_id},
        min_confidence=cfg.bottle_confidence,
        label="bottle",
    )

    assert len(boxes) == 1
    assert boxes[0].label == "bottle"
    assert boxes[0].x1 == 1


def test_confidence_threshold_filters_custom_boxes() -> None:
    cfg = Config(bottle_class_id=0, bottle_confidence=0.25)
    detector = VisionDetector(cfg)

    results = [FakeResult([FakeBox(0, 0.1, [1, 2, 3, 4])])]
    boxes = detector._extract_boxes(
        results,
        accepted_classes={cfg.bottle_class_id},
        min_confidence=cfg.bottle_confidence,
        label="bottle",
    )

    assert boxes == []
