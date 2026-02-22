import numpy as np

from app.detector import VisionDetector
from app.types import Box, Config


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


def test_fallback_activates_when_custom_missing_and_enabled() -> None:
    cfg = Config(use_coco_bottle_fallback=True)
    detector = VisionDetector(cfg)

    coco = [Box(10, 10, 30, 30, confidence=0.6, label="bottle")]
    selected, source = detector._select_bottle_boxes(custom_bottle_boxes=[], coco_bottle_boxes=coco)

    assert source == "coco_fallback"
    assert len(selected) == 1


def test_fallback_disabled_ignores_coco_boxes() -> None:
    cfg = Config(use_coco_bottle_fallback=False)
    detector = VisionDetector(cfg)

    coco = [Box(10, 10, 30, 30, confidence=0.6, label="bottle")]
    selected, source = detector._select_bottle_boxes(custom_bottle_boxes=[], coco_bottle_boxes=coco)

    assert source == "none"
    assert selected == []


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
