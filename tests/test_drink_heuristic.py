from app.detector import DrinkHeuristic
from app.types import Box, Config, DetectionFrame


CFG = Config(
    fps=5,
    object_confidence=0.45,
    drink_hold_seconds=2.0,
    drink_window_seconds=5.0,
    drink_cooldown_minutes=10.0,
)


def _frame(ts: float, bottle_in_mouth: bool, with_face: bool = True, conf: float = 0.9) -> DetectionFrame:
    mouth = Box(40, 40, 80, 80, label="mouth") if with_face else None
    bottle = [Box(50, 50, 60, 60, confidence=conf, label="bottle")] if bottle_in_mouth else []
    face = Box(20, 20, 100, 100, label="face") if with_face else None

    return DetectionFrame(
        timestamp=ts,
        person_boxes=[Box(0, 0, 120, 180, confidence=0.99, label="person")],
        bottle_boxes=bottle,
        face_box=face,
        mouth_roi=mouth,
    )


def test_detects_drink_when_bottle_overlaps_long_enough() -> None:
    heuristic = DrinkHeuristic(CFG)
    event = None

    # 10 frames at 5 fps ~= 2.0 seconds
    for i in range(10):
        event = heuristic.update(_frame(ts=i * 0.2, bottle_in_mouth=True))

    assert event is not None
    assert event.source == "vision"


def test_rejects_brief_contact() -> None:
    heuristic = DrinkHeuristic(CFG)

    event = None
    for i in range(4):  # 0.8s < 2.0s threshold
        event = heuristic.update(_frame(ts=i * 0.2, bottle_in_mouth=True))

    assert event is None


def test_rejects_when_no_face_or_mouth_roi() -> None:
    heuristic = DrinkHeuristic(CFG)

    event = None
    for i in range(12):
        event = heuristic.update(_frame(ts=i * 0.2, bottle_in_mouth=True, with_face=False))

    assert event is None


def test_rejects_low_confidence_bottle() -> None:
    heuristic = DrinkHeuristic(CFG)

    event = None
    for i in range(12):
        event = heuristic.update(_frame(ts=i * 0.2, bottle_in_mouth=True, conf=0.2))

    assert event is None
