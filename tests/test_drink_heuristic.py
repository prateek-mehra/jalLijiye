from app.detector import DrinkHeuristic
from app.types import Box, Config, DetectionFrame


CFG = Config(
    fps=5,
    object_confidence=0.45,
    bottle_confidence=0.25,
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


def _top_contact_only_frame(ts: float, conf: float = 0.9) -> DetectionFrame:
    # Bottle center is below mouth box, but bottle top enters mouth ROI.
    return DetectionFrame(
        timestamp=ts,
        person_boxes=[Box(0, 0, 120, 180, confidence=0.99, label="person")],
        bottle_boxes=[Box(50, 70, 60, 100, confidence=conf, label="bottle")],
        face_box=Box(20, 20, 100, 100, label="face"),
        mouth_roi=Box(40, 40, 80, 80, label="mouth"),
    )


def _occluded_mouth_frame(ts: float, conf: float = 0.9) -> DetectionFrame:
    # Mouth is not currently detected (occluded), but person+bottle remain visible.
    return DetectionFrame(
        timestamp=ts,
        person_boxes=[Box(0, 0, 120, 180, confidence=0.99, label="person")],
        bottle_boxes=[Box(50, 50, 60, 60, confidence=conf, label="bottle")],
        face_box=None,
        mouth_roi=None,
    )


def test_detects_drink_when_bottle_overlaps_long_enough() -> None:
    heuristic = DrinkHeuristic(CFG)
    event = None

    # 10 frames at 5 fps ~= 2.0 seconds
    for i in range(10):
        event = heuristic.update(_frame(ts=i * 0.2, bottle_in_mouth=True))

    assert event is not None
    assert event.source == "vision"


def test_detects_drink_when_only_top_of_bottle_reaches_mouth() -> None:
    heuristic = DrinkHeuristic(CFG)
    event = None

    for i in range(10):
        event = heuristic.update(_top_contact_only_frame(ts=i * 0.2))

    assert event is not None
    assert event.source == "vision"


def test_detects_drink_using_cached_mouth_when_current_mouth_missing() -> None:
    heuristic = DrinkHeuristic(
        Config(
            fps=5,
            bottle_confidence=0.25,
            drink_hold_seconds=1.0,
            drink_window_seconds=5.0,
            drink_cooldown_minutes=10.0,
            mouth_memory_seconds=3.0,
        )
    )

    event = heuristic.update(_frame(ts=0.0, bottle_in_mouth=True, with_face=True))
    assert event is None

    # Next frames simulate mouth occluded by bottle while person/bottle are still detected.
    for i in range(1, 6):
        candidate = heuristic.update(_occluded_mouth_frame(ts=i * 0.2))
        if candidate is not None:
            event = candidate
            break

    assert event is not None
    assert event.source == "vision"


def test_does_not_use_expired_cached_mouth() -> None:
    heuristic = DrinkHeuristic(
        Config(
            fps=5,
            bottle_confidence=0.25,
            drink_hold_seconds=1.0,
            drink_window_seconds=5.0,
            drink_cooldown_minutes=10.0,
            mouth_memory_seconds=0.2,
        )
    )

    # Cache one mouth ROI at t=0.
    heuristic.update(_frame(ts=0.0, bottle_in_mouth=True, with_face=True))

    # Mouth cache expires before next frame at t=1.0.
    event = heuristic.update(_occluded_mouth_frame(ts=1.0))
    assert event is None


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


def test_accepts_low_confidence_bottle_when_bottle_threshold_is_low() -> None:
    heuristic = DrinkHeuristic(
        Config(
            fps=5,
            object_confidence=0.8,
            bottle_confidence=0.05,
            drink_hold_seconds=2.0,
            drink_window_seconds=5.0,
            drink_cooldown_minutes=10.0,
        )
    )

    event = None
    for i in range(10):
        event = heuristic.update(_frame(ts=i * 0.2, bottle_in_mouth=True, conf=0.1))

    assert event is not None
