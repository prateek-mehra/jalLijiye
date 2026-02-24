from app.main import is_reliable_person_presence
from app.types import Box, DetectionFrame


class DummyFrame:
    def __init__(self, h: int, w: int) -> None:
        self.shape = (h, w, 3)


def _det(person_boxes: list[Box], h: int = 720, w: int = 1280) -> DetectionFrame:
    return DetectionFrame(
        timestamp=0.0,
        person_boxes=person_boxes,
        bottle_boxes=[],
        face_box=None,
        mouth_roi=None,
        frame=DummyFrame(h, w),
    )


def test_reliable_person_presence_with_large_centered_box() -> None:
    frame = _det([Box(420, 120, 940, 700, confidence=0.9, label="person")])

    assert is_reliable_person_presence(frame, min_area_ratio=0.06, center_margin=0.20)


def test_reliable_person_presence_rejects_small_box() -> None:
    frame = _det([Box(20, 20, 160, 160, confidence=0.9, label="person")])

    assert not is_reliable_person_presence(frame, min_area_ratio=0.06, center_margin=0.20)


def test_reliable_person_presence_rejects_off_center_box() -> None:
    frame = _det([Box(0, 60, 420, 700, confidence=0.9, label="person")])

    assert not is_reliable_person_presence(frame, min_area_ratio=0.06, center_margin=0.20)
