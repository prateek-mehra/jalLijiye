from app.detector import DrinkHeuristic
from app.presence import PresenceTracker
from app.state_machine import HydrationStateMachine
from app.types import Box, Config, DetectionFrame


def _cfg() -> Config:
    return Config(
        alert_after_minutes=90,
        escalating_minutes=3,
        absence_pause_minutes=2,
        fps=5,
    )


def _drink_frame(ts: float) -> DetectionFrame:
    return DetectionFrame(
        timestamp=ts,
        person_boxes=[Box(0, 0, 120, 180, confidence=0.9, label="person")],
        bottle_boxes=[Box(50, 50, 60, 60, confidence=0.9, label="bottle")],
        face_box=Box(20, 20, 100, 100, confidence=0.9, label="face"),
        mouth_roi=Box(40, 40, 80, 80, confidence=0.9, label="mouth"),
    )


def test_state_timeline_with_absence_and_drink_event() -> None:
    cfg = _cfg()
    sm = HydrationStateMachine(cfg, initial_now=0.0)
    presence = PresenceTracker(absent_after_seconds=120)
    heuristic = DrinkHeuristic(cfg)

    timeline = []

    timeline.append(sm.tick(now=89 * 60).mode)
    timeline.append(sm.tick(now=90 * 60).mode)
    timeline.append(sm.tick(now=93 * 60).mode)

    # Simulate 2+ minutes absence -> pauses alert.
    for i in range(13):
        stable = presence.update(raw_present=False, now=(93 * 60) + (i * 10))
    sm.set_presence(stable, now=(95 * 60) + 1)
    timeline.append(sm.tick(now=(95 * 60) + 1).mode)

    # Return present and detect drink event.
    for i in range(5):
        stable = presence.update(raw_present=True, now=(96 * 60) + i)
    sm.set_presence(stable, now=(96 * 60) + 5)

    event = None
    for i in range(10):
        candidate = heuristic.update(_drink_frame((96 * 60) + (i * 0.2)))
        if candidate is not None:
            event = candidate
            break

    assert event is not None
    sm.mark_drink(event, now=event.timestamp)
    timeline.append(sm.tick(now=event.timestamp).mode)

    assert timeline == [
        "TRACKING",
        "ALERT_ESCALATING",
        "ALERT_CONTINUOUS",
        "PAUSED_ABSENT",
        "TRACKING",
    ]
