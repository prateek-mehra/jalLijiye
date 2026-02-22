from app.state_machine import HydrationStateMachine
from app.types import Config, DrinkEvent


def _cfg() -> Config:
    return Config(alert_after_minutes=90, escalating_minutes=3)


def test_transitions_to_escalating_at_90_minutes() -> None:
    sm = HydrationStateMachine(_cfg(), initial_now=0.0)

    state = sm.tick(now=90 * 60)

    assert state.mode == "ALERT_ESCALATING"


def test_escalating_to_continuous_after_3_minutes() -> None:
    sm = HydrationStateMachine(_cfg(), initial_now=0.0)

    sm.tick(now=90 * 60)
    state = sm.tick(now=(90 * 60) + (3 * 60))

    assert state.mode == "ALERT_CONTINUOUS"


def test_drink_event_resets_alert() -> None:
    sm = HydrationStateMachine(_cfg(), initial_now=0.0)

    sm.tick(now=90 * 60)
    state = sm.mark_drink(DrinkEvent(timestamp=91 * 60, source="manual", confidence=1.0), now=91 * 60)

    assert state.mode == "TRACKING"
    assert int(state.minutes_since_drink) == 0


def test_pause_and_resume_on_presence() -> None:
    sm = HydrationStateMachine(_cfg(), initial_now=0.0)

    away_state = sm.set_presence(False, now=60)
    back_state = sm.set_presence(True, now=120)

    assert away_state.mode == "PAUSED_ABSENT"
    assert back_state.mode == "TRACKING"
