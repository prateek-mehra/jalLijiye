from __future__ import annotations

from collections.abc import Callable
import logging

try:
    import AppKit  # type: ignore
    import Foundation  # type: ignore
    import objc  # type: ignore
except Exception:  # pragma: no cover - optional outside macOS
    AppKit = None
    Foundation = None
    objc = None


if Foundation is not None:

    class _WakeNotificationObserver(Foundation.NSObject):
        def initWithCallback_(self, callback: Callable[[], None]):
            self = objc.super(_WakeNotificationObserver, self).init()
            if self is None:
                return None
            self._callback = callback
            return self

        def systemDidWake_(self, notification: object) -> None:
            self._callback()

else:
    _WakeNotificationObserver = None


class SystemWakeObserver:
    """Call the supplied handler when macOS wakes from system sleep."""

    def __init__(self, callback: Callable[[], None]) -> None:
        self._center = None
        self._observer = None
        if AppKit is None or Foundation is None or _WakeNotificationObserver is None:
            logging.warning("System wake notifications are unavailable on this platform")
            return

        self._center = AppKit.NSWorkspace.sharedWorkspace().notificationCenter()
        self._observer = _WakeNotificationObserver.alloc().initWithCallback_(callback)
        self._center.addObserver_selector_name_object_(
            self._observer,
            "systemDidWake:",
            AppKit.NSWorkspaceDidWakeNotification,
            None,
        )
        logging.info("Registered for macOS system-wake notifications")

    def close(self) -> None:
        if self._center is not None and self._observer is not None:
            self._center.removeObserver_(self._observer)
            self._center = None
            self._observer = None
