from __future__ import annotations

from collections.abc import Callable
import re

try:
    import rumps  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    rumps = None

try:
    import AppKit  # type: ignore
    import Foundation  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    AppKit = None
    Foundation = None


StatusProvider = Callable[[], tuple[str, str]]
Action = Callable[[], None]


class JalLijiyeMenuBar:
    def __init__(
        self,
        get_status: StatusProvider,
        on_mark_drink: Action,
        on_pause: Action,
        on_quit: Action,
    ) -> None:
        if rumps is None:
            raise RuntimeError("rumps is required for menu bar mode")

        self._get_status = get_status
        self._on_mark_drink = on_mark_drink
        self._on_pause = on_pause
        self._on_quit = on_quit

        self.app = rumps.App("JalLijiye", title="H2O", quit_button=None)

        self.count_item = rumps.MenuItem("Hydration: 0/8")
        self.mark_item = rumps.MenuItem("Mark Drink Now", callback=self._mark_drink)
        self.pause_item = rumps.MenuItem("Pause 30m", callback=self._pause)
        self.quit_item = rumps.MenuItem("Quit", callback=self._quit)

        self.app.menu = [
            self.count_item,
            None,
            self.mark_item,
            self.pause_item,
            None,
            self.quit_item,
        ]

        self.timer = rumps.Timer(self._refresh, 0.2)
        self.timer.start()

    def _refresh(self, _: object) -> None:
        title, counter_text = self._get_status()
        self.count_item.title = counter_text
        self._apply_title(title)

    def _mark_drink(self, _: object) -> None:
        self._on_mark_drink()

    def _pause(self, _: object) -> None:
        self._on_pause()

    def _quit(self, _: object) -> None:
        self._on_quit()
        rumps.quit_application()

    def run(self) -> None:
        self.app.run()

    def _apply_title(self, title: str) -> None:
        if title.startswith("Drink after:") or title == "Drink right now":
            if self._set_countdown_badge_title(title):
                return

        self._clear_attributed_title()
        self.app.title = title

    def _set_countdown_badge_title(self, title: str) -> bool:
        if AppKit is None or Foundation is None:
            return False
        nsapp = getattr(self.app, "_nsapp", None)
        status_item = getattr(nsapp, "nsstatusitem", None)
        if status_item is None:
            return False
        button = status_item.button()
        if button is None:
            return False

        self.app.title = ""
        self.app.icon = None

        font = AppKit.NSFont.boldSystemFontOfSize_(13.0)
        paragraph = AppKit.NSMutableParagraphStyle.alloc().init()
        paragraph.setAlignment_(AppKit.NSTextAlignmentCenter)
        background_color = self._countdown_color(title)
        attrs = {
            AppKit.NSFontAttributeName: font,
            AppKit.NSForegroundColorAttributeName: AppKit.NSColor.whiteColor(),
            AppKit.NSBackgroundColorAttributeName: background_color,
            AppKit.NSParagraphStyleAttributeName: paragraph,
        }
        text = f" {title} "
        attributed = Foundation.NSMutableAttributedString.alloc().initWithString_(text)
        attributed.addAttributes_range_(attrs, (0, len(text)))
        button.setAttributedTitle_(attributed)
        return True

    def _clear_attributed_title(self) -> None:
        if AppKit is None:
            return
        nsapp = getattr(self.app, "_nsapp", None)
        status_item = getattr(nsapp, "nsstatusitem", None)
        if status_item is None:
            return
        button = status_item.button()
        if button is None:
            return
        empty = getattr(Foundation, "NSAttributedString", None)
        if empty is not None:
            button.setAttributedTitle_(empty.alloc().initWithString_(""))

    def _countdown_color(self, title: str):  # type: ignore[no-untyped-def]
        if AppKit is None:
            return None
        if title == "Drink right now" or "<1m" in title:
            return AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.84, 0.16, 0.16, 1.0)
        if "<1m" in title:
            return AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.84, 0.16, 0.16, 1.0)

        match = re.search(r"(\d+)m", title)
        minutes = int(match.group(1)) if match else 0
        if minutes > 5:
            return AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.19, 0.60, 0.25, 1.0)
        if 1 <= minutes <= 5:
            return AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.90, 0.69, 0.13, 1.0)
        return AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.84, 0.16, 0.16, 1.0)
