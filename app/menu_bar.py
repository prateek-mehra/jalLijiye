from __future__ import annotations

from collections.abc import Callable

try:
    import rumps  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    rumps = None


StatusProvider = Callable[[], tuple[str, int]]
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

        self.count_item = rumps.MenuItem("Hydrations Today: 0")
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
        title, hydration_count = self._get_status()
        self.count_item.title = f"Hydrations Today: {hydration_count}"

        if title.startswith("ALERT"):
            self.app.title = "ALRT"
        elif title.startswith("Paused"):
            self.app.title = "PAUS"
        else:
            self.app.title = "H2O"

    def _mark_drink(self, _: object) -> None:
        self._on_mark_drink()

    def _pause(self, _: object) -> None:
        self._on_pause()

    def _quit(self, _: object) -> None:
        self._on_quit()
        rumps.quit_application()

    def run(self) -> None:
        self.app.run()
