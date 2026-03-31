"""Optional Plotly-based live plotting for optimizer convergence."""

from __future__ import annotations

from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from typing import Any
import webbrowser

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - depends on optional dependency
    go = None


class LiveConvergencePlot:
    """Track and optionally display best and mean fitness over time.

    The class degrades gracefully when Plotly or widget support is not
    available. In that case, all update calls become no-ops and optimization can
    proceed without interruption.
    """

    def __init__(self, enabled: bool = True, mode: str = "minimize") -> None:
        """Initialize the plotting helper.

        Args:
            enabled: Whether live plotting should be attempted.
            mode: Optimization direction shown in the figure title.
        """

        self.enabled = False
        self.disabled_reason: str | None = None
        self.figure: Any | None = None
        self.mode = mode
        self._generations: list[int] = []
        self._best_history: list[float] = []
        self._mean_history: list[float] = []
        self._current_generation = 0
        self._total_generations: int | None = None
        self.live_output_path: Path | None = None
        self.live_refresh_interval_seconds = 2.0
        self._last_live_publish_at = 0.0
        self._live_browser_opened = False
        self._live_browser_launch_started = False

        if not enabled:
            self.disabled_reason = "Live plotting disabled by configuration."
            return

        if go is None:
            self.disabled_reason = "Plotly is not installed in the current environment."
            return

        try:
            self.figure = go.FigureWidget()
        except Exception:  # pragma: no cover - backend availability varies
            try:
                self.figure = go.Figure()
            except Exception as exc:  # pragma: no cover - backend availability varies
                self.disabled_reason = str(exc)
                return

        self.figure.add_scatter(name="Best Evaluation", mode="lines", x=[], y=[])
        self.figure.add_scatter(name="Mean Evaluation", mode="lines", x=[], y=[])
        self._apply_title(self.figure)
        self.figure.update_layout(xaxis_title="Generation", yaxis_title="Evaluation")
        self.enabled = True

    def set_total_generations(self, total_generations: int) -> None:
        """Store the total generation count for title rendering."""

        if total_generations < 0:
            raise ValueError("total_generations must be non-negative.")

        self._total_generations = int(total_generations)
        if self.figure is not None:
            self._apply_title(self.figure)

    def update(
        self,
        generation: int,
        best_evaluation: float,
        mean_evaluation: float,
    ) -> None:
        """Record a new convergence data point and refresh the plot if enabled.

        Args:
            generation: Generation index to log.
            best_evaluation: Best objective evaluation observed at this generation.
            mean_evaluation: Mean objective evaluation at this generation.
        """

        self._current_generation = int(generation)
        self._generations.append(int(generation))
        self._best_history.append(float(best_evaluation))
        self._mean_history.append(float(mean_evaluation))

    def configure_live_updates(
        self,
        output_path: str | Path,
        refresh_interval_seconds: float = 2.0,
    ) -> None:
        """Configure periodic HTML publishing for browser-based live updates.

        Args:
            output_path: HTML file that the browser will refresh repeatedly.
            refresh_interval_seconds: Browser refresh cadence for the live file.

        Raises:
            ValueError: If the refresh interval is not positive.
        """

        if refresh_interval_seconds <= 0:
            raise ValueError("refresh_interval_seconds must be greater than 0.")

        self.live_output_path = Path(output_path)
        self.live_refresh_interval_seconds = float(refresh_interval_seconds)
        self._last_live_publish_at = 0.0
        self._live_browser_opened = False
        self._live_browser_launch_started = False

    def publish_live(
        self,
        force: bool = False,
        final: bool = False,
        retry_interval: float = 0.5,
        timeout_seconds: float = 5.0,
    ) -> bool:
        """Write the latest figure state to the live HTML file and open it once.

        Args:
            force: Whether to bypass the refresh timer and publish immediately.
            final: Whether this is the terminal live-plot write. Final writes do
                not include the browser auto-refresh tag.
            retry_interval: Delay between browser open retries.
            timeout_seconds: Maximum time allowed to open the live browser page.

        Returns:
            ``True`` when a live file was written, otherwise ``False``.
        """

        if self.live_output_path is None:
            return False

        now = monotonic()
        if not force and now - self._last_live_publish_at < self.live_refresh_interval_seconds:
            return False

        snapshot = self.to_figure()
        if snapshot is None:
            return False

        self.live_output_path.parent.mkdir(parents=True, exist_ok=True)
        refresh_seconds: float | None = self.live_refresh_interval_seconds
        if final:
            refresh_seconds = None
        html = self._figure_html(snapshot, refresh_seconds=refresh_seconds)
        self.live_output_path.write_text(html, encoding="utf-8")
        self._last_live_publish_at = now

        if final:
            return True

        if not self._live_browser_opened and not self._live_browser_launch_started:
            self._live_browser_launch_started = True
            self._open_live_browser_async(
                retry_interval=retry_interval,
                timeout_seconds=timeout_seconds,
            )

        return True

    def show(
        self,
        renderer: str | None = "browser",
        retry_interval: float = 0.5,
        timeout_seconds: float = 5.0,
    ) -> bool:
        """Render the convergence figure explicitly.

        This is required for plain Python scripts because Plotly figures do not
        auto-display in a terminal the way they do in notebooks.

        Args:
            renderer: Optional Plotly renderer name. ``"browser"`` works well
                for terminal-driven scripts on desktop systems.
            retry_interval: Delay between render retries after a failure.
            timeout_seconds: Maximum amount of time to spend retrying before a
                timeout warning is reported.

        Returns:
            ``True`` when a figure was rendered successfully, otherwise
            ``False``.
        """

        snapshot = self.to_figure()
        if snapshot is None:
            return False

        started_at = monotonic()
        last_error: Exception | None = None
        while monotonic() - started_at <= timeout_seconds:
            try:
                snapshot.show(renderer=renderer)
            except Exception as exc:  # pragma: no cover - renderer availability varies
                last_error = exc
                elapsed = monotonic() - started_at
                if elapsed + retry_interval > timeout_seconds:
                    break
                sleep(retry_interval)
                continue

            self.disabled_reason = None
            return True

        timeout_message = (
            f"Plot rendering timed out after {timeout_seconds:.1f} seconds "
            f"while retrying every {retry_interval:.1f} seconds."
        )
        if last_error is not None:
            timeout_message = f"{timeout_message} Last error: {last_error}"
        self.disabled_reason = timeout_message
        return False

    def save_html(self, output_path: str | Path) -> bool:
        """Write the convergence figure to an HTML file.

        Args:
            output_path: Destination HTML path.

        Returns:
            ``True`` when the file was written successfully, otherwise
            ``False``.
        """

        snapshot = self.to_figure()
        if snapshot is None:
            return False

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._figure_html(snapshot), encoding="utf-8")
        return True

    def to_figure(self) -> Any | None:
        """Build a standalone Plotly figure from the recorded history."""

        if go is None:
            self.disabled_reason = "Plotly is not installed in the current environment."
            return None

        figure = go.Figure()
        figure.add_scatter(
            name="Best Evaluation",
            mode="lines",
            x=tuple(self._generations),
            y=tuple(self._best_history),
        )
        figure.add_scatter(
            name="Mean Evaluation",
            mode="lines",
            x=tuple(self._generations),
            y=tuple(self._mean_history),
        )
        self._apply_title(figure)
        figure.update_layout(xaxis_title="Generation", yaxis_title="Evaluation")
        return figure

    def _apply_title(self, figure: Any) -> None:
        """Apply the current dynamic title to a Plotly figure."""

        total_suffix = "?"
        if self._total_generations is not None:
            total_suffix = str(self._total_generations)

        figure.update_layout(
            title=(
                f"Differential Evolution Convergence ({self.mode.title()})"
                f" - Generation {self._current_generation}/{total_suffix}"
            )
        )

    def _figure_html(self, figure: Any, refresh_seconds: float | None = None) -> str:
        """Render a Plotly figure to HTML, optionally with browser auto-refresh."""

        html = figure.to_html(full_html=True, include_plotlyjs="cdn")
        if refresh_seconds is None:
            return html

        refresh_tag = (
            f'<meta http-equiv="refresh" content="{refresh_seconds:.3f}">'  # noqa: E501
        )
        return html.replace("<head>", f"<head>\n    {refresh_tag}", 1)

    def _open_live_browser(
        self,
        retry_interval: float = 0.5,
        timeout_seconds: float = 5.0,
    ) -> bool:
        """Open the live HTML file in a browser with retry and timeout handling."""

        if self.live_output_path is None:
            return False

        uri = self.live_output_path.resolve().as_uri()
        started_at = monotonic()
        last_error: Exception | None = None
        while monotonic() - started_at <= timeout_seconds:
            try:
                opened = webbrowser.open(uri)
            except Exception as exc:  # pragma: no cover - environment dependent
                last_error = exc
                opened = False

            if opened:
                self.disabled_reason = None
                return True

            elapsed = monotonic() - started_at
            if elapsed + retry_interval > timeout_seconds:
                break
            sleep(retry_interval)

        timeout_message = (
            f"Plot rendering timed out after {timeout_seconds:.1f} seconds "
            f"while retrying every {retry_interval:.1f} seconds."
        )
        if last_error is not None:
            timeout_message = f"{timeout_message} Last error: {last_error}"
        self.disabled_reason = timeout_message
        return False

    def _open_live_browser_async(
        self,
        retry_interval: float = 0.5,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Launch browser opening on a daemon thread so optimization keeps running."""

        def target() -> None:
            opened = self._open_live_browser(
                retry_interval=retry_interval,
                timeout_seconds=timeout_seconds,
            )
            self._live_browser_opened = opened
            self._live_browser_launch_started = opened

        Thread(target=target, daemon=True).start()