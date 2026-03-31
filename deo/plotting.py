"""Optional Plotly-based live plotting for optimizer convergence."""

from __future__ import annotations

from pathlib import Path
from time import monotonic, sleep
from typing import Any

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
        self.figure.update_layout(
            title=f"Differential Evolution Convergence ({self.mode.title()})",
            xaxis_title="Generation",
            yaxis_title="Evaluation",
        )
        self.enabled = True

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

        self._generations.append(int(generation))
        self._best_history.append(float(best_evaluation))
        self._mean_history.append(float(mean_evaluation))

        if not self.enabled or self.figure is None:
            return

        self.figure.data[0].x = tuple(self._generations)
        self.figure.data[0].y = tuple(self._best_history)
        self.figure.data[1].x = tuple(self._generations)
        self.figure.data[1].y = tuple(self._mean_history)

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
        snapshot.write_html(str(path), auto_open=False)
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
        figure.update_layout(
            title=f"Differential Evolution Convergence ({self.mode.title()})",
            xaxis_title="Generation",
            yaxis_title="Evaluation",
        )
        return figure