"""Optional Plotly-based live plotting for optimizer convergence."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from typing import Any
import webbrowser

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - depends on optional dependency
    go = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - depends on optional dependency
    Image = None
    ImageDraw = None
    ImageFont = None


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
        self.gif_output_path: Path | None = None
        self.gif_frame_duration_seconds = 2.0
        self.gif_max_frames = 10
        self._last_live_publish_at = 0.0
        self._live_browser_opened = False
        self._live_browser_launch_started = False
        self.gif_disabled_reason: str | None = None
        self._gif_target_generations: list[int] = []
        self._gif_checkpoint_generations: list[int] = []
        self._gif_target_cursor = 0

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
        self._reset_gif_targets()
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
        self._record_gif_checkpoint(generation)

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

    def configure_gif_capture(
        self,
        output_path: str | Path,
        frame_duration_seconds: float = 2.0,
        max_frames: int = 10,
    ) -> None:
        """Configure animated GIF capture for live convergence history.

        Args:
            output_path: GIF file written from captured progress frames.
            frame_duration_seconds: Playback duration per GIF frame.
            max_frames: Maximum number of evenly spaced frames to include.

        Raises:
            ValueError: If the frame duration is not positive or max_frames is invalid.
        """

        if frame_duration_seconds <= 0:
            raise ValueError("frame_duration_seconds must be greater than 0.")
        if max_frames < 2:
            raise ValueError("max_frames must be at least 2.")

        self.gif_output_path = Path(output_path)
        self.gif_frame_duration_seconds = float(frame_duration_seconds)
        self.gif_max_frames = int(max_frames)
        self.gif_disabled_reason = None
        self._reset_gif_targets()

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
            self._record_gif_checkpoint(self._current_generation, force=True)
            self._write_gif()
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

    def _render_gif_frame(self, generation: int) -> Any | None:
        """Render the recorded convergence history to a static image frame."""

        if Image is None or ImageDraw is None or ImageFont is None:
            return None

        history_limit = bisect_right(self._generations, generation)
        generations = self._generations[:history_limit]
        best_history = self._best_history[:history_limit]
        mean_history = self._mean_history[:history_limit]

        width = 1000
        height = 600
        margin_left = 90
        margin_right = 40
        margin_top = 70
        margin_bottom = 70
        plot_left = margin_left
        plot_right = width - margin_right
        plot_top = margin_top
        plot_bottom = height - margin_bottom

        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        title = (
            f"Differential Evolution Convergence ({self.mode.title()})"
            f" - Generation {generation}/"
            f"{self._total_generations if self._total_generations is not None else '?'}"
        )
        draw.text((margin_left, 20), title, fill="black", font=font)

        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="black", width=2)

        if not generations:
            return image

        y_values = best_history + mean_history
        y_min = min(y_values)
        y_max = max(y_values)
        if y_min == y_max:
            padding = max(1.0, abs(y_min) * 0.05, 1e-6)
            y_min -= padding
            y_max += padding
        else:
            padding = (y_max - y_min) * 0.1
            y_min -= padding
            y_max += padding

        x_min = generations[0]
        x_max = generations[-1]

        def x_to_px(value: int) -> float:
            if x_max == x_min:
                return float((plot_left + plot_right) / 2)
            return plot_left + (value - x_min) * (plot_right - plot_left) / (x_max - x_min)

        def y_to_px(value: float) -> float:
            return plot_bottom - (value - y_min) * (plot_bottom - plot_top) / (y_max - y_min)

        for step in range(6):
            y_value = y_min + step * (y_max - y_min) / 5
            y_pixel = y_to_px(y_value)
            draw.line((plot_left, y_pixel, plot_right, y_pixel), fill="#dddddd", width=1)
            draw.text((10, y_pixel - 6), f"{y_value:.3g}", fill="black", font=font)

        for step in range(6):
            x_value = int(round(x_min + step * (x_max - x_min) / 5))
            x_pixel = x_to_px(x_value)
            draw.line((x_pixel, plot_top, x_pixel, plot_bottom), fill="#f0f0f0", width=1)
            draw.text((x_pixel - 10, plot_bottom + 10), str(x_value), fill="black", font=font)

        best_points = [
            (x_to_px(generation), y_to_px(value))
            for generation, value in zip(generations, best_history, strict=False)
        ]
        mean_points = [
            (x_to_px(generation), y_to_px(value))
            for generation, value in zip(generations, mean_history, strict=False)
        ]

        if len(best_points) > 1:
            draw.line(best_points, fill="#1f77b4", width=3)
        elif best_points:
            x_pixel, y_pixel = best_points[0]
            draw.ellipse((x_pixel - 3, y_pixel - 3, x_pixel + 3, y_pixel + 3), fill="#1f77b4")

        if len(mean_points) > 1:
            draw.line(mean_points, fill="#ff7f0e", width=3)
        elif mean_points:
            x_pixel, y_pixel = mean_points[0]
            draw.ellipse((x_pixel - 3, y_pixel - 3, x_pixel + 3, y_pixel + 3), fill="#ff7f0e")

        legend_top = plot_top + 10
        legend_left = plot_right - 200
        draw.line((legend_left, legend_top, legend_left + 25, legend_top), fill="#1f77b4", width=3)
        draw.text((legend_left + 35, legend_top - 8), "Best Evaluation", fill="black", font=font)
        draw.line((legend_left, legend_top + 25, legend_left + 25, legend_top + 25), fill="#ff7f0e", width=3)
        draw.text((legend_left + 35, legend_top + 17), "Mean Evaluation", fill="black", font=font)

        draw.text((width / 2 - 35, height - 30), "Generation", fill="black", font=font)
        draw.text((15, plot_top - 25), "Evaluation", fill="black", font=font)

        return image

    def _write_gif(self) -> bool:
        """Write the captured frames to an animated GIF file."""

        if self.gif_output_path is None or not self._gif_checkpoint_generations:
            return False

        if Image is None:
            self.gif_disabled_reason = "Pillow is not installed in the current environment."
            return False

        images: list[Any] = []
        try:
            for generation in self._gif_checkpoint_generations:
                frame = self._render_gif_frame(generation)
                if frame is None:
                    return False
                images.append(frame.convert("P", palette=Image.Palette.ADAPTIVE))

            self.gif_output_path.parent.mkdir(parents=True, exist_ok=True)
            images[0].save(
                self.gif_output_path,
                save_all=True,
                append_images=images[1:],
                duration=max(int(self.gif_frame_duration_seconds * 1000), 1),
                loop=0,
                disposal=2,
            )
        except Exception as exc:  # pragma: no cover - filesystem/encoder dependent
            self.gif_disabled_reason = str(exc)
            return False
        finally:
            for image in images:
                image.close()

        self.gif_disabled_reason = None
        return True

    def _reset_gif_targets(self) -> None:
        """Reset GIF checkpoint planning based on the current configuration."""

        self._gif_checkpoint_generations = []
        self._gif_target_cursor = 0
        self._gif_target_generations = []

        if self.gif_output_path is None or self._total_generations is None:
            return

        max_generation = self._total_generations
        frame_count = min(self.gif_max_frames, max_generation + 1)
        if frame_count < 2:
            frame_count = 2

        targets = {
            int(round(step * max_generation / (frame_count - 1)))
            for step in range(frame_count)
        }
        targets.add(0)
        targets.add(max_generation)
        self._gif_target_generations = sorted(targets)

    def _record_gif_checkpoint(self, generation: int, force: bool = False) -> None:
        """Record evenly spaced GIF checkpoints with negligible optimizer overhead."""

        if self.gif_output_path is None:
            return

        if not self._gif_target_generations:
            self._reset_gif_targets()

        if force:
            if not self._gif_checkpoint_generations or self._gif_checkpoint_generations[-1] != int(generation):
                self._gif_checkpoint_generations.append(int(generation))

        while self._gif_target_cursor < len(self._gif_target_generations):
            target_generation = self._gif_target_generations[self._gif_target_cursor]
            if target_generation > generation:
                break

            if (
                not self._gif_checkpoint_generations
                or self._gif_checkpoint_generations[-1] != target_generation
            ):
                self._gif_checkpoint_generations.append(target_generation)
            self._gif_target_cursor += 1

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