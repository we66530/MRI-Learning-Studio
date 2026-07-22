"""
Runner analogy for MRI T2 relaxation and T2* refocusing

Left panel:
    True T2 relaxation
    - Runners receive random phase disturbances.
    - Phase loss is not fully reversible.

Right panel:
    Static B0 inhomogeneity
    - Each runner has a fixed speed offset.
    - A 180-degree RF pulse reverses the phase distribution.
    - Runners rephase and form a spin echo.

Controls:
    Space : Pause / resume
    R     : Restart
    Esc   : Close
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Arc


# ============================================================
# Animation settings
# ============================================================

N_RUNNERS = 12

DT = 0.035
TOTAL_TIME = 12.0

RF180_TIME = 5.0
ECHO_TIME = 2.0 * RF180_TIME

TRACK_RADIUS = 1.0
RUNNER_RADIUS = 0.055

# Overall angular velocity shared by all runners.
BASE_OMEGA = 1.25

# Fixed speed differences caused by static field inhomogeneity.
STATIC_OFFSETS = np.linspace(-0.28, 0.28, N_RUNNERS)

# True T2 random interaction strength.
RANDOM_INTERACTION_STRENGTH = 0.24

# Reproducible random animation.
RNG_SEED = 7


# ============================================================
# Utility functions
# ============================================================

def circular_mean_resultant(phases: np.ndarray) -> tuple[float, float]:
    """
    Return:
        mean_phase:
            Direction of net transverse magnetization.

        coherence:
            Length of normalized vector sum, between 0 and 1.
            1 = perfectly synchronized
            0 = completely cancelled
    """
    complex_sum = np.mean(np.exp(1j * phases))
    return float(np.angle(complex_sum)), float(np.abs(complex_sum))


def wrap_phase(phases: np.ndarray) -> np.ndarray:
    """Wrap phases to the range [-pi, pi)."""
    return (phases + np.pi) % (2.0 * np.pi) - np.pi


def runner_positions(phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert runner phases into XY positions on a circular track."""
    x = TRACK_RADIUS * np.cos(phases)
    y = TRACK_RADIUS * np.sin(phases)
    return x, y


# ============================================================
# Main animation class
# ============================================================

class RunnerT2RefocusingAnimation:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(RNG_SEED)

        self.fig, self.axes = plt.subplots(
            1,
            2,
            figsize=(14, 7),
        )

        self.ax_t2 = self.axes[0]
        self.ax_static = self.axes[1]

        self.paused = False
        self.rf180_applied = False

        self.time = 0.0

        # Left panel phases:
        # true T2 includes random interaction-related phase changes.
        self.t2_phases = np.zeros(N_RUNNERS, dtype=float)

        # Each left-panel runner also gets a small persistent offset.
        self.t2_offsets = np.linspace(-0.16, 0.16, N_RUNNERS)

        # Right panel:
        # fixed offsets model static B0 inhomogeneity.
        self.static_phases = np.zeros(N_RUNNERS, dtype=float)

        # Histories for the coherence traces.
        self.time_history: list[float] = []
        self.t2_coherence_history: list[float] = []
        self.static_coherence_history: list[float] = []

        self._setup_axes()
        self._create_artists()
        self._connect_events()

        self.animation = FuncAnimation(
            self.fig,
            self._update,
            interval=int(DT * 1000),
            blit=False,
            cache_frame_data=False,
        )

    # ========================================================
    # Setup
    # ========================================================

    def _setup_axes(self) -> None:
        for ax in self.axes:
            ax.set_aspect("equal")
            ax.set_xlim(-1.55, 1.55)
            ax.set_ylim(-1.72, 1.48)
            ax.axis("off")

            # Circular track.
            outer = Circle(
                (0, 0),
                TRACK_RADIUS + 0.15,
                fill=False,
                linewidth=8,
                alpha=0.18,
            )
            inner = Circle(
                (0, 0),
                TRACK_RADIUS - 0.15,
                fill=False,
                linewidth=2,
                linestyle="--",
                alpha=0.35,
            )

            ax.add_patch(outer)
            ax.add_patch(inner)

            # Direction arc.
            direction_arc = Arc(
                (0, 0),
                2.3,
                2.3,
                theta1=20,
                theta2=105,
                linewidth=2,
                alpha=0.5,
            )
            ax.add_patch(direction_arc)

            ax.annotate(
                "",
                xy=(-0.30, 1.10),
                xytext=(-0.48, 1.04),
                arrowprops=dict(
                    arrowstyle="-|>",
                    linewidth=2,
                    alpha=0.5,
                ),
            )

        self.ax_t2.set_title(
            "True T2 relaxation\nrandom microscopic interactions",
            fontsize=15,
        )

        self.ax_static.set_title(
            "Static field inhomogeneity\nreversible with a 180° pulse",
            fontsize=15,
        )

        self.fig.suptitle(
            "Runner analogy for transverse dephasing",
            fontsize=18,
        )

        self.fig.subplots_adjust(
            left=0.04,
            right=0.96,
            bottom=0.08,
            top=0.84,
            wspace=0.16,
        )

    def _create_artists(self) -> None:
        # Runner markers.
        self.t2_runner_scatter = self.ax_t2.scatter(
            np.zeros(N_RUNNERS),
            np.zeros(N_RUNNERS),
            s=180,
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

        self.static_runner_scatter = self.ax_static.scatter(
            np.zeros(N_RUNNERS),
            np.zeros(N_RUNNERS),
            s=180,
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

        # Net transverse magnetization arrows.
        self.t2_net_arrow = self.ax_t2.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                linewidth=5,
            ),
            zorder=5,
        )

        self.static_net_arrow = self.ax_static.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                linewidth=5,
            ),
            zorder=5,
        )

        # Status text.
        self.t2_status = self.ax_t2.text(
            0,
            -1.40,
            "",
            ha="center",
            va="center",
            fontsize=12,
        )

        self.static_status = self.ax_static.text(
            0,
            -1.40,
            "",
            ha="center",
            va="center",
            fontsize=12,
        )

        # Pulse indicator.
        self.pulse_indicator = self.ax_static.text(
            0,
            1.32,
            "",
            ha="center",
            va="center",
            fontsize=15,
            weight="bold",
        )

        # Time display.
        self.time_text = self.fig.text(
            0.5,
            0.035,
            "",
            ha="center",
            fontsize=13,
        )

    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    # ========================================================
    # Simulation logic
    # ========================================================

    def _reset(self) -> None:
        self.rng = np.random.default_rng(RNG_SEED)

        self.time = 0.0
        self.rf180_applied = False

        self.t2_phases[:] = 0.0
        self.static_phases[:] = 0.0

        self.time_history.clear()
        self.t2_coherence_history.clear()
        self.static_coherence_history.clear()

    def _apply_180_pulse(self) -> None:
        """
        Educational phase reversal.

        For transverse phases, reflection around the X-axis can be represented
        by phase -> -phase.

        The runners keep their original speed offsets after the pulse.
        Therefore fixed frequency differences cause them to reconverge.
        """
        self.t2_phases = -self.t2_phases
        self.static_phases = -self.static_phases
        self.rf180_applied = True

    def _advance_simulation(self) -> None:
        self.time += DT

        # ----------------------------------------------------
        # True T2 panel
        # ----------------------------------------------------
        # Persistent phase offsets plus random microscopic
        # interaction kicks.
        random_kicks = self.rng.normal(
            loc=0.0,
            scale=RANDOM_INTERACTION_STRENGTH * np.sqrt(DT),
            size=N_RUNNERS,
        )

        self.t2_phases += (
            BASE_OMEGA
            + self.t2_offsets
        ) * DT

        self.t2_phases += random_kicks

        # ----------------------------------------------------
        # Static field inhomogeneity panel
        # ----------------------------------------------------
        # Each runner keeps a fixed angular velocity.
        self.static_phases += (
            BASE_OMEGA
            + STATIC_OFFSETS
        ) * DT

        # Apply the 180° pulse once.
        if (
            not self.rf180_applied
            and self.time >= RF180_TIME
        ):
            self._apply_180_pulse()

        self.t2_phases = wrap_phase(self.t2_phases)
        self.static_phases = wrap_phase(self.static_phases)

        # Store coherence histories.
        _, t2_coherence = circular_mean_resultant(self.t2_phases)
        _, static_coherence = circular_mean_resultant(self.static_phases)

        self.time_history.append(self.time)
        self.t2_coherence_history.append(t2_coherence)
        self.static_coherence_history.append(static_coherence)

        if self.time >= TOTAL_TIME:
            self._reset()

    # ========================================================
    # Drawing
    # ========================================================

    def _update_runner_panel(
        self,
        phases: np.ndarray,
        scatter,
        net_arrow,
        status_text,
        panel_name: str,
    ) -> None:
        x, y = runner_positions(phases)
        scatter.set_offsets(np.column_stack([x, y]))

        mean_phase, coherence = circular_mean_resultant(phases)

        # Net transverse magnetization.
        arrow_length = 0.82 * coherence
        end_x = arrow_length * np.cos(mean_phase)
        end_y = arrow_length * np.sin(mean_phase)

        net_arrow.xy = (end_x, end_y)
        net_arrow.set_position((0, 0))

        if panel_name == "t2":
            message = (
                f"Coherence: {coherence:.2f}\n"
                "Random interactions prevent full recovery"
            )
        else:
            if self.time < RF180_TIME:
                stage = "Dephasing"
            elif abs(self.time - ECHO_TIME) < 0.32:
                stage = "Spin echo"
            elif self.time < ECHO_TIME:
                stage = "Rephasing"
            else:
                stage = "Dephasing again"

            message = (
                f"Coherence: {coherence:.2f}\n"
                f"{stage}"
            )

        status_text.set_text(message)

    def _update(self, _frame_index: int):
        if not self.paused:
            self._advance_simulation()

        self._update_runner_panel(
            phases=self.t2_phases,
            scatter=self.t2_runner_scatter,
            net_arrow=self.t2_net_arrow,
            status_text=self.t2_status,
            panel_name="t2",
        )

        self._update_runner_panel(
            phases=self.static_phases,
            scatter=self.static_runner_scatter,
            net_arrow=self.static_net_arrow,
            status_text=self.static_status,
            panel_name="static",
        )

        if abs(self.time - RF180_TIME) < 0.20:
            self.pulse_indicator.set_text("180° RF pulse")
        elif abs(self.time - ECHO_TIME) < 0.30:
            self.pulse_indicator.set_text("Echo")
        else:
            self.pulse_indicator.set_text("")

        pause_state = "Paused" if self.paused else "Playing"

        self.time_text.set_text(
            f"Time: {self.time:4.1f} s    "
            f"180° pulse: {RF180_TIME:.1f} s    "
            f"Echo: {ECHO_TIME:.1f} s    "
            f"[{pause_state}]"
        )

        return (
            self.t2_runner_scatter,
            self.static_runner_scatter,
            self.t2_net_arrow,
            self.static_net_arrow,
            self.t2_status,
            self.static_status,
            self.pulse_indicator,
            self.time_text,
        )

    # ========================================================
    # Keyboard controls
    # ========================================================

    def _on_key_press(self, event) -> None:
        if event.key == " ":
            self.paused = not self.paused

        elif event.key in {"r", "R"}:
            self._reset()

        elif event.key == "escape":
            plt.close(self.fig)

    def show(self) -> None:
        plt.show()


def main() -> None:
    app = RunnerT2RefocusingAnimation()
    app.show()


if __name__ == "__main__":
    main()