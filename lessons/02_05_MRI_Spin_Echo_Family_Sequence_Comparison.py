import tkinter as tk
from tkinter import ttk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyBboxPatch


# ============================================================
# Spin-Echo Family Animation
#
# Panels:
#   1. Spin Echo
#   2. Inversion-Recovery Spin Echo
#   3. Inversion-Recovery Fast Spin Echo
#
# Run:
#   pip install numpy matplotlib
#   python spin_echo_family_comparison_animation.py
# ============================================================


class SpinEchoFamilyAnimation:
    TIMER_MS = 30

    def __init__(self, root):
        self.root = root
        self.root.title("Spin-Echo Family")
        self.root.geometry("1550x900")

        # --------------------------------------------------------
        # Control variables
        # --------------------------------------------------------
        self.speed_var = tk.DoubleVar(value=1.0)
        self.ti_var = tk.DoubleVar(value=1.10)
        self.te_var = tk.DoubleVar(value=1.20)
        self.echo_spacing_var = tk.DoubleVar(value=0.48)
        self.etl_var = tk.IntVar(value=5)
        self.auto_restart_var = tk.BooleanVar(value=True)

        # --------------------------------------------------------
        # Animation state
        # --------------------------------------------------------
        self.running = False
        self.t = 0.0
        self.ui_ready = False

        self._build_ui()

        self.ui_ready = True
        self.update_labels()
        self.redraw_all()
        self._animation_loop()

    # ============================================================
    # UI
    # ============================================================

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        main.columnconfigure(0, weight=0, minsize=275)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        controls = ttk.Frame(
            main,
            padding=12,
            width=275
        )
        controls.grid(
            row=0,
            column=0,
            sticky="nsew"
        )
        controls.grid_propagate(False)

        plot_frame = ttk.Frame(main)
        plot_frame.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        ttk.Label(
            controls,
            text="Spin-Echo Family",
            font=("Arial", 17, "bold")
        ).pack(anchor="w", pady=(0, 12))

        self.start_button = ttk.Button(
            controls,
            text="Start",
            command=self.toggle_animation
        )
        self.start_button.pack(fill=tk.X, pady=3)

        ttk.Button(
            controls,
            text="Restart",
            command=self.restart
        ).pack(fill=tk.X, pady=3)

        ttk.Button(
            controls,
            text="Reset",
            command=self.reset
        ).pack(fill=tk.X, pady=3)

        ttk.Checkbutton(
            controls,
            text="Auto restart",
            variable=self.auto_restart_var
        ).pack(anchor="w", pady=(7, 3))

        ttk.Separator(controls).pack(
            fill=tk.X,
            pady=10
        )

        self._add_slider(
            controls,
            "Speed",
            self.speed_var,
            0.25,
            3.0,
            "{:.2f}×"
        )

        self._add_slider(
            controls,
            "TI",
            self.ti_var,
            0.50,
            2.20,
            "{:.2f} s"
        )

        self._add_slider(
            controls,
            "TE",
            self.te_var,
            0.70,
            1.80,
            "{:.2f} s"
        )

        self._add_slider(
            controls,
            "Echo spacing",
            self.echo_spacing_var,
            0.30,
            0.70,
            "{:.2f} s"
        )

        self._add_int_slider(
            controls,
            "Echo train length",
            self.etl_var,
            3,
            8
        )

        ttk.Separator(controls).pack(
            fill=tk.X,
            pady=12
        )

        self.time_label = ttk.Label(
            controls,
            text="t = 0.00 s",
            font=("Arial", 11)
        )
        self.time_label.pack(anchor="w")

        # --------------------------------------------------------
        # Figure
        # --------------------------------------------------------
        self.fig = Figure(
            figsize=(12.8, 8.3),
            dpi=100,
            facecolor="white"
        )

        grid = self.fig.add_gridspec(
            3,
            1,
            left=0.07,
            right=0.98,
            top=0.93,
            bottom=0.07,
            hspace=0.42
        )

        self.ax_se = self.fig.add_subplot(grid[0, 0])
        self.ax_ir_se = self.fig.add_subplot(grid[1, 0])
        self.ax_ir_fse = self.fig.add_subplot(grid[2, 0])

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=plot_frame
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

    def _add_slider(
        self,
        parent,
        name,
        variable,
        minimum,
        maximum,
        value_format
    ):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=6)

        label = ttk.Label(frame)
        label.pack(anchor="w")

        variable._display_label = label
        variable._display_name = name
        variable._display_format = value_format

        def changed(_value=None):
            self.update_labels()

            if self.ui_ready:
                self.redraw_all()

        slider = ttk.Scale(
            frame,
            from_=minimum,
            to=maximum,
            variable=variable,
            orient=tk.HORIZONTAL,
            command=changed
        )
        slider.pack(fill=tk.X)

    def _add_int_slider(
        self,
        parent,
        name,
        variable,
        minimum,
        maximum
    ):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=6)

        label = ttk.Label(
            frame,
            text=f"{name}: {int(variable.get())}"
        )
        label.pack(anchor="w")

        variable._display_label = label
        variable._display_name = name

        callback_lock = {"active": False}

        def changed(value):
            if callback_lock["active"]:
                return

            callback_lock["active"] = True

            try:
                rounded_value = int(round(float(value)))
                rounded_value = max(
                    minimum,
                    min(maximum, rounded_value)
                )

                variable.set(rounded_value)

                label.config(
                    text=f"{name}: {rounded_value}"
                )

                if self.ui_ready:
                    self.redraw_all()

            finally:
                callback_lock["active"] = False

        slider = ttk.Scale(
            frame,
            from_=minimum,
            to=maximum,
            variable=variable,
            orient=tk.HORIZONTAL,
            command=changed
        )
        slider.pack(fill=tk.X)

    def update_labels(self):
        for variable in [
            self.speed_var,
            self.ti_var,
            self.te_var,
            self.echo_spacing_var
        ]:
            if hasattr(variable, "_display_label"):
                variable._display_label.config(
                    text=(
                        f"{variable._display_name}: "
                        f"{variable._display_format.format(variable.get())}"
                    )
                )

        if hasattr(self.etl_var, "_display_label"):
            self.etl_var._display_label.config(
                text=(
                    f"{self.etl_var._display_name}: "
                    f"{int(self.etl_var.get())}"
                )
            )

    # ============================================================
    # Timing
    # ============================================================

    def spin_echo_timing(self):
        pulse_90 = 0.45
        te = float(self.te_var.get())

        pulse_180 = pulse_90 + te / 2.0
        echo = pulse_90 + te
        end = echo + 0.55

        return {
            "90": pulse_90,
            "180": pulse_180,
            "echo": echo,
            "end": end
        }

    def ir_spin_echo_timing(self):
        inversion = 0.30
        excitation = inversion + float(self.ti_var.get())
        te = float(self.te_var.get())

        refocus = excitation + te / 2.0
        echo = excitation + te
        end = echo + 0.55

        return {
            "inv": inversion,
            "90": excitation,
            "180": refocus,
            "echo": echo,
            "end": end
        }

    def ir_fse_timing(self):
        inversion = 0.30
        excitation = inversion + float(self.ti_var.get())

        spacing = float(
            self.echo_spacing_var.get()
        )

        echo_count = max(
            3,
            int(self.etl_var.get())
        )

        refocus_times = [
            excitation + spacing * (index + 0.5)
            for index in range(echo_count)
        ]

        echo_times = [
            excitation + spacing * (index + 1.0)
            for index in range(echo_count)
        ]

        end = echo_times[-1] + 0.50

        return {
            "inv": inversion,
            "90": excitation,
            "refocus": refocus_times,
            "echo": echo_times,
            "end": end
        }

    def sequence_end(self):
        return max(
            self.spin_echo_timing()["end"],
            self.ir_spin_echo_timing()["end"],
            self.ir_fse_timing()["end"]
        )

    # ============================================================
    # Animation controls
    # ============================================================

    def toggle_animation(self):
        if self.t >= self.sequence_end():
            self.t = 0.0

        self.running = not self.running

        if self.running:
            self.start_button.config(text="Pause")
        else:
            self.start_button.config(text="Start")

    def restart(self):
        self.t = 0.0
        self.running = True
        self.start_button.config(text="Pause")
        self.redraw_all()

    def reset(self):
        self.t = 0.0
        self.running = False
        self.start_button.config(text="Start")
        self.redraw_all()

    def _animation_loop(self):
        if self.running:
            self.t += (
                0.018
                * float(self.speed_var.get())
            )

            if self.t >= self.sequence_end():
                if self.auto_restart_var.get():
                    self.t = 0.0
                else:
                    self.t = self.sequence_end()
                    self.running = False
                    self.start_button.config(text="Start")

            self.redraw_all()

        self.root.after(
            self.TIMER_MS,
            self._animation_loop
        )

    # ============================================================
    # Drawing helpers
    # ============================================================

    @staticmethod
    def draw_pulse(
        ax,
        center,
        base,
        width,
        height,
        color,
        label,
        active=False
    ):
        pulse = FancyBboxPatch(
            (
                center - width / 2.0,
                base
            ),
            width,
            height,
            boxstyle=(
                "round,pad=0.01,"
                "rounding_size=0.025"
            ),
            facecolor=color,
            edgecolor=(
                "gold"
                if active
                else color
            ),
            linewidth=(
                3.0
                if active
                else 1.0
            ),
            alpha=0.95
        )

        ax.add_patch(pulse)

        ax.text(
            center,
            base + height / 2.0,
            label,
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold"
        )

    @staticmethod
    def draw_echo(
        ax,
        center,
        baseline,
        amplitude,
        width,
        color,
        visible_fraction=1.0
    ):
        x = np.linspace(
            center - width,
            center + width,
            180
        )

        envelope = np.exp(
            -(
                (x - center)
                / (width * 0.40)
            ) ** 2
        )

        y = (
            baseline
            + amplitude
            * envelope
            * np.sin(
                2 * np.pi
                * 11
                * (x - center)
            )
        )

        visible_fraction = np.clip(
            visible_fraction,
            0.0,
            1.0
        )

        visible_count = max(
            1,
            int(len(x) * visible_fraction)
        )

        ax.plot(
            x[:visible_count],
            y[:visible_count],
            color=color,
            linewidth=2.1
        )

    def setup_axis(
        self,
        ax,
        title,
        end_time
    ):
        ax.clear()

        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
            pad=8
        )

        ax.set_xlim(
            -0.05 * end_time,
            end_time
        )

        ax.set_ylim(
            0.0,
            3.0
        )

        ax.set_xlabel("Time")
        ax.set_yticks([])

        ax.grid(
            True,
            axis="x",
            alpha=0.16
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.hlines(
            2.20,
            0,
            end_time,
            color="black",
            linewidth=1.0,
            alpha=0.55
        )

        ax.hlines(
            0.85,
            0,
            end_time,
            color="lightgray",
            linewidth=1.0
        )

        ax.axvline(
            self.t,
            color="crimson",
            linewidth=2.0,
            alpha=0.88
        )

    @staticmethod
    def draw_interval(
        ax,
        start,
        end,
        y,
        label,
        color
    ):
        ax.annotate(
            "",
            xy=(end, y),
            xytext=(start, y),
            arrowprops=dict(
                arrowstyle="<->",
                color=color,
                linewidth=1.5
            )
        )

        ax.text(
            (start + end) / 2.0,
            y + 0.07,
            label,
            color=color,
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    @staticmethod
    def pulse_is_active(
        current_time,
        pulse_time,
        window=0.08
    ):
        return abs(
            current_time - pulse_time
        ) <= window

    @staticmethod
    def echo_visibility(
        current_time,
        echo_time,
        width
    ):
        start = echo_time - width

        if current_time <= start:
            return 0.0

        if current_time >= echo_time + width:
            return 1.0

        return (
            current_time - start
        ) / (2.0 * width)

    # ============================================================
    # Panel 1: Spin Echo
    # ============================================================

    def draw_spin_echo(self):
        ax = self.ax_se
        timing = self.spin_echo_timing()

        self.setup_axis(
            ax,
            "Spin Echo",
            timing["end"]
        )

        self.draw_pulse(
            ax,
            timing["90"],
            2.20,
            0.18,
            0.48,
            "royalblue",
            "90°",
            active=self.pulse_is_active(
                self.t,
                timing["90"]
            )
        )

        self.draw_pulse(
            ax,
            timing["180"],
            2.20,
            0.20,
            0.48,
            "crimson",
            "180°",
            active=self.pulse_is_active(
                self.t,
                timing["180"]
            )
        )

        echo_width = 0.22

        echo_amplitude = (
            0.62
            * np.exp(
                -float(self.te_var.get()) / 2.8
            )
        )

        visibility = self.echo_visibility(
            self.t,
            timing["echo"],
            echo_width
        )

        self.draw_echo(
            ax,
            timing["echo"],
            0.85,
            echo_amplitude,
            echo_width,
            "goldenrod",
            visibility
        )

        self.draw_interval(
            ax,
            timing["90"],
            timing["echo"],
            0.25,
            "TE",
            "royalblue"
        )

        if abs(self.t - timing["echo"]) < 0.10:
            ax.scatter(
                [timing["echo"]],
                [0.85],
                s=180,
                facecolors="none",
                edgecolors="goldenrod",
                linewidths=3
            )

    # ============================================================
    # Panel 2: Inversion-Recovery Spin Echo
    # ============================================================

    def draw_ir_spin_echo(self):
        ax = self.ax_ir_se
        timing = self.ir_spin_echo_timing()

        self.setup_axis(
            ax,
            "Inversion-Recovery Spin Echo",
            timing["end"]
        )

        self.draw_pulse(
            ax,
            timing["inv"],
            2.20,
            0.22,
            0.48,
            "darkorange",
            "180°",
            active=self.pulse_is_active(
                self.t,
                timing["inv"]
            )
        )

        self.draw_pulse(
            ax,
            timing["90"],
            2.20,
            0.18,
            0.48,
            "royalblue",
            "90°",
            active=self.pulse_is_active(
                self.t,
                timing["90"]
            )
        )

        self.draw_pulse(
            ax,
            timing["180"],
            2.20,
            0.20,
            0.48,
            "crimson",
            "180°",
            active=self.pulse_is_active(
                self.t,
                timing["180"]
            )
        )

        ti = float(
            self.ti_var.get()
        )

        t1 = 1.0

        mz_at_ti = (
            1.0
            - 2.0
            * np.exp(-ti / t1)
        )

        echo_amplitude = (
            abs(mz_at_ti)
            * np.exp(
                -float(self.te_var.get()) / 2.8
            )
        )

        echo_width = 0.22

        visibility = self.echo_visibility(
            self.t,
            timing["echo"],
            echo_width
        )

        self.draw_echo(
            ax,
            timing["echo"],
            0.85,
            echo_amplitude,
            echo_width,
            "goldenrod",
            visibility
        )

        self.draw_interval(
            ax,
            timing["inv"],
            timing["90"],
            1.42,
            "TI",
            "darkorange"
        )

        self.draw_interval(
            ax,
            timing["90"],
            timing["echo"],
            0.25,
            "TE",
            "royalblue"
        )

        ax.text(
            timing["inv"],
            2.82,
            "INV",
            color="darkorange",
            fontsize=9,
            fontweight="bold",
            ha="center"
        )

        ax.text(
            timing["180"],
            2.82,
            "REFOCUS",
            color="crimson",
            fontsize=9,
            fontweight="bold",
            ha="center"
        )

        if abs(self.t - timing["echo"]) < 0.10:
            ax.scatter(
                [timing["echo"]],
                [0.85],
                s=180,
                facecolors="none",
                edgecolors="goldenrod",
                linewidths=3
            )

    # ============================================================
    # Panel 3: Inversion-Recovery Fast Spin Echo
    # ============================================================

    def draw_ir_fast_spin_echo(self):
        ax = self.ax_ir_fse
        timing = self.ir_fse_timing()

        self.setup_axis(
            ax,
            "Inversion-Recovery Fast Spin Echo",
            timing["end"]
        )

        self.draw_pulse(
            ax,
            timing["inv"],
            2.20,
            0.22,
            0.48,
            "darkorange",
            "180°",
            active=self.pulse_is_active(
                self.t,
                timing["inv"]
            )
        )

        self.draw_pulse(
            ax,
            timing["90"],
            2.20,
            0.18,
            0.48,
            "royalblue",
            "90°",
            active=self.pulse_is_active(
                self.t,
                timing["90"]
            )
        )

        for refocus_time in timing["refocus"]:
            self.draw_pulse(
                ax,
                refocus_time,
                2.20,
                0.16,
                0.48,
                "crimson",
                "180°",
                active=self.pulse_is_active(
                    self.t,
                    refocus_time,
                    window=0.06
                )
            )

        echo_width = 0.16

        for index, echo_time in enumerate(
            timing["echo"]
        ):
            amplitude = max(
                0.18,
                0.62
                * np.exp(
                    -0.24 * index
                )
            )

            visibility = self.echo_visibility(
                self.t,
                echo_time,
                echo_width
            )

            self.draw_echo(
                ax,
                echo_time,
                0.85,
                amplitude,
                echo_width,
                "mediumorchid",
                visibility
            )

            if abs(self.t - echo_time) < 0.08:
                ax.scatter(
                    [echo_time],
                    [0.85],
                    s=130,
                    facecolors="none",
                    edgecolors="mediumorchid",
                    linewidths=2.5
                )

        self.draw_interval(
            ax,
            timing["inv"],
            timing["90"],
            1.42,
            "TI",
            "darkorange"
        )

        ax.text(
            timing["inv"],
            2.82,
            "INV",
            color="darkorange",
            fontsize=9,
            fontweight="bold",
            ha="center"
        )

        ax.text(
            np.mean(timing["refocus"]),
            2.82,
            "REFOCUS TRAIN",
            color="crimson",
            fontsize=9,
            fontweight="bold",
            ha="center"
        )

    # ============================================================
    # Main redraw
    # ============================================================

    def redraw_all(self):
        if not self.ui_ready:
            return

        self.update_labels()

        self.draw_spin_echo()
        self.draw_ir_spin_echo()
        self.draw_ir_fast_spin_echo()

        self.time_label.config(
            text=f"t = {self.t:.2f} s"
        )

        self.fig.suptitle(
            "Spin-Echo Family",
            fontsize=16,
            fontweight="bold"
        )

        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    SpinEchoFamilyAnimation(root)
    root.mainloop()


if __name__ == "__main__":
    main()