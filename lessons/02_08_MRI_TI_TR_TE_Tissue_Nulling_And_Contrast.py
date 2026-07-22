import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# Interactive MRI Timing / Tissue Nulling Demo
#
# Changes in this version:
# 1. Selected tissue null line becomes bright; other null lines fade.
# 2. Added Start / Pause button to animate the recovery and decay curves.
# 3. Removed Numerical Summary and enlarged the remaining panels.
#
# Run:
#   pip install numpy matplotlib
#   python Interactive_TI_TR_TE_Nulling_v2.py
# ============================================================


class InteractiveTimingNullingDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive MRI Timing: TI / TR / TE Nulling Demo")

        # --------------------------------------------------------
        # Tissue parameters
        # Educational approximate values
        # --------------------------------------------------------
        self.tissues = {
            "Fat": {
                "T1": 250,
                "T2": 70,
                "color": "darkorange",
            },
            "White\nMatter": {
                "T1": 700,
                "T2": 80,
                "color": "forestgreen",
            },
            "Gray\nMatter": {
                "T1": 1000,
                "T2": 100,
                "color": "mediumpurple",
            },
            "CSF": {
                "T1": 2900,
                "T2": 300,
                "color": "royalblue",
            },
        }

        # --------------------------------------------------------
        # Animation state
        # --------------------------------------------------------
        self.animation_running = False
        self.animation_progress = 1.0
        self.animation_step = 0.018
        self.animation_delay_ms = 35

        # --------------------------------------------------------
        # GUI layout
        # --------------------------------------------------------
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(
            side=tk.LEFT,
            fill=tk.Y,
            padx=10,
            pady=10
        )

        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True
        )

        # --------------------------------------------------------
        # Figure layout
        #
        # Top row:
        #   TI recovery | TE decay | final signal
        #
        # Bottom row:
        #   timing diagram across full width
        # --------------------------------------------------------
        self.fig = plt.figure(figsize=(15.5, 8.5))

        gs = self.fig.add_gridspec(
            nrows=2,
            ncols=3,
            height_ratios=[2.6, 1.15],
            hspace=0.42,
            wspace=0.30
        )

        self.ax_ir = self.fig.add_subplot(gs[0, 0])
        self.ax_te = self.fig.add_subplot(gs[0, 1])
        self.ax_signal = self.fig.add_subplot(gs[0, 2])

        self.ax_timing = self.fig.add_subplot(gs[1, :])

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=self.plot_frame
        )
        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        # --------------------------------------------------------
        # Variables
        # --------------------------------------------------------
        self.TI_var = tk.DoubleVar(value=173)
        self.TR_var = tk.DoubleVar(value=2500)
        self.TE_var = tk.DoubleVar(value=20)

        self.target_var = tk.StringVar(value="Fat")

        # --------------------------------------------------------
        # Control panel
        # --------------------------------------------------------
        ttk.Label(
            self.control_frame,
            text="MRI Timing Controls",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(
            self.control_frame,
            text="Target tissue to null"
        ).pack(anchor="w")

        self.target_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.target_var,
            values=list(self.tissues.keys()),
            state="readonly"
        )
        self.target_combo.pack(fill=tk.X, pady=(0, 10))
        self.target_combo.bind(
            "<<ComboboxSelected>>",
            self.on_target_changed
        )

        self.add_slider(
            label="TI / Inversion Time (ms)",
            variable=self.TI_var,
            min_value=0,
            max_value=3000
        )

        self.add_slider(
            label="TR / Repetition Time (ms)",
            variable=self.TR_var,
            min_value=200,
            max_value=7000
        )

        self.add_slider(
            label="TE / Echo Time (ms)",
            variable=self.TE_var,
            min_value=5,
            max_value=250
        )

        ttk.Separator(
            self.control_frame
        ).pack(fill=tk.X, pady=10)

        # --------------------------------------------------------
        # Animation buttons
        # --------------------------------------------------------
        self.start_button = ttk.Button(
            self.control_frame,
            text="Start Animation",
            command=self.toggle_animation
        )
        self.start_button.pack(fill=tk.X, pady=3)

        ttk.Button(
            self.control_frame,
            text="Restart Animation",
            command=self.restart_animation
        ).pack(fill=tk.X, pady=3)

        ttk.Separator(
            self.control_frame
        ).pack(fill=tk.X, pady=10)

        # --------------------------------------------------------
        # Presets
        # --------------------------------------------------------
        ttk.Button(
            self.control_frame,
            text="Preset: STIR / Fat Null",
            command=self.preset_stir
        ).pack(fill=tk.X, pady=3)

        ttk.Button(
            self.control_frame,
            text="Preset: FLAIR / CSF Null",
            command=self.preset_flair
        ).pack(fill=tk.X, pady=3)

        ttk.Button(
            self.control_frame,
            text="Preset: T1-weighted",
            command=self.preset_t1
        ).pack(fill=tk.X, pady=3)

        ttk.Button(
            self.control_frame,
            text="Preset: T2-weighted",
            command=self.preset_t2
        ).pack(fill=tk.X, pady=3)

        ttk.Separator(
            self.control_frame
        ).pack(fill=tk.X, pady=10)

        self.status_label = ttk.Label(
            self.control_frame,
            text="",
            wraplength=300,
            justify=tk.LEFT
        )
        self.status_label.pack(anchor="w", pady=5)

        ttk.Separator(
            self.control_frame
        ).pack(fill=tk.X, pady=10)

        note = (
            "Core concept:\n\n"
            "RF frequency remains near the Larmor frequency.\n\n"
            "TI selects the tissue whose longitudinal magnetization "
            "is near zero at the 90° excitation pulse.\n\n"
            "TR controls T1 recovery and repeated-excitation effects.\n\n"
            "TE controls T2 decay before signal readout."
        )

        ttk.Label(
            self.control_frame,
            text=note,
            wraplength=300,
            justify=tk.LEFT
        ).pack(anchor="w", pady=5)

        self.update_plot()

    # ============================================================
    # GUI helpers
    # ============================================================

    def add_slider(
        self,
        label,
        variable,
        min_value,
        max_value
    ):
        frame = ttk.Frame(self.control_frame)
        frame.pack(fill=tk.X, pady=6)

        ttk.Label(
            frame,
            text=label
        ).pack(anchor="w")

        value_label = ttk.Label(
            frame,
            text=f"{variable.get():.0f}"
        )
        value_label.pack(anchor="e")

        def slider_changed(_):
            value_label.config(
                text=f"{variable.get():.0f}"
            )
            self.update_plot()

        slider = ttk.Scale(
            frame,
            from_=min_value,
            to=max_value,
            orient=tk.HORIZONTAL,
            variable=variable,
            command=slider_changed
        )
        slider.pack(fill=tk.X)

    def on_target_changed(self, _event=None):
        self.update_plot()

    # ============================================================
    # MRI signal model
    # ============================================================

    def inversion_recovery_mz(
        self,
        TI,
        T1,
        TR
    ):
        """
        Simplified inversion recovery model.

        Ideal inversion recovery:
            Mz(TI) = 1 - 2 exp(-TI / T1)

        A finite-TR scaling term is included for educational display.
        """

        ideal_ir = 1 - 2 * np.exp(-TI / T1)
        tr_recovery = 1 - np.exp(-TR / T1)

        return ideal_ir * tr_recovery

    def t2_decay(
        self,
        TE,
        T2
    ):
        return np.exp(-TE / T2)

    def final_signal(
        self,
        TI,
        TR,
        TE,
        T1,
        T2
    ):
        """
        Simplified magnitude signal:
            |Mz(TI)| × exp(-TE/T2)
        """

        mz = self.inversion_recovery_mz(
            TI,
            T1,
            TR
        )

        decay = self.t2_decay(
            TE,
            T2
        )

        signal = abs(mz) * decay

        return signal, mz, decay

    def ideal_null_ti(
        self,
        T1
    ):
        return T1 * np.log(2)

    # ============================================================
    # Presets
    # ============================================================

    def preset_stir(self):
        self.target_var.set("Fat")
        self.TI_var.set(
            self.ideal_null_ti(
                self.tissues["Fat"]["T1"]
            )
        )
        self.TR_var.set(2500)
        self.TE_var.set(20)

        self.restart_animation()

    def preset_flair(self):
        self.target_var.set("CSF")
        self.TI_var.set(
            self.ideal_null_ti(
                self.tissues["CSF"]["T1"]
            )
        )
        self.TR_var.set(7000)
        self.TE_var.set(120)

        self.restart_animation()

    def preset_t1(self):
        self.target_var.set("Fat")
        self.TI_var.set(0)
        self.TR_var.set(500)
        self.TE_var.set(15)

        self.restart_animation()

    def preset_t2(self):
        self.target_var.set("CSF")
        self.TI_var.set(0)
        self.TR_var.set(5000)
        self.TE_var.set(110)

        self.restart_animation()

    # ============================================================
    # Animation
    # ============================================================

    def toggle_animation(self):
        self.animation_running = not self.animation_running

        if self.animation_running:
            if self.animation_progress >= 1.0:
                self.animation_progress = 0.0

            self.start_button.config(
                text="Pause Animation"
            )
            self.animate_step()

        else:
            self.start_button.config(
                text="Start Animation"
            )

    def restart_animation(self):
        self.animation_running = False
        self.animation_progress = 0.0

        self.start_button.config(
            text="Start Animation"
        )

        self.update_plot()

    def animate_step(self):
        if not self.animation_running:
            return

        self.animation_progress += self.animation_step

        if self.animation_progress >= 1.0:
            self.animation_progress = 1.0
            self.animation_running = False

            self.start_button.config(
                text="Start Animation"
            )

            self.update_plot()
            return

        self.update_plot()

        self.root.after(
            self.animation_delay_ms,
            self.animate_step
        )

    # ============================================================
    # Main plot update
    # ============================================================

    def update_plot(self):
        TI = self.TI_var.get()
        TR = self.TR_var.get()
        TE = self.TE_var.get()
        target = self.target_var.get()

        self.draw_ir_recovery_panel(
            TI,
            TR,
            target
        )

        self.draw_te_decay_panel(
            TE
        )

        self.draw_final_signal_panel(
            TI,
            TR,
            TE,
            target
        )

        self.draw_timing_diagram(
            TI,
            TR,
            TE
        )

        self.fig.suptitle(
            "Interactive Tissue Nulling by Timing: TI / TR / TE",
            fontsize=15,
            y=0.985
        )

        self.update_status(
            TI,
            TR,
            TE,
            target
        )

        self.fig.tight_layout(
            rect=[0, 0, 1, 0.95]
        )

        self.canvas.draw_idle()

    # ============================================================
    # TI recovery panel
    # ============================================================

    def draw_ir_recovery_panel(
        self,
        TI,
        TR,
        target
    ):
        self.ax_ir.clear()

        self.ax_ir.set_title(
            "TI chooses which tissue crosses Mz = 0",
            fontsize=11
        )

        self.ax_ir.set_xlabel("TI / ms")
        self.ax_ir.set_ylabel("Mz at 90° pulse")

        self.ax_ir.set_xlim(0, 3000)
        self.ax_ir.set_ylim(-1.05, 1.05)

        self.ax_ir.grid(True, alpha=0.25)
        self.ax_ir.axhline(
            0,
            color="black",
            linewidth=1
        )

        full_t = np.linspace(
            0,
            3000,
            900
        )

        # Curves appear progressively with animation.
        visible_end = max(
            1,
            int(len(full_t) * self.animation_progress)
        )

        visible_t = full_t[:visible_end]

        for name, parameters in self.tissues.items():
            T1 = parameters["T1"]
            color = parameters["color"]

            full_y = self.inversion_recovery_mz(
                full_t,
                T1,
                TR
            )

            visible_y = full_y[:visible_end]

            line_width = 3.0 if name == target else 2.0
            line_alpha = 1.0 if name == target else 0.72

            self.ax_ir.plot(
                visible_t,
                visible_y,
                color=color,
                linewidth=line_width,
                alpha=line_alpha,
                label=f"{name} T1={T1} ms"
            )

            current_mz = self.inversion_recovery_mz(
                TI,
                T1,
                TR
            )

            self.ax_ir.scatter(
                [TI],
                [current_mz],
                color=color,
                s=65 if name == target else 38,
                alpha=1.0 if name == target else 0.65,
                zorder=6
            )

            # ----------------------------------------------------
            # Null-point vertical lines
            #
            # Selected tissue:
            #   bright, thick, opaque
            #
            # Nonselected tissues:
            #   thin, highly transparent
            # ----------------------------------------------------
            null_ti = self.ideal_null_ti(T1)

            if null_ti <= 3000:
                selected = name == target

                self.ax_ir.axvline(
                    null_ti,
                    color=color,
                    linestyle=":",
                    linewidth=3.0 if selected else 1.3,
                    alpha=0.95 if selected else 0.16,
                    zorder=1
                )

                if selected:
                    self.ax_ir.text(
                        null_ti,
                        -0.98,
                        f"{name} null ≈ {null_ti:.0f} ms",
                        color=color,
                        fontsize=9,
                        weight="bold",
                        ha="center",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": color,
                            "alpha": 0.82,
                            "pad": 2
                        }
                    )

        # Current TI marker
        self.ax_ir.axvline(
            TI,
            color="red",
            linestyle="--",
            linewidth=2.2,
            alpha=0.9,
            zorder=4
        )

        self.ax_ir.text(
            TI,
            0.98,
            f"TI={TI:.0f}",
            color="red",
            fontsize=9,
            rotation=90,
            va="top",
            ha="right"
        )

        self.ax_ir.legend(
            loc="lower right",
            fontsize=8,
            framealpha=0.88
        )

    # ============================================================
    # TE decay panel
    # ============================================================

    def draw_te_decay_panel(
        self,
        TE
    ):
        self.ax_te.clear()

        self.ax_te.set_title(
            "TE controls how much T2 decay remains",
            fontsize=11
        )

        self.ax_te.set_xlabel("TE / ms")
        self.ax_te.set_ylabel("T2 decay factor")

        self.ax_te.set_xlim(0, 250)
        self.ax_te.set_ylim(0, 1.05)

        self.ax_te.grid(True, alpha=0.25)

        full_t = np.linspace(
            0,
            250,
            700
        )

        visible_end = max(
            1,
            int(len(full_t) * self.animation_progress)
        )

        visible_t = full_t[:visible_end]

        for name, parameters in self.tissues.items():
            T2 = parameters["T2"]
            color = parameters["color"]

            full_y = self.t2_decay(
                full_t,
                T2
            )

            visible_y = full_y[:visible_end]

            self.ax_te.plot(
                visible_t,
                visible_y,
                color=color,
                linewidth=2.3,
                label=f"{name} T2={T2} ms"
            )

            current_decay = self.t2_decay(
                TE,
                T2
            )

            self.ax_te.scatter(
                [TE],
                [current_decay],
                color=color,
                s=42,
                zorder=5
            )

        self.ax_te.axvline(
            TE,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.85
        )

        self.ax_te.text(
            TE,
            0.98,
            f"TE={TE:.0f}",
            color="red",
            fontsize=9,
            rotation=90,
            va="top",
            ha="right"
        )

        self.ax_te.legend(
            loc="upper right",
            fontsize=8,
            framealpha=0.85
        )

    # ============================================================
    # Final signal bar panel
    # ============================================================

    def draw_final_signal_panel(
        self,
        TI,
        TR,
        TE,
        target
    ):
        self.ax_signal.clear()

        self.ax_signal.set_title(
            "Final magnitude signal",
            fontsize=11
        )

        self.ax_signal.set_ylabel(
            "Signal ≈ |Mz(TI)| × exp(-TE/T2)"
        )

        self.ax_signal.set_ylim(0, 1.05)
        self.ax_signal.grid(
            True,
            axis="y",
            alpha=0.25
        )

        names = list(self.tissues.keys())
        values = []
        colors = []

        for name in names:
            parameters = self.tissues[name]

            signal, _mz, _decay = self.final_signal(
                TI,
                TR,
                TE,
                parameters["T1"],
                parameters["T2"]
            )

            values.append(signal)
            colors.append(parameters["color"])

        bars = self.ax_signal.bar(
            names,
            values,
            color=colors,
            alpha=0.85
        )

        for bar, name, value in zip(
            bars,
            names,
            values
        ):
            self.ax_signal.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.2f}",
                ha="center",
                fontsize=9
            )

            if name == target:
                bar.set_edgecolor("red")
                bar.set_linewidth(3.2)

        target_index = names.index(target)
        target_signal = values[target_index]

        self.ax_signal.text(
            0.03,
            0.96,
            (
                f"Target: {target}\n"
                f"Signal = {target_signal:.3f}"
            ),
            transform=self.ax_signal.transAxes,
            fontsize=10,
            va="top",
            color="red" if target_signal < 0.08 else "black"
        )

        if target_signal < 0.08:
            self.ax_signal.text(
                0.03,
                0.76,
                "Target is nearly nulled",
                transform=self.ax_signal.transAxes,
                fontsize=11,
                color="red",
                weight="bold"
            )

    # ============================================================
    # Timing diagram
    # ============================================================

    def draw_timing_diagram(
        self,
        TI,
        TR,
        TE
    ):
        self.ax_timing.clear()

        self.ax_timing.set_title(
            "Sequence timing diagram",
            fontsize=11
        )

        echo_time = TI + TE
        xmax = max(
            TR,
            echo_time + 300
        )

        self.ax_timing.set_xlim(
            -0.03 * xmax,
            1.03 * xmax
        )

        self.ax_timing.set_ylim(
            -0.35,
            1.45
        )

        self.ax_timing.set_yticks([])
        self.ax_timing.set_xlabel("Time / ms")

        self.ax_timing.grid(
            True,
            axis="x",
            alpha=0.2
        )

        self.ax_timing.hlines(
            0.20,
            0,
            xmax,
            color="gray",
            linewidth=1.2
        )

        # 180° inversion pulse
        self.ax_timing.vlines(
            0,
            0.20,
            1.10,
            color="red",
            linewidth=4
        )

        self.ax_timing.text(
            0,
            1.15,
            "180° inversion",
            color="red",
            fontsize=9,
            ha="center"
        )

        # 90° excitation pulse
        self.ax_timing.vlines(
            TI,
            0.20,
            1.00,
            color="red",
            linewidth=4
        )

        self.ax_timing.text(
            TI,
            1.05,
            "90° excitation",
            color="red",
            fontsize=9,
            ha="center"
        )

        # Echo readout
        self.ax_timing.vlines(
            echo_time,
            0.20,
            0.85,
            color="purple",
            linewidth=3
        )

        self.ax_timing.text(
            echo_time,
            0.90,
            "Echo readout",
            color="purple",
            fontsize=9,
            ha="center"
        )

        # Next TR
        self.ax_timing.vlines(
            TR,
            0.20,
            1.10,
            color="black",
            linestyle="--",
            linewidth=2
        )

        self.ax_timing.text(
            TR,
            1.15,
            "Next TR",
            color="black",
            fontsize=9,
            ha="center"
        )

        # TI arrow
        self.ax_timing.annotate(
            "",
            xy=(TI, 0.02),
            xytext=(0, 0.02),
            arrowprops={
                "arrowstyle": "<->",
                "color": "black",
                "linewidth": 1.4
            }
        )

        self.ax_timing.text(
            TI / 2,
            0.07,
            f"TI = {TI:.0f} ms",
            ha="center",
            fontsize=9
        )

        # TE arrow
        self.ax_timing.annotate(
            "",
            xy=(echo_time, -0.13),
            xytext=(TI, -0.13),
            arrowprops={
                "arrowstyle": "<->",
                "color": "purple",
                "linewidth": 1.4
            }
        )

        self.ax_timing.text(
            (TI + echo_time) / 2,
            -0.08,
            f"TE = {TE:.0f} ms",
            ha="center",
            fontsize=9,
            color="purple"
        )

        # TR arrow
        self.ax_timing.annotate(
            "",
            xy=(TR, 1.33),
            xytext=(0, 1.33),
            arrowprops={
                "arrowstyle": "<->",
                "color": "gray",
                "linewidth": 1.4
            }
        )

        self.ax_timing.text(
            TR / 2,
            1.35,
            f"TR = {TR:.0f} ms",
            ha="center",
            fontsize=9,
            color="gray"
        )

        if echo_time > TR:
            self.ax_timing.text(
                0.02,
                0.72,
                "Warning: TI + TE exceeds TR",
                transform=self.ax_timing.transAxes,
                color="red",
                fontsize=11,
                weight="bold"
            )

    # ============================================================
    # Status panel
    # ============================================================

    def update_status(
        self,
        TI,
        TR,
        TE,
        target
    ):
        parameters = self.tissues[target]

        target_null_ti = self.ideal_null_ti(
            parameters["T1"]
        )

        target_signal, target_mz, target_decay = self.final_signal(
            TI,
            TR,
            TE,
            parameters["T1"],
            parameters["T2"]
        )

        if target_signal < 0.08:
            result = "The target tissue is effectively nulled."
        elif TI < target_null_ti:
            result = "TI is too short; the target has not reached its null point."
        else:
            result = "TI is too long; the target has already passed its null point."

        self.status_label.config(
            text=(
                f"Target: {target}\n\n"
                f"Ideal null TI ≈ {target_null_ti:.0f} ms\n"
                f"Current TI = {TI:.0f} ms\n"
                f"Mz at excitation = {target_mz:.3f}\n"
                f"T2 decay factor = {target_decay:.3f}\n"
                f"Final signal = {target_signal:.3f}\n\n"
                f"{result}"
            )
        )


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1600x900")

    app = InteractiveTimingNullingDemo(root)

    root.mainloop()