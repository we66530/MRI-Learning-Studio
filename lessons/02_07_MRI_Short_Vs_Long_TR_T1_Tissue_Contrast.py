import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# Short TR vs Long TR: T1 Tissue Contrast Animation
#
# Concept:
#   T1 contrast depends on how much longitudinal recovery occurs before
#   the next RF excitation.
#
#   Short TR:
#       Fat / short-T1 tissue recovers quickly.
#       Water / long-T1 tissue recovers slowly.
#       Difference in Mz before next RF pulse is large.
#       Strong T1 contrast.
#
#   Long TR:
#       Both tissues recover close to equilibrium.
#       Difference in Mz becomes small.
#       T1 contrast is reduced.
#
# Run:
#   pip install numpy matplotlib
#   python ShortTR_LongTR_T1_Contrast.py
# ============================================================


class T1ContrastDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Short TR vs Long TR: T1 Tissue Contrast")

        # -----------------------------
        # Animation settings
        # -----------------------------
        self.running = False
        self.frame = 0

        self.total_frames = 520

        # Simulated time in ms
        self.dt_ms = 10

        # Tissue T1 values in ms
        self.T1_fat = 250
        self.T1_water = 1200

        # TR settings
        self.short_TR = 500
        self.long_TR = 3000

        # Convert TR to frames
        self.short_TR_frames = int(self.short_TR / self.dt_ms)
        self.long_TR_frames = int(self.long_TR / self.dt_ms)

        # Flip angle
        self.flip_angle_deg = 90

        # -----------------------------
        # GUI layout
        # -----------------------------
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(15, 8))

        gs = self.fig.add_gridspec(
            nrows=2,
            ncols=2,
            height_ratios=[2.5, 1.2],
            hspace=0.35,
            wspace=0.25
        )

        self.ax_short_3d = self.fig.add_subplot(gs[0, 0], projection="3d")
        self.ax_long_3d = self.fig.add_subplot(gs[0, 1], projection="3d")

        self.ax_short_signal = self.fig.add_subplot(gs[1, 0])
        self.ax_long_signal = self.fig.add_subplot(gs[1, 1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # -----------------------------
        # Controls
        # -----------------------------
        ttk.Label(
            self.control_frame,
            text="T1 Contrast Demo",
            font=("Arial", 14, "bold")
        ).pack(pady=(0, 10))

        ttk.Button(
            self.control_frame,
            text="Start / Pause",
            command=self.toggle_animation
        ).pack(fill=tk.X, pady=4)

        ttk.Button(
            self.control_frame,
            text="Reset",
            command=self.reset_animation
        ).pack(fill=tk.X, pady=4)

        ttk.Separator(self.control_frame).pack(fill=tk.X, pady=10)

        ttk.Label(self.control_frame, text="Animation Speed").pack(anchor="w")
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(
            self.control_frame,
            from_=0.3,
            to=5.0,
            variable=self.speed_var,
            orient=tk.HORIZONTAL
        ).pack(fill=tk.X, pady=4)

        ttk.Separator(self.control_frame).pack(fill=tk.X, pady=10)

        self.frame_label = ttk.Label(self.control_frame, text="Frame: 0")
        self.frame_label.pack(anchor="w", pady=4)

        self.status_label = ttk.Label(
            self.control_frame,
            text="Status: Ready",
            wraplength=260,
            justify=tk.LEFT
        )
        self.status_label.pack(anchor="w", pady=4)

        ttk.Separator(self.control_frame).pack(fill=tk.X, pady=10)

        legend = (
            "Color meaning:\n\n"
            "Orange: short-T1 tissue, e.g. fat\n"
            "Blue: long-T1 tissue, e.g. water/CSF\n"
            "Red: RF pulse\n"
            "Green bar: contrast difference\n\n"
            "Top panels:\n"
            "3D Bloch vectors showing longitudinal recovery.\n\n"
            "Bottom panels:\n"
            "Mz recovery curves and signal difference."
        )

        ttk.Label(
            self.control_frame,
            text=legend,
            wraplength=260,
            justify=tk.LEFT
        ).pack(anchor="w", pady=4)

        self.anim = FuncAnimation(
            self.fig,
            self.update,
            frames=self.total_frames,
            interval=35,
            blit=False
        )

        self.reset_animation()

    # ============================================================
    # T1 recovery model
    # ============================================================

    def recovery_Mz(self, t_ms, T1):
        """
        Longitudinal recovery after a 90-degree pulse:
            Mz(t) = M0 * (1 - exp(-t / T1))
        M0 is normalized to 1.
        """
        return 1 - np.exp(-t_ms / T1)

    def transverse_signal_after_pulse(self, Mz_before_pulse):
        """
        For a 90-degree pulse, transverse signal is proportional to
        available longitudinal magnetization immediately before excitation.
        """
        return Mz_before_pulse * np.sin(np.deg2rad(self.flip_angle_deg))

    def time_in_TR(self, frame, TR_frames):
        return (frame % TR_frames) * self.dt_ms

    def is_rf_pulse(self, frame, TR_frames):
        local = frame % TR_frames
        return local < 5 or local > TR_frames - 5

    # ============================================================
    # Drawing helpers: 3D
    # ============================================================

    def setup_3d_axis(self, ax, title):
        ax.clear()

        ax.set_title(title, fontsize=11)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-0.25, 1.5)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("Mz / longitudinal")

        ax.view_init(elev=25, azim=45)

        # Bloch sphere / recovery guide
        u = np.linspace(0, 2 * np.pi, 36)
        v = np.linspace(0, np.pi, 18)

        x = 1.0 * np.outer(np.cos(u), np.sin(v))
        y = 1.0 * np.outer(np.sin(u), np.sin(v))
        z = 1.0 * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_wireframe(x, y, z, color="gray", alpha=0.18, linewidth=0.35)

        # Axes
        ax.quiver(0, 0, 0, 1.25, 0, 0, color="gray", linewidth=1)
        ax.quiver(0, 0, 0, 0, 1.25, 0, color="gray", linewidth=1)
        ax.quiver(0, 0, 0, 0, 0, 1.35, color="black", linewidth=1.5)

        ax.text(0, 0, 1.42, "B0 / z", fontsize=8)
        ax.text(1.32, 0, 0, "x", fontsize=8)
        ax.text(0, 1.32, 0, "y", fontsize=8)

    def draw_vector(self, ax, vec, color, label, linewidth=4):
        if np.linalg.norm(vec) < 1e-6:
            return

        x, y, z = vec

        ax.quiver(
            0, 0, 0,
            x, y, z,
            color=color,
            linewidth=linewidth,
            arrow_length_ratio=0.18
        )

        ax.text(
            x * 1.08,
            y * 1.08,
            z * 1.08 + 0.04,
            label,
            color=color,
            fontsize=9
        )

    def draw_rf_pulse(self, ax, active):
        if not active:
            return

        start = np.array([-1.35, -1.25, 0.25])
        direction = np.array([0.9, 0, 0])

        ax.quiver(
            start[0], start[1], start[2],
            direction[0], direction[1], direction[2],
            color="red",
            linewidth=4,
            arrow_length_ratio=0.25
        )

        ax.text(
            start[0] + direction[0],
            start[1],
            start[2] + 0.08,
            "90° RF pulse",
            color="red",
            fontsize=9
        )

    def draw_contrast_bar_3d(self, ax, diff, x0=-1.35, y0=1.25, z0=-0.10):
        max_len = 1.0
        bar_len = np.clip(diff, 0, max_len)

        ax.text(x0, y0, z0 + 0.18, "T1 contrast", color="green", fontsize=8)

        ax.plot(
            [x0, x0 + max_len],
            [y0, y0],
            [z0, z0],
            color="lightgray",
            linewidth=5
        )

        ax.plot(
            [x0, x0 + bar_len],
            [y0, y0],
            [z0, z0],
            color="green",
            linewidth=5
        )

    # ============================================================
    # Drawing helpers: 2D
    # ============================================================

    def setup_signal_axis(self, ax, title, TR_ms):
        ax.clear()
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, TR_ms)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Time after RF pulse / ms")
        ax.set_ylabel("Recovered Mz / signal")
        ax.grid(True, alpha=0.25)

        ax.axvline(TR_ms, color="red", linestyle="--", alpha=0.5)
        ax.text(TR_ms * 0.92, 1.02, "Next RF", color="red", fontsize=8)

    def draw_recovery_curves(self, ax, TR_ms, current_t_ms):
        t = np.linspace(0, TR_ms, 300)

        fat = self.recovery_Mz(t, self.T1_fat)
        water = self.recovery_Mz(t, self.T1_water)

        ax.plot(t, fat, color="darkorange", linewidth=2.5, label=f"Short T1 tissue, T1={self.T1_fat} ms")
        ax.plot(t, water, color="royalblue", linewidth=2.5, label=f"Long T1 tissue, T1={self.T1_water} ms")

        current_t_ms = np.clip(current_t_ms, 0, TR_ms)

        fat_now = self.recovery_Mz(current_t_ms, self.T1_fat)
        water_now = self.recovery_Mz(current_t_ms, self.T1_water)
        diff_now = abs(fat_now - water_now)

        ax.scatter([current_t_ms], [fat_now], color="darkorange", s=45, zorder=5)
        ax.scatter([current_t_ms], [water_now], color="royalblue", s=45, zorder=5)

        ax.plot(
            [current_t_ms, current_t_ms],
            [water_now, fat_now],
            color="green",
            linewidth=3,
            alpha=0.75
        )

        ax.text(
            current_t_ms,
            min(max(fat_now, water_now) + 0.05, 1.05),
            f"Δ={diff_now:.2f}",
            color="green",
            fontsize=9
        )

        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

        return fat_now, water_now, diff_now

    # ============================================================
    # Main update
    # ============================================================

    def draw_scene(self):
        # -----------------------------
        # Short TR
        # -----------------------------
        t_short = self.time_in_TR(self.frame, self.short_TR_frames)
        rf_short = self.is_rf_pulse(self.frame, self.short_TR_frames)

        fat_short = self.recovery_Mz(t_short, self.T1_fat)
        water_short = self.recovery_Mz(t_short, self.T1_water)
        diff_short = abs(fat_short - water_short)

        self.setup_3d_axis(
            self.ax_short_3d,
            f"SHORT TR = {self.short_TR} ms\nStrong T1 contrast"
        )

        self.draw_vector(
            self.ax_short_3d,
            np.array([0, 0, fat_short]),
            "darkorange",
            "Short T1"
        )

        self.draw_vector(
            self.ax_short_3d,
            np.array([0.08, 0.08, water_short]),
            "royalblue",
            "Long T1"
        )

        self.draw_rf_pulse(self.ax_short_3d, rf_short)
        self.draw_contrast_bar_3d(self.ax_short_3d, diff_short)

        self.setup_signal_axis(
            self.ax_short_signal,
            "Short TR: tissues recover unequally before next RF pulse",
            self.short_TR
        )

        self.draw_recovery_curves(
            self.ax_short_signal,
            self.short_TR,
            t_short
        )

        # -----------------------------
        # Long TR
        # -----------------------------
        t_long = self.time_in_TR(self.frame, self.long_TR_frames)
        rf_long = self.is_rf_pulse(self.frame, self.long_TR_frames)

        fat_long = self.recovery_Mz(t_long, self.T1_fat)
        water_long = self.recovery_Mz(t_long, self.T1_water)
        diff_long = abs(fat_long - water_long)

        self.setup_3d_axis(
            self.ax_long_3d,
            f"LONG TR = {self.long_TR} ms\nReduced T1 contrast"
        )

        self.draw_vector(
            self.ax_long_3d,
            np.array([0, 0, fat_long]),
            "darkorange",
            "Short T1"
        )

        self.draw_vector(
            self.ax_long_3d,
            np.array([0.08, 0.08, water_long]),
            "royalblue",
            "Long T1"
        )

        self.draw_rf_pulse(self.ax_long_3d, rf_long)
        self.draw_contrast_bar_3d(self.ax_long_3d, diff_long)

        self.setup_signal_axis(
            self.ax_long_signal,
            "Long TR: both tissues recover close to M0 before next RF pulse",
            self.long_TR
        )

        self.draw_recovery_curves(
            self.ax_long_signal,
            self.long_TR,
            t_long
        )

        self.frame_label.config(text=f"Frame: {self.frame}")

        self.status_label.config(
            text=(
                f"Short TR: t={t_short:.0f} ms, "
                f"short-T1 Mz={fat_short:.2f}, long-T1 Mz={water_short:.2f}, "
                f"contrast Δ={diff_short:.2f}\n\n"
                f"Long TR: t={t_long:.0f} ms, "
                f"short-T1 Mz={fat_long:.2f}, long-T1 Mz={water_long:.2f}, "
                f"contrast Δ={diff_long:.2f}"
            )
        )

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def update(self, _):
        if not self.running:
            return

        speed = self.speed_var.get()
        self.frame = int((self.frame + speed) % self.total_frames)

        self.draw_scene()

    # ============================================================
    # Buttons
    # ============================================================

    def toggle_animation(self):
        self.running = not self.running

    def reset_animation(self):
        self.frame = 0
        self.running = False
        self.draw_scene()


if __name__ == "__main__":
    root = tk.Tk()
    app = T1ContrastDemo(root)
    root.mainloop()