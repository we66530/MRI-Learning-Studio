import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass, field
import numpy as np

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# Utility functions
# ============================================================

def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=float)


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """
    Educational RF pulse rotation.

    B0 axis is X.
    A positive 90-degree RF pulse rotates M from +X toward +Y.

    This is a simplified rotating-frame model for teaching.
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def safe_norm(v: np.ndarray) -> float:
    n = float(np.linalg.norm(v))
    return max(n, 1e-8)


def vector_tilt_deg_from_b0(m: np.ndarray) -> float:
    """
    Angle between M vector and +B0 axis.
    B0 axis = +X.
    """
    n = safe_norm(m)
    cos_theta = np.clip(m[0] / n, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_theta)))


# ============================================================
# 1. State layer
# ============================================================

@dataclass
class MRIParams:
    # Coil settings
    b0_strength: float = 1.5
    rf_amplitude: float = 0.50
    rf_duration: float = 60.0
    gradient_x: float = 0.20

    # Which coil is selected for control
    active_coil: str = "none"  # "none", "main", "rf", "gradient"

    # Time
    time_step: int = 0

    # Real magnetization vector:
    # B0 is +X direction.
    # Mx = longitudinal magnetization.
    # My/Mz = transverse magnetization.
    M: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))

    # RF pulse execution state
    pulse_active: bool = False
    pulse_progress_step: int = 0
    pulse_total_steps: int = 0
    pulse_start_M: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    pulse_flip_rad: float = 0.0
    last_flip_delta_deg: float = 0.0

    # Receive signal
    receive_signal: float = 0.0

    # Educational relaxation constants, in animation steps
    t1_tau_steps: float = 900.0
    t2_tau_steps: float = 220.0


# ============================================================
# 2. Main application
# ============================================================

class MRIInteractiveDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("MRI Coil Interactive Visual Demo - Vector Magnetization")
        self.root.geometry("1740x940")

        self.params = MRIParams()

        # Tk variables
        self.active_coil_var = tk.StringVar(value=self.params.active_coil)

        self.b0_var = tk.DoubleVar(value=self.params.b0_strength)
        self.rf_amp_var = tk.DoubleVar(value=self.params.rf_amplitude)
        self.rf_dur_var = tk.DoubleVar(value=self.params.rf_duration)
        self.gx_var = tk.DoubleVar(value=self.params.gradient_x)

        self.show_bore = tk.BooleanVar(value=True)
        self.show_main = tk.BooleanVar(value=True)
        self.show_gradient = tk.BooleanVar(value=True)
        self.show_rf = tk.BooleanVar(value=True)
        self.show_receive = tk.BooleanVar(value=True)

        self.auto_rotate = tk.BooleanVar(value=False)
        self.animate_spin = tk.BooleanVar(value=True)
        self.animate_rf_field = tk.BooleanVar(value=True)

        self.coil_azim = -65

        self._build_ui()
        self.redraw_all()
        self.animate()

    # ============================================================
    # UI
    # ============================================================

    def _build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(main_frame, padding=12)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        visual_frame = ttk.Frame(main_frame)
        visual_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        visual_frame.columnconfigure(0, weight=3)
        visual_frame.columnconfigure(1, weight=2)
        visual_frame.rowconfigure(0, weight=1)
        visual_frame.rowconfigure(1, weight=1)

        # ---------------- Control panel ----------------

        title = ttk.Label(
            control_frame,
            text="MRI Coil Control",
            font=("Arial", 18, "bold")
        )
        title.pack(anchor="w", pady=(0, 12))

        desc = ttk.Label(
            control_frame,
            text=(
                "Vector spin model:\n"
                "M = [Mx, My, Mz]\n\n"
                "Mx: longitudinal along B0\n"
                "My/Mz: transverse signal\n\n"
                "RF pulse rotates M.\n"
                "T2 decays My/Mz.\n"
                "T1 recovers Mx toward +B0."
            ),
            justify=tk.LEFT
        )
        desc.pack(anchor="w", pady=(0, 16))

        ttk.Separator(control_frame).pack(fill=tk.X, pady=10)

        active_label = ttk.Label(
            control_frame,
            text="Active coil",
            font=("Arial", 12, "bold")
        )
        active_label.pack(anchor="w", pady=(0, 4))

        for label, value in [
            ("None / all coils inactive", "none"),
            ("Main Coil / B0", "main"),
            ("RF Coil / B1", "rf"),
            ("Gradient Coil / Gx", "gradient"),
        ]:
            ttk.Radiobutton(
                control_frame,
                text=label,
                variable=self.active_coil_var,
                value=value,
                command=self.on_control_changed
            ).pack(anchor="w", pady=2)

        ttk.Separator(control_frame).pack(fill=tk.X, pady=10)

        control_label = ttk.Label(
            control_frame,
            text="Coil controls",
            font=("Arial", 12, "bold")
        )
        control_label.pack(anchor="w", pady=(0, 6))

        self.b0_value_label = ttk.Label(control_frame, text="")
        self.b0_value_label.pack(anchor="w")
        self.b0_scale = ttk.Scale(
            control_frame,
            from_=0.2,
            to=3.0,
            orient=tk.HORIZONTAL,
            variable=self.b0_var,
            command=lambda value: self.on_control_changed()
        )
        self.b0_scale.pack(anchor="w", fill=tk.X, pady=(0, 10))

        self.rf_amp_value_label = ttk.Label(control_frame, text="")
        self.rf_amp_value_label.pack(anchor="w")
        self.rf_amp_scale = ttk.Scale(
            control_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.rf_amp_var,
            command=lambda value: self.on_control_changed()
        )
        self.rf_amp_scale.pack(anchor="w", fill=tk.X, pady=(0, 10))

        self.rf_dur_value_label = ttk.Label(control_frame, text="")
        self.rf_dur_value_label.pack(anchor="w")
        self.rf_dur_scale = ttk.Scale(
            control_frame,
            from_=5.0,
            to=120.0,
            orient=tk.HORIZONTAL,
            variable=self.rf_dur_var,
            command=lambda value: self.on_control_changed()
        )
        self.rf_dur_scale.pack(anchor="w", fill=tk.X, pady=(0, 10))

        self.apply_rf_button = ttk.Button(
            control_frame,
            text="Apply RF Pulse",
            command=self.apply_rf_pulse
        )
        self.apply_rf_button.pack(anchor="w", fill=tk.X, pady=(2, 6))

        preset_frame = ttk.Frame(control_frame)
        preset_frame.pack(anchor="w", fill=tk.X, pady=(0, 10))

        self.pulse_90_button = ttk.Button(
            preset_frame,
            text="90° Pulse",
            command=lambda: self.apply_preset_rf_pulse(90.0)
        )
        self.pulse_90_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        self.pulse_180_button = ttk.Button(
            preset_frame,
            text="180° Pulse",
            command=lambda: self.apply_preset_rf_pulse(180.0)
        )
        self.pulse_180_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        self.gx_value_label = ttk.Label(control_frame, text="")
        self.gx_value_label.pack(anchor="w")
        self.gx_scale = ttk.Scale(
            control_frame,
            from_=-1.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.gx_var,
            command=lambda value: self.on_control_changed()
        )
        self.gx_scale.pack(anchor="w", fill=tk.X, pady=(0, 10))

        ttk.Separator(control_frame).pack(fill=tk.X, pady=10)

        visibility_label = ttk.Label(
            control_frame,
            text="Show / hide structures",
            font=("Arial", 12, "bold")
        )
        visibility_label.pack(anchor="w", pady=(0, 4))

        ttk.Checkbutton(
            control_frame,
            text="Show MRI bore",
            variable=self.show_bore,
            command=self.draw_coil_scene
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            control_frame,
            text="Show Main Coil",
            variable=self.show_main,
            command=self.draw_coil_scene
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            control_frame,
            text="Show Gradient Coils",
            variable=self.show_gradient,
            command=self.draw_coil_scene
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            control_frame,
            text="Show RF Coil",
            variable=self.show_rf,
            command=self.draw_coil_scene
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            control_frame,
            text="Show Receive Coil",
            variable=self.show_receive,
            command=self.draw_coil_scene
        ).pack(anchor="w", pady=2)

        ttk.Separator(control_frame).pack(fill=tk.X, pady=10)

        animation_label = ttk.Label(
            control_frame,
            text="Animation",
            font=("Arial", 12, "bold")
        )
        animation_label.pack(anchor="w", pady=(0, 4))

        ttk.Checkbutton(
            control_frame,
            text="Auto rotate coil",
            variable=self.auto_rotate
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            control_frame,
            text="Animate RF field",
            variable=self.animate_rf_field
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            control_frame,
            text="Animate hydrogen spin",
            variable=self.animate_spin
        ).pack(anchor="w", pady=2)

        ttk.Button(
            control_frame,
            text="Reset View / Spin",
            command=self.reset_view
        ).pack(anchor="w", fill=tk.X, pady=(12, 4))

        ttk.Button(
            control_frame,
            text="Redraw All",
            command=self.redraw_all
        ).pack(anchor="w", fill=tk.X, pady=4)

        self.status_label = ttk.Label(
            control_frame,
            text="",
            justify=tk.LEFT
        )
        self.status_label.pack(anchor="w", pady=(16, 0))

        note = ttk.Label(
            control_frame,
            text=(
                "\nColor guide:\n"
                "Blue       : Main coil\n"
                "Yellow     : RF transmit coil\n"
                "Purple     : Gradient coil\n"
                "Lime green : Receive coil\n"
                "Red        : B0 field\n"
                "Orange     : RF / B1 field\n"
                "Cyan       : Hydrogen sample\n"
            ),
            justify=tk.LEFT
        )
        note.pack(anchor="w", pady=(8, 0))

        # ---------------- Main coil plot ----------------

        coil_panel = ttk.Frame(visual_frame)
        coil_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 8), pady=8)

        self.fig_coil = Figure(figsize=(8, 8), dpi=100)
        self.ax_coil = self.fig_coil.add_subplot(111, projection="3d")
        self.canvas_coil = FigureCanvasTkAgg(self.fig_coil, master=coil_panel)
        self.canvas_coil.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ---------------- Field plot ----------------

        field_panel = ttk.Frame(visual_frame)
        field_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 8), pady=(8, 4))

        self.fig_field = Figure(figsize=(5, 4), dpi=100)
        self.ax_field = self.fig_field.add_subplot(111, projection="3d")
        self.canvas_field = FigureCanvasTkAgg(self.fig_field, master=field_panel)
        self.canvas_field.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ---------------- Spin plot ----------------

        spin_panel = ttk.Frame(visual_frame)
        spin_panel.grid(row=1, column=1, sticky="nsew", padx=(8, 8), pady=(4, 8))

        self.fig_spin = Figure(figsize=(5, 4), dpi=100)
        self.ax_spin = self.fig_spin.add_subplot(111, projection="3d")
        self.canvas_spin = FigureCanvasTkAgg(self.fig_spin, master=spin_panel)
        self.canvas_spin.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_slider_locks()
        self.update_value_labels()
        self.update_status_text()

    # ============================================================
    # Control helpers
    # ============================================================

    def _set_ttk_scale_enabled(self, scale_widget, enabled: bool):
        if enabled:
            scale_widget.state(["!disabled"])
        else:
            scale_widget.state(["disabled"])

    def on_control_changed(self):
        self.params.active_coil = self.active_coil_var.get()

        self.params.b0_strength = float(self.b0_var.get())
        self.params.rf_amplitude = float(self.rf_amp_var.get())
        self.params.rf_duration = float(self.rf_dur_var.get())
        self.params.gradient_x = float(self.gx_var.get())

        self.update_slider_locks()
        self.update_value_labels()
        self.update_status_text()

        self.draw_coil_scene()
        self.draw_field_scene()
        self.draw_spin_scene()

    def update_slider_locks(self):
        active = self.params.active_coil

        self._set_ttk_scale_enabled(self.b0_scale, active == "main")

        rf_enabled = active == "rf"
        self._set_ttk_scale_enabled(self.rf_amp_scale, rf_enabled)
        self._set_ttk_scale_enabled(self.rf_dur_scale, rf_enabled)
        self.apply_rf_button.configure(state=("normal" if rf_enabled else "disabled"))
        self.pulse_90_button.configure(state=("normal" if rf_enabled else "disabled"))
        self.pulse_180_button.configure(state=("normal" if rf_enabled else "disabled"))

        self._set_ttk_scale_enabled(self.gx_scale, active == "gradient")

    def update_value_labels(self):
        active = self.params.active_coil

        b0_status = "ACTIVE" if active == "main" else "locked"
        rf_status = "ACTIVE" if active == "rf" else "locked"
        gx_status = "ACTIVE" if active == "gradient" else "locked"

        self.b0_value_label.config(
            text=f"Main coil strength B0: {self.params.b0_strength:.2f}  [{b0_status}]"
        )

        self.rf_amp_value_label.config(
            text=f"RF amplitude B1: {self.params.rf_amplitude:.2f}  [{rf_status}]"
        )

        self.rf_dur_value_label.config(
            text=f"RF pulse duration: {self.params.rf_duration:.1f} ms  [{rf_status}]"
        )

        self.gx_value_label.config(
            text=f"Gradient coil strength Gx: {self.params.gradient_x:+.2f}  [{gx_status}]"
        )

    def update_status_text(self):
        p = self.params
        M = p.M
        tilt = vector_tilt_deg_from_b0(M)
        transverse = np.sqrt(M[1] ** 2 + M[2] ** 2)

        pulse_text = "ON" if p.pulse_active else "OFF"

        self.status_label.config(
            text=(
                f"Active coil: {p.active_coil}\n"
                f"RF pulse: {pulse_text}\n"
                f"Last RF rotation: {p.last_flip_delta_deg:.1f}°\n"
                f"Tilt from +B0: {tilt:.1f}°\n"
                f"Mx: {M[0]:+.2f}\n"
                f"My: {M[1]:+.2f}\n"
                f"Mz: {M[2]:+.2f}\n"
                f"Mxy: {transverse:.2f}\n"
                f"Receive signal: {p.receive_signal:.2f}"
            )
        )

    def reset_view(self):
        self.coil_azim = -65
        self.params = MRIParams()

        self.active_coil_var.set(self.params.active_coil)
        self.b0_var.set(self.params.b0_strength)
        self.rf_amp_var.set(self.params.rf_amplitude)
        self.rf_dur_var.set(self.params.rf_duration)
        self.gx_var.set(self.params.gradient_x)

        self.on_control_changed()

    def redraw_all(self):
        self.on_control_changed()
        self.draw_coil_scene()
        self.draw_field_scene()
        self.draw_spin_scene()

    # ============================================================
    # RF pulse logic
    # ============================================================

    def apply_rf_pulse(self):
        if self.params.active_coil != "rf":
            return

        amp = float(self.rf_amp_var.get())
        dur = float(self.rf_dur_var.get())

        # Educational rule:
        # amplitude * duration determines RF rotation angle.
        # amp=1.0, duration=60 ms gives about 90 degrees.
        flip_delta_deg = 1.5 * amp * dur
        self.start_rf_rotation(flip_delta_deg, dur)

    def apply_preset_rf_pulse(self, flip_deg: float):
        if self.params.active_coil != "rf":
            return

        dur = float(self.rf_dur_var.get())
        self.start_rf_rotation(flip_deg, dur)

    def start_rf_rotation(self, flip_delta_deg: float, duration_ms: float):
        p = self.params

        # If pulse already running, start a new one from current M.
        p.pulse_active = True
        p.pulse_progress_step = 0
        p.pulse_total_steps = max(6, int(duration_ms / 4.0))

        p.pulse_start_M = p.M.copy()
        p.pulse_flip_rad = np.deg2rad(flip_delta_deg)
        p.last_flip_delta_deg = float(flip_delta_deg)

        self.update_status_text()

    # ============================================================
    # Physics / animation state update
    # ============================================================

    def update_simulation_state(self):
        p = self.params
        p.time_step += 1

        if p.time_step > 200000:
            p.time_step = 0

        # --------------------------------------------------------
        # RF pulse phase:
        # rotate M around a conceptual RF axis.
        #
        # In this teaching model:
        # +90 degree RF pulse rotates +X toward +Y.
        # This prevents clipping at 180 degrees and allows repeated pulses.
        # --------------------------------------------------------

        if p.pulse_active:
            p.pulse_progress_step += 1

            frac = min(1.0, p.pulse_progress_step / max(1, p.pulse_total_steps))
            current_flip = p.pulse_flip_rad * frac

            R = rotation_matrix_z(current_flip)
            p.M = R @ p.pulse_start_M

            if p.pulse_progress_step >= p.pulse_total_steps:
                p.pulse_active = False
                p.pulse_progress_step = 0

        # --------------------------------------------------------
        # Free precession and relaxation phase
        # --------------------------------------------------------

        else:
            # B0-driven free precession around X axis.
            # Stronger B0 -> faster visible phase rotation in YZ plane.
            dphi = 0.030 + 0.050 * p.b0_strength
            Rx = rotation_matrix_x(dphi)
            p.M = Rx @ p.M

            # T2-like transverse decay.
            # Stronger gradient causes faster dephasing.
            effective_t2 = p.t2_tau_steps / (1.0 + 2.8 * abs(p.gradient_x))
            t2_decay = np.exp(-1.0 / max(20.0, effective_t2))
            p.M[1] *= t2_decay
            p.M[2] *= t2_decay

            # T1-like longitudinal recovery:
            # Mx gradually returns to +1.
            t1_recovery = 1.0 - np.exp(-1.0 / max(20.0, p.t1_tau_steps))
            p.M[0] += (1.0 - p.M[0]) * t1_recovery

            # Avoid numerical drift beyond a reasonable educational range.
            p.M = np.clip(p.M, -1.2, 1.2)

        # Receive signal is proportional to transverse magnetization.
        transverse_mag = np.sqrt(p.M[1] ** 2 + p.M[2] ** 2)
        p.receive_signal = float(np.clip(transverse_mag, 0.0, 1.0))

    # ============================================================
    # Glow helpers
    # ============================================================

    def is_active(self, coil_name):
        return self.params.active_coil == coil_name

    def glow_alpha(self, coil_name):
        if not self.is_active(coil_name):
            return 0.0
        pulse = 0.5 + 0.5 * np.sin(self.params.time_step * 0.12)
        return 0.25 + 0.45 * pulse

    def glow_width(self, coil_name):
        if not self.is_active(coil_name):
            return 0.0
        pulse = 0.5 + 0.5 * np.sin(self.params.time_step * 0.12)
        return 6.0 + 5.0 * pulse

    # ============================================================
    # Coil scene
    # ============================================================

    def draw_coil_scene(self):
        ax = self.ax_coil
        ax.clear()

        ax.set_title("Horizontal MRI Coil Structure", fontsize=16, pad=18)
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-1.8, 1.8)
        ax.set_zlim(-1.8, 1.8)

        ax.set_xlabel("X / Bore axis")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_box_aspect((2.6, 1.3, 1.3))
        ax.view_init(elev=18, azim=self.coil_azim)

        self._draw_b0_arrow(ax)

        if self.show_bore.get():
            self._draw_bore(ax)

        if self.show_main.get():
            self._draw_main_coil(ax)

        if self.show_gradient.get():
            self._draw_gradient_coils(ax)

        if self.show_rf.get():
            self._draw_rf_coil(ax)

        if self.show_receive.get():
            self._draw_receive_coil(ax)

        self._draw_sample_marker(ax)
        self._draw_coil_labels(ax)

        self.canvas_coil.draw_idle()

    def _draw_b0_arrow(self, ax):
        b0_len = 1.4 + 1.1 * (self.params.b0_strength / 3.0)
        ax.quiver(
            -b0_len, -1.45, -1.45,
            2.0 * b0_len, 0, 0,
            color="red",
            linewidth=2.5,
            arrow_length_ratio=0.06
        )
        ax.text(-0.25, -1.58, -1.58, "B0", color="red", fontsize=12)

    def _draw_bore(self, ax):
        radius = 0.95
        x = np.linspace(-2.7, 2.7, 80)
        theta = np.linspace(0, 2 * np.pi, 90)

        x_grid, theta_grid = np.meshgrid(x, theta)
        y_grid = radius * np.cos(theta_grid)
        z_grid = radius * np.sin(theta_grid)

        ax.plot_surface(
            x_grid, y_grid, z_grid,
            alpha=0.12,
            color="gray",
            linewidth=0
        )

    def _draw_main_coil(self, ax):
        radius = 1.18
        length = 5.4
        turns = 26

        t = np.linspace(0, turns * 2 * np.pi, 2400)
        x = np.linspace(-length / 2, length / 2, len(t))
        y = radius * np.cos(t)
        z = radius * np.sin(t)

        if self.is_active("main"):
            ax.plot(
                x, y, z,
                color="cyan",
                linewidth=self.glow_width("main"),
                alpha=self.glow_alpha("main")
            )

        ax.plot(
            x, y, z,
            color="royalblue",
            linewidth=2.0 + 1.5 * (self.params.b0_strength / 3.0)
        )

        b0_len = 4.0 + 1.2 * (self.params.b0_strength / 3.0)
        ax.quiver(
            -b0_len / 2, 0, 0,
            b0_len, 0, 0,
            color="red",
            linewidth=2.0 + 2.0 * (self.params.b0_strength / 3.0),
            arrow_length_ratio=0.06
        )

    def _draw_gradient_coils(self, ax):
        radius = 1.36
        x_positions = [-2.1, -1.55, 1.55, 2.1]

        for x0 in x_positions:
            self._draw_ring_x_axis(
                ax,
                radius,
                x0,
                color="purple",
                linewidth=3.2,
                glow_name="gradient"
            )

        for phase in [0, np.pi]:
            x = np.linspace(-2.2, 2.2, 400)
            theta = 0.9 * np.sin(1.5 * x) + phase
            y = radius * np.cos(theta)
            z = radius * np.sin(theta)

            if self.is_active("gradient"):
                ax.plot(
                    x, y, z,
                    color="violet",
                    linewidth=self.glow_width("gradient"),
                    alpha=self.glow_alpha("gradient")
                )

            ax.plot(
                x, y, z,
                color="mediumorchid",
                linewidth=2.4 + 2.0 * abs(self.params.gradient_x)
            )

        x = np.linspace(-1.65, 1.65, 40)
        y = np.linspace(-0.65, 0.65, 20)
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = np.ones_like(x_grid) * -1.05

        ax.plot_surface(
            x_grid, y_grid, z_grid,
            alpha=0.10 + 0.18 * abs(self.params.gradient_x),
            color="magenta",
            linewidth=0
        )

    def _draw_rf_coil(self, ax):
        radius = 0.76
        length = 3.4

        self._draw_saddle_loop_x_axis(
            ax,
            radius=radius,
            length=length,
            theta1=np.deg2rad(55),
            theta2=np.deg2rad(125),
            color="gold",
            linewidth=4.0 + 2.0 * self.params.rf_amplitude,
            glow_name="rf"
        )

        self._draw_saddle_loop_x_axis(
            ax,
            radius=radius,
            length=length,
            theta1=np.deg2rad(235),
            theta2=np.deg2rad(305),
            color="gold",
            linewidth=4.0 + 2.0 * self.params.rf_amplitude,
            glow_name="rf"
        )

        # RF field arrow only during pulse.
        if self.params.pulse_active:
            amp = self.params.rf_amplitude
            rf_wave = np.sin(self.params.time_step * 0.35)
            rf_len = 0.35 + 1.1 * amp * abs(rf_wave)
            direction = 1.0 if rf_wave >= 0 else -1.0

            ax.quiver(
                0, -direction * rf_len / 2, 0,
                0, direction * rf_len, 0,
                color="orange",
                linewidth=2.0 + 3.0 * amp,
                arrow_length_ratio=0.12
            )
            ax.text(0.08, -0.15, 0.12, "B1 pulse", color="orange", fontsize=11)
        else:
            ax.text(0.08, -0.15, 0.12, "B1 idle", color="orange", fontsize=11)

    def _draw_receive_coil(self, ax):
        radius = 0.52
        theta = np.linspace(0, 2 * np.pi, 300)

        x = np.zeros_like(theta) + 0.05
        y = radius * np.cos(theta)
        z = radius * np.sin(theta)

        sig = self.params.receive_signal

        ax.plot(
            x, y, z,
            color="limegreen",
            linewidth=3.2 + 4.0 * sig
        )

        x2 = np.zeros_like(theta) - 0.08
        y2 = (radius * 0.88) * np.cos(theta)
        z2 = (radius * 0.88) * np.sin(theta)

        ax.plot(
            x2, y2, z2,
            color="mediumseagreen",
            linewidth=2.0,
            alpha=0.85
        )

        ax.quiver(
            0.00, 0.0, 0.0,
            0.00, 0.75 * sig, 0.0,
            color="limegreen",
            linewidth=2.0 + 3.0 * sig,
            arrow_length_ratio=0.18
        )

    def _draw_sample_marker(self, ax):
        ax.scatter([0], [0], [0], s=100, color="cyan", edgecolor="black")
        ax.text(0.10, 0.05, 0.08, "Hydrogen sample", color="black", fontsize=10)

    def _draw_ring_x_axis(self, ax, radius, x0, color, linewidth=2.0, glow_name=None):
        theta = np.linspace(0, 2 * np.pi, 400)
        x = np.ones_like(theta) * x0
        y = radius * np.cos(theta)
        z = radius * np.sin(theta)

        if glow_name and self.is_active(glow_name):
            ax.plot(
                x, y, z,
                color="violet",
                linewidth=self.glow_width(glow_name),
                alpha=self.glow_alpha(glow_name)
            )

        ax.plot(x, y, z, color=color, linewidth=linewidth)

    def _draw_saddle_loop_x_axis(
        self,
        ax,
        radius,
        length,
        theta1,
        theta2,
        color,
        linewidth,
        glow_name=None
    ):
        x1 = np.linspace(-length / 2, length / 2, 220)
        y1 = radius * np.cos(theta1) * np.ones_like(x1)
        z1 = radius * np.sin(theta1) * np.ones_like(x1)

        arc1 = np.linspace(theta1, theta2, 100)
        x2 = np.ones_like(arc1) * (length / 2)
        y2 = radius * np.cos(arc1)
        z2 = radius * np.sin(arc1)

        x3 = np.linspace(length / 2, -length / 2, 220)
        y3 = radius * np.cos(theta2) * np.ones_like(x3)
        z3 = radius * np.sin(theta2) * np.ones_like(x3)

        arc2 = np.linspace(theta2, theta1, 100)
        x4 = np.ones_like(arc2) * (-length / 2)
        y4 = radius * np.cos(arc2)
        z4 = radius * np.sin(arc2)

        x = np.concatenate([x1, x2, x3, x4])
        y = np.concatenate([y1, y2, y3, y4])
        z = np.concatenate([z1, z2, z3, z4])

        if glow_name and self.is_active(glow_name):
            ax.plot(
                x, y, z,
                color="yellow",
                linewidth=self.glow_width(glow_name),
                alpha=self.glow_alpha(glow_name)
            )

        ax.plot(x, y, z, color=color, linewidth=linewidth)

    def _draw_coil_labels(self, ax):
        ax.text(-2.8, 1.45, 1.35, "Main Coil", color="royalblue", fontsize=12)
        ax.text(-2.8, -1.55, 1.35, "Gradient Coils", color="purple", fontsize=12)
        ax.text(-0.7, 0.95, 0.7, "RF Coil", color="goldenrod", fontsize=12)
        ax.text(0.18, 0.60, 0.60, "Receive Coil", color="green", fontsize=12)

    # ============================================================
    # Field scene
    # ============================================================

    def draw_field_scene(self):
        ax = self.ax_field
        ax.clear()

        ax.set_title("Magnetic Field Vector Response", fontsize=13, pad=12)
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        ax.set_box_aspect((2.2, 1.2, 1.2))
        ax.view_init(elev=20, azim=-62)

        ax.set_xlabel("X / bore")
        ax.set_ylabel("Y / RF")
        ax.set_zlabel("Z")

        xs = np.linspace(-1.35, 1.35, 9)
        ys = np.linspace(-0.45, 0.45, 3)
        zs = np.linspace(-0.35, 0.35, 3)

        X, Y, Z = np.meshgrid(xs, ys, zs)
        X = X.ravel()
        Y = Y.ravel()
        Z = Z.ravel()

        b0 = self.params.b0_strength
        gx = self.params.gradient_x
        rf_amp = self.params.rf_amplitude

        # Bx = B0 + Gx*x
        U = 0.16 * b0 + 0.22 * gx * X

        # RF transverse field exists only during RF pulse.
        if self.params.pulse_active:
            rf_wave = np.sin(self.params.time_step * 0.35)
            V = 0.60 * rf_amp * rf_wave * np.ones_like(U)
        else:
            rf_wave = 0.0
            V = np.zeros_like(U)

        W = np.zeros_like(U)

        vector_magnitude = np.sqrt(U ** 2 + V ** 2 + W ** 2)

        ax.quiver(
            X, Y, Z,
            U, V, W,
            length=1.0,
            normalize=False,
            color="darkred",
            linewidth=1.1,
            arrow_length_ratio=0.25
        )

        b0_len = 1.4 + 1.2 * (b0 / 3.0)
        ax.quiver(
            -b0_len / 2, -0.8, -0.8,
            b0_len, 0, 0,
            color="red",
            linewidth=2.4,
            arrow_length_ratio=0.08
        )
        ax.text(-0.15, -0.9, -0.9, "B0", color="red", fontsize=10)

        rf_len = 1.25 * rf_amp * rf_wave
        ax.quiver(
            0.0, 0.0, 0.75,
            0, rf_len, 0,
            color="orange",
            linewidth=2.4,
            arrow_length_ratio=0.12
        )
        ax.text(0.05, 0.1, 0.82, "B1 RF", color="orange", fontsize=9)

        line_x = np.linspace(-1.4, 1.4, 100)
        line_y = np.zeros_like(line_x) + 0.75
        line_z = gx * 0.55 * line_x

        ax.plot(line_x, line_y, line_z, color="purple", linewidth=3)
        ax.text(-1.45, 0.78, -0.55, "Bx(x)=B0+Gx·x", color="purple", fontsize=9)


        self.canvas_field.draw_idle()

    # ============================================================
    # Spin scene
    # ============================================================

    def draw_spin_scene(self):
        ax = self.ax_spin
        ax.clear()

        ax.set_title("Hydrogen Magnetization Vector M", fontsize=13, pad=12)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-62)

        ax.set_xlabel("Mx / B0 axis")
        ax.set_ylabel("My")
        ax.set_zlabel("Mz")

        self._draw_bloch_sphere(ax)

        # B0 direction
        ax.quiver(
            -1.0, -1.05, -1.05,
            2.0, 0, 0,
            color="red",
            linewidth=2.5,
            arrow_length_ratio=0.08
        )
        ax.text(-0.1, -1.15, -1.15, "B0", color="red", fontsize=10)

        M = self.params.M.copy()
        M_norm = safe_norm(M)

        # M vector
        ax.quiver(
            0, 0, 0,
            M[0], M[1], M[2],
            color="blue",
            linewidth=4,
            arrow_length_ratio=0.12
        )
        ax.text(M[0] * 1.05, M[1] * 1.05, M[2] * 1.05, "M", color="blue", fontsize=12)

        # Transverse component projection
        ax.quiver(
            0, 0, 0,
            0, M[1], M[2],
            color="limegreen",
            linewidth=3,
            arrow_length_ratio=0.12
        )
        ax.text(0.03, M[1] * 1.05, M[2] * 1.05, "Mxy", color="green", fontsize=10)

        # Longitudinal component projection
        ax.quiver(
            0, 0, 0,
            M[0], 0, 0,
            color="red",
            linewidth=2,
            arrow_length_ratio=0.12
        )
        ax.text(M[0] * 1.05, 0.03, 0.03, "Mx", color="red", fontsize=10)

        # Reference transverse circle at current Mx
        theta = np.linspace(0, 2 * np.pi, 200)
        r_trans = np.sqrt(max(0.0, M_norm ** 2 - M[0] ** 2))
        circle_x = np.ones_like(theta) * M[0]
        circle_y = r_trans * np.cos(theta)
        circle_z = r_trans * np.sin(theta)

        ax.plot(
            circle_x, circle_y, circle_z,
            color="gray",
            linewidth=1.0,
            linestyle="--",
            alpha=0.7
        )

        # Short precession trace around X axis
        trace_t = np.linspace(0, 80, 120)
        dphi = 0.030 + 0.050 * self.params.b0_strength
        trace_phase = -dphi * trace_t
        current_phase = np.arctan2(M[2], M[1])
        current_trans = np.sqrt(M[1] ** 2 + M[2] ** 2)

        effective_t2 = self.params.t2_tau_steps / (1.0 + 2.8 * abs(self.params.gradient_x))
        trace_decay = np.exp(-trace_t / max(20.0, effective_t2))

        trace_x = np.ones_like(trace_t) * M[0]
        trace_y = current_trans * np.cos(current_phase + trace_phase) * trace_decay
        trace_z = current_trans * np.sin(current_phase + trace_phase) * trace_decay

        ax.plot(trace_x, trace_y, trace_z, color="navy", linewidth=2.0, alpha=0.8)

        ax.scatter([0], [0], [0], s=80, color="cyan", edgecolor="black")

        # Receive signal indicator
        sig = self.params.receive_signal
        ax.quiver(
            1.02, -0.95, -1.0,
            0, 0, 0.85 * sig,
            color="limegreen",
            linewidth=4,
            arrow_length_ratio=0.16
        )
        ax.text(0.86, -0.98, -1.10, "Receive", color="green", fontsize=10)

        tilt = vector_tilt_deg_from_b0(M)
        transverse = np.sqrt(M[1] ** 2 + M[2] ** 2)

        self.canvas_spin.draw_idle()

    def _draw_bloch_sphere(self, ax):
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(
            x, y, z,
            color="cyan",
            alpha=0.10,
            linewidth=0
        )

        theta = np.linspace(0, 2 * np.pi, 200)

        # YZ equator, perpendicular to B0 axis
        ax.plot(
            np.zeros_like(theta), np.cos(theta), np.sin(theta),
            color="gray",
            linewidth=0.8,
            alpha=0.7
        )

        # XY circle
        ax.plot(
            np.cos(theta), np.sin(theta), np.zeros_like(theta),
            color="gray",
            linewidth=0.8,
            alpha=0.45
        )

        # XZ circle
        ax.plot(
            np.cos(theta), np.zeros_like(theta), np.sin(theta),
            color="gray",
            linewidth=0.8,
            alpha=0.45
        )

    # ============================================================
    # Main animation loop
    # ============================================================

    def animate(self):
        self.update_simulation_state()
        self.update_status_text()

        if self.auto_rotate.get():
            self.coil_azim += 0.45

        self.draw_field_scene()

        if self.animate_spin.get():
            self.draw_spin_scene()

        if (
            self.auto_rotate.get()
            or self.params.active_coil != "none"
            or self.params.pulse_active
            or self.params.receive_signal > 0.02
        ):
            self.draw_coil_scene()

        self.root.after(40, self.animate)


def main():
    root = tk.Tk()
    app = MRIInteractiveDemo(root)
    root.mainloop()


if __name__ == "__main__":
    main()