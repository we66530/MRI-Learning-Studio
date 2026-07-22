
import tkinter as tk
from tkinter import ttk
import numpy as np

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Circle


class T2T2StarRefocusLabSlideStyleV2:
    """
    Simplified MRI education demo focused on:
      1) 90-degree RF excitation
      2) T2
      3) T2*
      4) 180-degree RF refocusing
      5) Echo time TE

    Graphics:
      • One merged 2D curve plot, drawn progressively with time
      • One 3D hydrogen animation
      • One static explanation/legend panel beside the 3D plot

    Coordinate convention:
      B0 axis = +X
      Transverse plane = Y-Z
      90° RF B1 visual direction = +Z
      180° RF B1 visual direction = +Y
    """

    RF90_DURATION_MS = 12.0

    def __init__(self, root):
        self.root = root
        self.root.title("MRI T2 / T2* / TE Refocusing Lab — Unified B0 Vector")
        self.root.geometry("1760x980")

        # ----------------------------
        # Controls
        # ----------------------------
        self.b0_strength_var = tk.DoubleVar(value=1.5)
        self.b0_inhom_var = tk.DoubleVar(value=0.45)
        self.t2_ms_var = tk.DoubleVar(value=140.0)
        self.te_ms_var = tk.DoubleVar(value=120.0)
        self.b1_90_strength_var = tk.DoubleVar(value=1.0)
        self.b1_180_strength_var = tk.DoubleVar(value=1.0)
        self.dt_ms_var = tk.DoubleVar(value=1.0)
        self.max_time_ms_var = tk.DoubleVar(value=240.0)

        self.auto_rotate_var = tk.BooleanVar(value=False)
        self.show_90_pulse_var = tk.BooleanVar(value=True)
        self.show_180_pulse_var = tk.BooleanVar(value=True)
        self.auto_restart_var = tk.BooleanVar(value=False)

        # ----------------------------
        # Simulation state
        # ----------------------------
        self.running = False
        self.t = 0.0
        self.azim_3d = -56

        self.time_history = []
        self.t2_history = []
        self.t2star_history = []
        self.refocus_history = []

        self.fast_phase = 0.0
        self.slow_phase = 0.0
        self.refocus_signal = 0.0
        self.t2_signal = 0.0
        self.t2star_signal = 0.0

        self.fast_vec = np.array([1.0, 0.0, 0.0])
        self.slow_vec = np.array([1.0, 0.0, 0.0])

        self._build_ui()
        self.reset_simulation()
        self.redraw_all()
        self._animate_loop()

    # ============================================================
    # UI
    # ============================================================

    def _build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 20/80 layout
        main_frame.columnconfigure(0, weight=2, minsize=340)
        main_frame.columnconfigure(1, weight=8)
        main_frame.rowconfigure(0, weight=1)

        control_frame = ttk.Frame(main_frame, padding=10, width=340)
        control_frame.grid(row=0, column=0, sticky="nsew")
        control_frame.grid_propagate(False)

        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=1, sticky="nsew")

        title = ttk.Label(control_frame, text="T2 / T2* / Echo Time", font=("Arial", 18, "bold"))
        title.pack(anchor="w", pady=(0, 10))

        desc = ttk.Label(
            control_frame,
            text=(
                "This version separates RF roles:\n"
                "• 90° RF: creates transverse Mxy\n"
                "• 180° RF: refocuses phase spread\n\n"
                "Main teaching chain:\n"
                "90° RF → dephase → 180° RF → rephase → TE echo\n\n"
                "T2* dephasing is reversible only for the static ΔB part;\n"
                "true T2 loss remains."
            ),
            justify=tk.LEFT,
            wraplength=305,
            font=("Arial", 10),
        )
        desc.pack(anchor="w", pady=(0, 12))

        ttk.Separator(control_frame).pack(fill=tk.X, pady=8)

        self._add_slider(control_frame, "Main B0 strength", self.b0_strength_var, 0.3, 3.0)
        self._add_slider(control_frame, "B0 inhomogeneity / ΔB", self.b0_inhom_var, 0.0, 1.0)
        self._add_slider(control_frame, "Intrinsic T2 time", self.t2_ms_var, 40.0, 320.0)
        self._add_slider(control_frame, "Echo time TE", self.te_ms_var, 40.0, 200.0)
        self._add_slider(control_frame, "90° RF B1 visual strength", self.b1_90_strength_var, 0.2, 1.5)
        self._add_slider(control_frame, "180° RF B1 visual strength", self.b1_180_strength_var, 0.2, 1.5)
        self._add_slider(control_frame, "dt per frame", self.dt_ms_var, 0.4, 6.0)
        self._add_slider(control_frame, "Max time", self.max_time_ms_var, 120.0, 380.0)

        ttk.Separator(control_frame).pack(fill=tk.X, pady=8)

        ttk.Button(control_frame, text="Start / Restart Animation", command=self.start_animation).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Pause / Resume", command=self.toggle_pause).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Reset", command=self.reset_simulation).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Reset 3D View", command=self.reset_3d_view).pack(anchor="w", fill=tk.X, pady=2)

        ttk.Separator(control_frame).pack(fill=tk.X, pady=8)

        ttk.Checkbutton(control_frame, text="Show 90° RF pulse", variable=self.show_90_pulse_var, command=self._on_pulse_toggle).pack(anchor="w", pady=2)
        ttk.Checkbutton(control_frame, text="Show 180° RF pulse", variable=self.show_180_pulse_var, command=self._on_pulse_toggle).pack(anchor="w", pady=2)
        ttk.Checkbutton(control_frame, text="Auto rotate 3D demo", variable=self.auto_rotate_var).pack(anchor="w", pady=2)
        ttk.Checkbutton(control_frame, text="Auto restart", variable=self.auto_restart_var).pack(anchor="w", pady=2)

        self.status_label = ttk.Label(control_frame, text="", justify=tk.LEFT, wraplength=305, font=("Arial", 10))
        self.status_label.pack(anchor="w", pady=(14, 0))

        self.explain_label = ttk.Label(control_frame, text="", justify=tk.LEFT, wraplength=305, font=("Arial", 10))
        self.explain_label.pack(anchor="w", pady=(10, 0))

        # Graphics area:
        # top: merged curve plot
        # bottom: 3D plot + static guide panel
        self.fig = Figure(figsize=(14.8, 9.2), dpi=100, facecolor="white")
        gs = self.fig.add_gridspec(
            2, 2,
            height_ratios=[1.0, 1.85],
            width_ratios=[1.0, 0.30],
            hspace=0.26,
            wspace=0.12,
        )

        self.ax_curve = self.fig.add_subplot(gs[0, :])
        self.ax_3d = self.fig.add_subplot(gs[1, 0], projection="3d")
        self.ax_guide = self.fig.add_subplot(gs[1, 1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_slider_labels()

    def _add_slider(self, parent, label, variable, from_, to):
        frame = ttk.Frame(parent)
        frame.pack(anchor="w", fill=tk.X, pady=(0, 7))

        text_label = ttk.Label(frame, text="", font=("Arial", 10))
        text_label.pack(anchor="w")

        scale = ttk.Scale(
            frame,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=variable,
            command=lambda _: self.on_param_change(),
        )
        scale.pack(anchor="w", fill=tk.X)

        variable._label_widget = text_label
        variable._label_name = label

    # ============================================================
    # Parameter helpers
    # ============================================================

    def _on_pulse_toggle(self):
        if not self.running:
            self.compute_current_state()
        self.redraw_all()

    def update_slider_labels(self):
        for var in [
            self.b0_strength_var,
            self.b0_inhom_var,
            self.t2_ms_var,
            self.te_ms_var,
            self.b1_90_strength_var,
            self.b1_180_strength_var,
            self.dt_ms_var,
            self.max_time_ms_var,
        ]:
            if not hasattr(var, "_label_widget"):
                continue

            name = var._label_name
            val = var.get()
            if name in ["Intrinsic T2 time", "Echo time TE", "dt per frame", "Max time"]:
                text = f"{name}: {val:.1f} ms"
            else:
                text = f"{name}: {val:.3f}"
            var._label_widget.config(text=text)

        t2 = self.t2_ms_var.get()
        t2star = self.effective_t2star_ms()
        t180 = self.rf90_end_ms() + self.te_ms_var.get() / 2.0
        techo = self.rf90_end_ms() + self.te_ms_var.get()
        self.explain_label.config(
            text=(
                f"Effective T2* ≈ {t2star:.1f} ms\n"
                f"T2 = {t2:.1f} ms\n"
                f"90° RF duration = {self.RF90_DURATION_MS:.1f} ms\n"
                f"180° RF at ≈ {t180:.1f} ms\n"
                f"Echo TE peak at ≈ {techo:.1f} ms\n\n"
                "Color convention:\n"
                "• T2 = blue\n"
                "• T2* = gray dashed\n"
                "• T2* + 180° RF = dark blue"
            )
        )

    def on_param_change(self):
        min_max = self.rf90_end_ms() + self.te_ms_var.get() + 35
        if self.max_time_ms_var.get() < min_max:
            self.max_time_ms_var.set(min_max)
        self.update_slider_labels()
        if not self.running:
            self.compute_current_state()
        self.redraw_all()

    def rf90_end_ms(self):
        return self.RF90_DURATION_MS if self.show_90_pulse_var.get() else 0.0

    def effective_t2star_ms(self):
        t2 = max(1.0, float(self.t2_ms_var.get()))
        inh = float(self.b0_inhom_var.get())
        return 1.0 / (1.0 / t2 + 0.020 * inh)

    def phase_offset_rad_per_ms(self):
        inh = float(self.b0_inhom_var.get())
        return 0.004 + 0.022 * inh

    def is_90_pulse_active(self):
        return self.show_90_pulse_var.get() and 0.0 <= self.t <= self.RF90_DURATION_MS

    def is_180_pulse_active(self):
        if not self.show_180_pulse_var.get():
            return False
        tau_time = self.rf90_end_ms() + float(self.te_ms_var.get()) / 2.0
        pulse_width = max(7.0, min(18.0, 0.10 * float(self.te_ms_var.get())))
        return abs(self.t - tau_time) <= pulse_width / 2.0

    # ============================================================
    # Simulation
    # ============================================================

    def reset_simulation(self):
        self.running = False
        self.t = 0.0
        self.time_history = []
        self.t2_history = []
        self.t2star_history = []
        self.refocus_history = []
        self.compute_current_state()
        self.redraw_all()

    def start_animation(self):
        self.running = True
        self.t = 0.0
        self.time_history = []
        self.t2_history = []
        self.t2star_history = []
        self.refocus_history = []
        self.compute_current_state()
        self.redraw_all()

    def toggle_pause(self):
        self.running = not self.running
        self.redraw_all()

    def reset_3d_view(self):
        self.azim_3d = -56
        self.redraw_all()

    def compute_signals_at_elapsed(self, elapsed):
        """
        elapsed is time after the 90° excitation has completed.
        """
        t2 = max(1.0, float(self.t2_ms_var.get()))
        t2star = self.effective_t2star_ms()
        te = float(self.te_ms_var.get())
        tau = te / 2.0
        delta = self.phase_offset_rad_per_ms()

        if elapsed < 0:
            ramp = (self.t / max(self.RF90_DURATION_MS, 1e-6)) if self.show_90_pulse_var.get() else 1.0
            ramp = np.clip(ramp, 0.0, 1.0)
            return ramp, ramp, ramp, 0.0, 0.0

        if self.show_180_pulse_var.get():
            rel_factor = elapsed if elapsed <= tau else (elapsed - te)
        else:
            rel_factor = elapsed

        t2_signal = float(np.exp(-elapsed / t2))
        t2star_signal = float(np.exp(-elapsed / t2star))
        coherence = abs(np.cos(delta * rel_factor))
        refocus_signal = float(t2_signal * coherence)

        fast_phase = +delta * rel_factor
        slow_phase = -delta * rel_factor

        return t2_signal, t2star_signal, refocus_signal, fast_phase, slow_phase

    def compute_current_state(self):
        t = float(self.t)
        rf90_end = self.rf90_end_ms()
        elapsed = t - rf90_end

        # 90° RF excitation stage: spins rotate from +X/B0 axis into the transverse plane (+Y).
        if self.show_90_pulse_var.get() and t < self.RF90_DURATION_MS:
            progress = np.clip(t / self.RF90_DURATION_MS, 0.0, 1.0)
            angle = progress * (np.pi / 2.0)
            base_vec = np.array([np.cos(angle), np.sin(angle), 0.0])

            self.fast_phase = 0.0
            self.slow_phase = 0.0
            self.fast_vec = base_vec.copy()
            self.slow_vec = base_vec.copy()

            ramp = np.sin(angle)
            self.t2_signal = float(ramp)
            self.t2star_signal = float(ramp)
            self.refocus_signal = float(ramp)
            return

        self.t2_signal, self.t2star_signal, self.refocus_signal, self.fast_phase, self.slow_phase = self.compute_signals_at_elapsed(elapsed)

        # After 90° RF, spins precess in the transverse Y-Z plane around X/B0.
        visual_spin_speed = 0.5  # rad/ms，調大會讓整體 precession 更快
        visual_global_phase = visual_spin_speed * elapsed

        fast_draw_phase = visual_global_phase + self.fast_phase
        slow_draw_phase = visual_global_phase + self.slow_phase

        self.fast_vec = np.array([0.0, np.cos(fast_draw_phase), np.sin(fast_draw_phase)])
        self.slow_vec = np.array([0.0, np.cos(slow_draw_phase), np.sin(slow_draw_phase)])

    def _step_simulation(self):
        if self.auto_rotate_var.get():
            self.azim_3d += 0.34

        if not self.running:
            return

        self.t += float(self.dt_ms_var.get())

        if self.t > float(self.max_time_ms_var.get()):
            self.running = False
            if self.auto_restart_var.get():
                self.start_animation()
            return

        self.compute_current_state()

        self.time_history.append(self.t)
        self.t2_history.append(self.t2_signal)
        self.t2star_history.append(self.t2star_signal)
        self.refocus_history.append(self.refocus_signal)

    def _animate_loop(self):
        self._step_simulation()
        self.redraw_all()
        self.root.after(28, self._animate_loop)

    # ============================================================
    # Drawing
    # ============================================================

    def redraw_all(self):
        self.update_slider_labels()
        self.update_status()
        self.draw_merged_curve_plot()
        self.draw_3d_animation()
        self.draw_static_guide_panel()
        self.canvas.draw_idle()

    def update_status(self):
        rf90_end = self.rf90_end_ms()
        te = float(self.te_ms_var.get())
        tau_time = rf90_end + te / 2.0
        echo_time = rf90_end + te

        if self.is_90_pulse_active():
            status = "90° RF excitation ON: creating transverse Mxy"
        elif self.is_180_pulse_active():
            status = "180° RF refocusing pulse ON"
        elif self.t < tau_time:
            status = "After 90° RF: dephasing"
        elif self.show_180_pulse_var.get() and self.t < echo_time:
            status = "After 180° RF: rephasing toward echo"
        elif self.show_180_pulse_var.get() and abs(self.t - echo_time) <= 8:
            status = "At TE: echo peak"
        else:
            status = "After TE: T2 decay continues"

        self.status_label.config(
            text=(
                f"Running: {self.running}\n"
                f"Time: {self.t:.1f} ms\n"
                f"Status: {status}\n"
                f"Current T2 signal: {self.t2_signal:.3f}\n"
                f"Current T2* signal: {self.t2star_signal:.3f}\n"
                f"Current refocused signal: {self.refocus_signal:.3f}\n"
                f"Fast spin phase: {np.rad2deg(self.fast_phase):+.1f}°\n"
                f"Slow spin phase: {np.rad2deg(self.slow_phase):+.1f}°"
            )
        )

    def draw_merged_curve_plot(self):
        ax = self.ax_curve
        ax.clear()
        ax.set_facecolor("white")

        max_t = float(self.max_time_ms_var.get())
        rf90_end = self.rf90_end_ms()
        te = float(self.te_ms_var.get())
        tau_time = rf90_end + te / 2.0
        echo_time = rf90_end + te

        current_t = max(0.0, self.t)
        if current_t <= 0.0:
            current_t = 0.001

        t = np.linspace(0, current_t, 650)

        y_t2 = np.zeros_like(t)
        y_t2star = np.zeros_like(t)
        y_refocus = np.zeros_like(t)

        for i, ti in enumerate(t):
            elapsed = ti - rf90_end
            yy_t2, yy_t2star, yy_refocus, _, _ = self.compute_signals_at_elapsed(elapsed)
            y_t2[i] = yy_t2
            y_t2star[i] = yy_t2star
            y_refocus[i] = yy_refocus

        ax.set_title("T2, T2*, and T2* compensation with 90° + 180° RF pulses", fontsize=16, pad=10, weight="bold")

        ax.plot(t, y_t2, color="royalblue", linewidth=2.6, label="T2")
        ax.plot(t, y_t2star, color="gray", linestyle="--", linewidth=2.6, label="T2*")
        ax.plot(t, y_refocus, color="navy", linewidth=3.0, label="T2* + 180° RF pulse")

        # Current endpoints
        ax.scatter([t[-1]], [y_t2[-1]], color="royalblue", s=40, zorder=5)
        ax.scatter([t[-1]], [y_t2star[-1]], color="gray", s=40, zorder=5)
        ax.scatter([t[-1]], [y_refocus[-1]], color="navy", s=48, zorder=6)

        # 37% line
        ax.axhline(np.exp(-1), color="lightgray", linestyle=":", linewidth=1.1)
        ax.text(4, np.exp(-1) + 0.03, "37%", fontsize=11, color="gray")

        # Reference T2/T2* times after RF90
        t2 = max(1.0, float(self.t2_ms_var.get()))
        t2star = self.effective_t2star_ms()
        if rf90_end + t2star <= max_t:
            ax.axvline(rf90_end + t2star, color="gray", linestyle="--", linewidth=1.1, alpha=0.45)
            ax.text(rf90_end + t2star + 2, 0.10, "T2*", color="gray", fontsize=11)
        if rf90_end + t2 <= max_t:
            ax.axvline(rf90_end + t2, color="royalblue", linestyle="--", linewidth=1.1, alpha=0.45)
            ax.text(rf90_end + t2 + 2, 0.18, "T2", color="royalblue", fontsize=11)

        # RF90, RF180, and TE markers
        if self.show_90_pulse_var.get():
            ax.axvspan(0, self.RF90_DURATION_MS, color="orange", alpha=0.12)
            ax.text(1.5, 0.97, "90° RF\nexcitation", color="darkorange", fontsize=11, va="top", weight="bold")

        if self.show_180_pulse_var.get():
            ax.axvline(tau_time, color="orangered", linestyle="--", linewidth=1.6)
            ax.text(tau_time + 2, 0.95, "180° RF", color="orangered", fontsize=12, va="top", weight="bold")

        ax.axvline(echo_time, color="green", linestyle="--", linewidth=1.6)
        ax.text(echo_time + 2, 0.84, "TE / echo", color="green", fontsize=12, va="top", weight="bold")

        ax.set_xlim(0, max_t)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Time (ms)", fontsize=13)
        ax.set_ylabel("Signal / envelope", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.20)
        ax.legend(loc="upper right", fontsize=11, frameon=True)

        # Clean slide-like spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _make_sphere_xyz(self, center, radius, nu=42, nv=28):
        u = np.linspace(0, 2*np.pi, nu)
        v = np.linspace(0, np.pi, nv)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        return x, y, z

    def _plot_sphere(self, ax, center, radius, color, alpha=0.25):
        """
        Simple translucent sphere, used for RF glow shells.
        """
        x, y, z = self._make_sphere_xyz(center, radius, nu=28, nv=18)
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=False)

    def _plot_shaded_sphere(self, ax, center, radius, color, alpha=0.34):
        """
        Translucent hydrogen sphere with fake directional lighting.
        This makes the sphere look 3D/lateral instead of a flat transparent disk.
        """
        import matplotlib.colors as mcolors

        x, y, z = self._make_sphere_xyz(center, radius, nu=54, nv=34)

        # Surface normal of a sphere
        nx = (x - center[0]) / radius
        ny = (y - center[1]) / radius
        nz = (z - center[2]) / radius

        # Light comes from upper-left-front in the current display.
        light = np.array([-0.55, -0.35, 0.75], dtype=float)
        light = light / np.linalg.norm(light)

        intensity = nx * light[0] + ny * light[1] + nz * light[2]
        intensity = np.clip(intensity, -1.0, 1.0)

        # Convert light intensity to a calm slide-like shading factor.
        shade = 0.52 + 0.48 * (intensity + 1.0) / 2.0

        base = np.array(mcolors.to_rgb(color))
        rgb = np.clip(base[None, None, :] * shade[:, :, None] + 0.18 * (1.0 - shade[:, :, None]), 0, 1)

        rgba = np.zeros((*rgb.shape[:2], 4))
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = alpha

        ax.plot_surface(
            x, y, z,
            facecolors=rgba,
            linewidth=0.25,
            edgecolor=(0, 0, 0, 0.10),
            shade=False,
            antialiased=True,
        )

        # Tiny specular highlight cue
        highlight = center + np.array([-0.08, -0.08, 0.10])
        ax.scatter([highlight[0]], [highlight[1]], [highlight[2]], s=18, color="white", alpha=0.55)

    def _plot_glow_sphere(self, ax, center, base_radius, color):
        for mul, alpha in [(1.0, 0.25), (1.6, 0.13), (2.4, 0.06)]:
            self._plot_sphere(ax, center, base_radius * mul, color, alpha=alpha)

    def _wavy_b0_path(self, origin, length, inhom, n=220, phase=0.0):
        """
        Unified B0 visual:
        - length follows main B0 strength
        - wave amplitude follows B0 inhomogeneity
        - direction remains globally along +X
        """
        x = np.linspace(origin[0], origin[0] + length, n)

        # Inhomogeneity should deform the line, not reverse its direction.
        amp = 0.010 + 0.070 * float(inhom)
        if float(inhom) <= 1e-4:
            amp = 0.0

        normalized = np.linspace(0.0, 1.0, n)
        y = origin[1] + amp * np.sin(2.5 * 2*np.pi * normalized + phase)
        z = origin[2] + 0.60 * amp * np.sin(4.0 * 2*np.pi * normalized + 0.7 + phase)
        return x, y, z

    def _draw_unified_b0_vector_3d(self, ax, origin, b0_strength, inhom, label="B0"):
        """
        Draw one solid red B0 vector.
        Stronger B0 -> longer vector.
        More inhomogeneity -> more irregular/wavy vector.
        """
        length = 0.55 + 0.28 * float(b0_strength)
        x, y, z = self._wavy_b0_path(origin, length, inhom, n=260, phase=0.0)

        # Main solid vector body
        lw = 3.0 + 0.35 * float(b0_strength)
        ax.plot(x, y, z, color="red", linewidth=lw, alpha=0.90)

        # Arrowhead at the local tangent direction
        dx = x[-1] - x[-6]
        dy = y[-1] - y[-6]
        dz = z[-1] - z[-6]
        tangent = np.array([dx, dy, dz], dtype=float)
        tangent = tangent / (np.linalg.norm(tangent) + 1e-9)

        ax.quiver(
            x[-1] - 0.001,
            y[-1],
            z[-1],
            0.16 * tangent[0],
            0.16 * tangent[1],
            0.16 * tangent[2],
            color="red",
            linewidth=lw,
            arrow_length_ratio=0.55,
        )

        ax.text(x[-1] + 0.08, y[-1], z[-1], label, color="red", fontsize=12, weight="bold")
        if float(inhom) > 1e-4:
            ax.text(
                origin[0] + 0.12,
                origin[1] + 0.12,
                origin[2] + 0.10,
                "wavy B0 = inhomogeneity",
                color="red",
                fontsize=10,
            )

    def draw_3d_animation(self):
        ax = self.ax_3d
        ax.clear()
        ax.set_facecolor("white")

        ax.set_title("3D animation: 90° RF excitation, dephasing, and 180° RF refocusing", fontsize=16, pad=10, weight="bold")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.15, 1.15)
        ax.set_zlim(-1.05, 1.05)
        ax.set_xlabel("X / B0 axis", fontsize=13)
        ax.set_ylabel("Y / transverse", fontsize=13)
        ax.set_zlabel("Z / transverse", fontsize=13)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_box_aspect((2.3, 1.6, 1.5))
        ax.view_init(elev=22, azim=self.azim_3d)

        # Hydrogen centers
        fast_pos = np.array([-0.05, -0.45, 0.0])
        slow_pos = np.array([-0.05, +0.45, 0.0])

        # Reference circles in Y-Z plane (normal to X)
        theta = np.linspace(0, 2*np.pi, 240)
        r = 0.42
        for center, color, label in [
            (fast_pos, "firebrick", " "),
            (slow_pos, "royalblue", " "),
        ]:
            ax.plot(
                np.full_like(theta, center[0]),
                center[1] + r * np.cos(theta),
                center[2] + r * np.sin(theta),
                color="#d8d8d8",
                linewidth=1.0,
                alpha=0.85,
            )
            self._plot_shaded_sphere(ax, center, radius=0.20, color=color, alpha=0.36)
            ax.text(center[0] - 0.10, center[1] - 0.66, center[2] - 0.08, label, color=color, fontsize=10)

        # Spin arrows
        spin_len = 0.56
        fast_vec = spin_len * self.fast_vec
        slow_vec = spin_len * self.slow_vec

        ax.quiver(
            fast_pos[0], fast_pos[1], fast_pos[2],
            fast_vec[0], fast_vec[1], fast_vec[2],
            color="firebrick", linewidth=3.5, arrow_length_ratio=0.18
        )
        ax.quiver(
            slow_pos[0], slow_pos[1], slow_pos[2],
            slow_vec[0], slow_vec[1], slow_vec[2],
            color="royalblue", linewidth=3.5, arrow_length_ratio=0.18
        )

        # Net transverse magnetization.
        # During RF90, net vector follows excitation from longitudinal to transverse.
        net_origin = np.array([0.40, 0.0, 0.0])
        net_vec = 0.5 * (fast_vec + slow_vec)
        ax.quiver(
            net_origin[0], net_origin[1], net_origin[2],
            net_vec[0], net_vec[1], net_vec[2],
            color="black", linewidth=4.2, arrow_length_ratio=0.16
        )
        net_label = "M" if self.is_90_pulse_active() else "Net Mxy"
        ax.text(
            net_origin[0] + net_vec[0] + 0.02,
            net_origin[1] + net_vec[1] + 0.03,
            net_origin[2] + net_vec[2],
            net_label,
            color="black",
            fontsize=11,
        )

        # Unified B0 vector:
        #   stronger B0 -> longer vector
        #   stronger inhomogeneity -> more irregular/wavy vector
        inh = float(self.b0_inhom_var.get())
        self._draw_unified_b0_vector_3d(
            ax,
            origin=np.array([-1.15, -0.95, -0.82]),
            b0_strength=float(self.b0_strength_var.get()),
            inhom=inh,
            label="B0",
        )

        # 90° RF pulse direction and glow
        if self.show_90_pulse_var.get():
            b1_90_len = 0.48 + 0.34 * float(self.b1_90_strength_var.get())
            active90 = self.is_90_pulse_active()
            alpha90 = 1.0 if active90 else 0.22
            lw90 = 5.2 if active90 else 2.4
            origin90 = np.array([-0.95, -0.02, -0.72])

            if active90:
                self._plot_glow_sphere(ax, origin90 + np.array([0.0, 0.0, b1_90_len * 0.55]), 0.10 + 0.05 * float(self.b1_90_strength_var.get()), "orange")

            ax.quiver(
                origin90[0], origin90[1], origin90[2],
                0.0, 0.0, b1_90_len,
                color="orange", linewidth=lw90, alpha=alpha90, arrow_length_ratio=0.16
            )
            label90 = "90° RF B1: excitation" if active90 else "90° RF B1"
            ax.text(
                origin90[0] + 0.02,
                origin90[1],
                origin90[2] + b1_90_len + 0.05,
                label90,
                color="darkorange",
                fontsize=11,
                alpha=max(alpha90, 0.5),
                weight="bold" if active90 else "normal",
            )

        # 180° RF pulse direction and stronger flash
        if self.show_180_pulse_var.get():
            b1_180_len = 0.55 + 0.35 * float(self.b1_180_strength_var.get())
            active180 = self.is_180_pulse_active()
            alpha180 = 1.0 if active180 else 0.28
            lw180 = 5.8 if active180 else 2.8
            origin180 = np.array([-0.95, 0.78, -0.55])

            if active180:
                self._plot_glow_sphere(ax, origin180 + np.array([0.0, b1_180_len * 0.55, 0.0]), 0.12 + 0.05 * float(self.b1_180_strength_var.get()), "orangered")
                self._plot_glow_sphere(ax, origin180 + np.array([0.0, b1_180_len * 0.20, 0.0]), 0.07 + 0.03 * float(self.b1_180_strength_var.get()), "orange")

            ax.quiver(
                origin180[0], origin180[1], origin180[2],
                0.0, b1_180_len, 0.0,
                color="orangered", linewidth=lw180, alpha=alpha180, arrow_length_ratio=0.16
            )
            label180 = "180° RF B1: refocusing" if active180 else "180° RF B1 at TE/2"
            ax.text(
                origin180[0] + 0.02,
                origin180[1] + b1_180_len + 0.05,
                origin180[2],
                label180,
                color="orangered",
                fontsize=11,
                alpha=max(alpha180, 0.5),
                weight="bold" if active180 else "normal",
            )

            if active180:
                ax.text2D(0.72, 0.88, "180° RF FLASH", transform=ax.transAxes, fontsize=16, color="orangered", weight="bold")
            elif self.is_90_pulse_active():
                ax.text2D(0.72, 0.88, "90° RF EXCITATION", transform=ax.transAxes, fontsize=16, color="darkorange", weight="bold")

        # Keep the 3D scene visually clean: no extra explanatory text boxes on the 3D panel.

        # Clean pane colors for slide style
        ax.xaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        ax.yaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        ax.zaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))

    def draw_static_guide_panel(self):
        ax = self.ax_guide
        ax.clear()
        ax.set_facecolor("white")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(0.05, 0.96, "Static guide", fontsize=16, weight="bold", va="top")

        # Unified B0 visual guide:
        # A straight solid vector means homogeneous B0.
        # More waviness means stronger B0 inhomogeneity.
        xs = np.linspace(0.16, 0.84, 160)
        guide_inh = float(self.b0_inhom_var.get())
        amp = 0.005 + 0.060 * guide_inh
        ys = 0.73 + amp * np.sin(5.0 * 2*np.pi * (xs - xs.min()) / (xs.max() - xs.min()))
        ax.plot(xs, ys, color="red", linestyle="-", lw=3, alpha=0.90)
        ax.annotate(
            "",
            xy=(0.86, ys[-1]),
            xytext=(0.79, ys[-8]),
            arrowprops=dict(arrowstyle="->", color="red", lw=3),
        )

        ax.text(0.16, 0.82, "B0 vector", fontsize=12, color="red", weight="bold")

        # Hydrogen color cue
        # Increase vertical separation here if you want the two local-B0 bubbles farther apart.
        higher_y = 0.26
        lower_y = 0.10
        bubble_r = 0.060

        c1 = Circle((0.25, higher_y), bubble_r, color="firebrick", alpha=0.28)
        c2 = Circle((0.25, lower_y), bubble_r, color="royalblue", alpha=0.28)
        ax.add_patch(c1)
        ax.add_patch(c2)

        ax.text(0.34, higher_y, "Higher local B0", color="firebrick", fontsize=11, va="center")
        ax.text(0.34, lower_y, "Lower local B0", color="royalblue", fontsize=11, va="center")


def main():
    root = tk.Tk()
    app = T2T2StarRefocusLabSlideStyleV2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
