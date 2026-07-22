
import tkinter as tk
from tkinter import ttk
import numpy as np

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle


class T1RelaxationLab:
    """
    MRI Chapter 04 — T1 Relaxation Lab

    This script intentionally contains NO T1* concept.

    Teaching goals:
        1. T1 is longitudinal recovery toward B0.
        2. 90° RF saturation: Mx -> 0, then Mx recovers toward +B0.
        3. 180° RF inversion: Mx -> -M0, then Mx recovers toward +B0.
        4. Different tissues can have different T1 values:
              short T1 recovers faster
              long T1 recovers slower
        5. The receiver still only detects transverse signal; T1 becomes visible
           only when recovered Mx is later converted into Mxy by an RF pulse.

    Coordinate convention:
        B0 axis = +X
        Transverse plane = Y-Z

    Time unit:
        seconds
    """

    RF_DURATION_S = 0.18

    def __init__(self, root):
        self.root = root
        self.root.title("MRI Chapter 04 — T1 Relaxation Lab")
        self.root.geometry("1760x980")

        # ----------------------------
        # Controls
        # ----------------------------
        self.b0_strength_var = tk.DoubleVar(value=1.5)
        self.t1_s_var = tk.DoubleVar(value=1.2)
        self.short_t1_factor_var = tk.DoubleVar(value=0.60)
        self.long_t1_factor_var = tk.DoubleVar(value=1.55)
        self.b1_90_strength_var = tk.DoubleVar(value=1.0)
        self.b1_180_strength_var = tk.DoubleVar(value=1.0)
        self.dt_s_var = tk.DoubleVar(value=0.025)
        self.max_time_s_var = tk.DoubleVar(value=6.0)

        self.auto_rotate_var = tk.BooleanVar(value=False)
        self.auto_restart_var = tk.BooleanVar(value=False)

        # ----------------------------
        # Simulation state
        # ----------------------------
        self.mode = "saturation"  # saturation or inversion
        self.running = False
        self.t = 0.0
        self.azim_3d = -56

        self.ideal_mx = 1.0
        self.short_mx = 1.0
        self.long_mx = 1.0
        self.current_mx = 1.0

        self.short_vec = np.array([1.0, 0.0, 0.0])
        self.long_vec = np.array([1.0, 0.0, 0.0])
        self.net_vec = np.array([1.0, 0.0, 0.0])

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

        # 20 / 80 layout
        main_frame.columnconfigure(0, weight=2, minsize=340)
        main_frame.columnconfigure(1, weight=8)
        main_frame.rowconfigure(0, weight=1)

        control_frame = ttk.Frame(main_frame, padding=10, width=340)
        control_frame.grid(row=0, column=0, sticky="nsew")
        control_frame.grid_propagate(False)

        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=1, sticky="nsew")

        ttk.Label(control_frame, text="T1 Relaxation", font=("Arial", 18, "bold")).pack(anchor="w", pady=(0, 10))

        desc = ttk.Label(
            control_frame,
            text=(
                "This chapter explains T1 only.\n\n"
                "T1 = longitudinal spin-lattice recovery toward B0.\n\n"
                "90° RF saturation:\n"
                "Mx → 0, then Mx recovers.\n\n"
                "180° RF inversion:\n"
                "Mx → -M0, then Mx recovers.\n\n"
                "There is no T1* in this basic MRI physics model."
            ),
            justify=tk.LEFT,
            wraplength=305,
            font=("Arial", 10),
        )
        desc.pack(anchor="w", pady=(0, 12))

        ttk.Separator(control_frame).pack(fill=tk.X, pady=8)

        self._add_slider(control_frame, "Main B0 strength", self.b0_strength_var, 0.3, 3.0)
        self._add_slider(control_frame, "Reference T1 time", self.t1_s_var, 0.2, 4.0)
        self._add_slider(control_frame, "Short T1 factor", self.short_t1_factor_var, 0.25, 0.95)
        self._add_slider(control_frame, "Long T1 factor", self.long_t1_factor_var, 1.05, 3.00)
        self._add_slider(control_frame, "90° RF B1 visual strength", self.b1_90_strength_var, 0.2, 1.5)
        self._add_slider(control_frame, "180° RF B1 visual strength", self.b1_180_strength_var, 0.2, 1.5)
        self._add_slider(control_frame, "dt per frame", self.dt_s_var, 0.005, 0.080)
        self._add_slider(control_frame, "Max time", self.max_time_s_var, 2.0, 12.0)

        ttk.Separator(control_frame).pack(fill=tk.X, pady=8)

        ttk.Button(control_frame, text="Start 90° Saturation Recovery", command=self.start_saturation).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Start 180° Inversion Recovery", command=self.start_inversion).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Pause / Resume", command=self.toggle_pause).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Reset", command=self.reset_simulation).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Reset 3D View", command=self.reset_3d_view).pack(anchor="w", fill=tk.X, pady=2)

        ttk.Separator(control_frame).pack(fill=tk.X, pady=8)

        ttk.Checkbutton(control_frame, text="Auto rotate 3D demo", variable=self.auto_rotate_var).pack(anchor="w", pady=2)
        ttk.Checkbutton(control_frame, text="Auto restart", variable=self.auto_restart_var).pack(anchor="w", pady=2)

        self.status_label = ttk.Label(control_frame, text="", justify=tk.LEFT, wraplength=305, font=("Arial", 10))
        self.status_label.pack(anchor="w", pady=(14, 0))

        self.explain_label = ttk.Label(control_frame, text="", justify=tk.LEFT, wraplength=305, font=("Arial", 10))
        self.explain_label.pack(anchor="w", pady=(10, 0))

        # ----------------------------
        # Graphics
        # ----------------------------
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
    # Parameters
    # ============================================================

    def update_slider_labels(self):
        for var in [
            self.b0_strength_var,
            self.t1_s_var,
            self.short_t1_factor_var,
            self.long_t1_factor_var,
            self.b1_90_strength_var,
            self.b1_180_strength_var,
            self.dt_s_var,
            self.max_time_s_var,
        ]:
            if not hasattr(var, "_label_widget"):
                continue

            name = var._label_name
            val = var.get()

            if name in ["Reference T1 time", "dt per frame", "Max time"]:
                txt = f"{name}: {val:.3f} s"
            else:
                txt = f"{name}: {val:.3f}"

            var._label_widget.config(text=txt)

        t1 = float(self.t1_s_var.get())
        short_t1 = self.short_t1_s()
        long_t1 = self.long_t1_s()
        null_time = t1 * np.log(2.0)

        self.explain_label.config(
            text=(
                f"Reference T1 = {t1:.2f} s\n"
                f"Short T1 = {short_t1:.2f} s\n"
                f"Long T1 = {long_t1:.2f} s\n"
                f"Inversion null time ≈ T1 ln 2 = {null_time:.2f} s\n\n"
                "Color convention:\n"
                "• Reference T1 = blue\n"
                "• Short T1 = red\n"
                "• Long T1 = dark blue\n\n"
                "No T1* is used here."
            )
        )

    def on_param_change(self):
        self.update_slider_labels()
        if not self.running:
            self.compute_current_state()
        self.redraw_all()

    def short_t1_s(self):
        return max(0.05, float(self.short_t1_factor_var.get()) * float(self.t1_s_var.get()))

    def long_t1_s(self):
        return max(0.05, float(self.long_t1_factor_var.get()) * float(self.t1_s_var.get()))

    def rf_active(self):
        return 0.0 <= self.t <= self.RF_DURATION_S

    # ============================================================
    # Recovery model
    # ============================================================

    def saturation_recovery(self, elapsed, t1):
        if elapsed < 0:
            progress = np.clip(self.t / self.RF_DURATION_S, 0.0, 1.0)
            return np.cos(progress * np.pi / 2.0)
        return 1.0 - np.exp(-elapsed / max(0.05, t1))

    def inversion_recovery(self, elapsed, t1):
        if elapsed < 0:
            progress = np.clip(self.t / self.RF_DURATION_S, 0.0, 1.0)
            return np.cos(progress * np.pi)
        return 1.0 - 2.0 * np.exp(-elapsed / max(0.05, t1))

    def recovery_at(self, t, t1):
        elapsed = t - self.RF_DURATION_S
        if self.mode == "saturation":
            return self.saturation_recovery(elapsed, t1)
        return self.inversion_recovery(elapsed, t1)

    def compute_current_state(self):
        t = float(self.t)
        elapsed = t - self.RF_DURATION_S

        self.ideal_mx = self.recovery_at(t, float(self.t1_s_var.get()))
        self.short_mx = self.recovery_at(t, self.short_t1_s())
        self.long_mx = self.recovery_at(t, self.long_t1_s())
        self.current_mx = 0.5 * (self.short_mx + self.long_mx)

        if self.rf_active():
            progress = np.clip(t / self.RF_DURATION_S, 0.0, 1.0)
            angle = progress * (np.pi / 2.0 if self.mode == "saturation" else np.pi)
            base_vec = np.array([np.cos(angle), np.sin(angle), 0.0])
            self.short_vec = base_vec.copy()
            self.long_vec = base_vec.copy()
            self.net_vec = base_vec.copy()
            return

        # After RF, show longitudinal recovery back toward B0.
        # A small transverse remnant after saturation makes the motion easier to see.
        transverse_tail = 0.16 * np.exp(-max(0.0, elapsed) / 0.35) if self.mode == "saturation" else 0.0

        self.short_vec = np.array([self.short_mx, transverse_tail, 0.0])
        self.long_vec = np.array([self.long_mx, transverse_tail, 0.0])
        self.net_vec = 0.5 * (self.short_vec + self.long_vec)

    # ============================================================
    # Buttons
    # ============================================================

    def start_saturation(self):
        self.mode = "saturation"
        self.running = True
        self.t = 0.0
        self.compute_current_state()
        self.redraw_all()

    def start_inversion(self):
        self.mode = "inversion"
        self.running = True
        self.t = 0.0
        self.compute_current_state()
        self.redraw_all()

    def toggle_pause(self):
        self.running = not self.running
        self.redraw_all()

    def reset_simulation(self):
        self.running = False
        self.t = 0.0
        self.compute_current_state()
        self.redraw_all()

    def reset_3d_view(self):
        self.azim_3d = -56
        self.redraw_all()

    def _step_simulation(self):
        if self.auto_rotate_var.get():
            self.azim_3d += 0.30

        if not self.running:
            return

        self.t += float(self.dt_s_var.get())

        if self.t > float(self.max_time_s_var.get()):
            self.running = False
            if self.auto_restart_var.get():
                if self.mode == "saturation":
                    self.start_saturation()
                else:
                    self.start_inversion()
            return

        self.compute_current_state()

    def _animate_loop(self):
        self._step_simulation()
        self.redraw_all()
        self.root.after(35, self._animate_loop)

    # ============================================================
    # Drawing
    # ============================================================

    def redraw_all(self):
        self.update_slider_labels()
        self.update_status()
        self.draw_curve_plot()
        self.draw_3d_animation()
        self.draw_static_guide_panel()
        self.canvas.draw_idle()

    def update_status(self):
        if self.rf_active() and self.mode == "saturation":
            status = "90° RF saturation pulse ON"
        elif self.rf_active() and self.mode == "inversion":
            status = "180° RF inversion pulse ON"
        elif self.mode == "saturation":
            status = "Longitudinal recovery after saturation"
        else:
            status = "Longitudinal recovery after inversion"

        self.status_label.config(
            text=(
                f"Mode: {self.mode.upper()}\n"
                f"Running: {self.running}\n"
                f"Time: {self.t:.2f} s\n"
                f"Status: {status}\n"
                f"Reference T1 Mx: {self.ideal_mx:+.3f}\n"
                f"Short T1 Mx: {self.short_mx:+.3f}\n"
                f"Long T1 Mx: {self.long_mx:+.3f}\n"
                f"Net Mx: {self.current_mx:+.3f}"
            )
        )

    def draw_curve_plot(self):
        ax = self.ax_curve
        ax.clear()
        ax.set_facecolor("white")

        max_t = float(self.max_time_s_var.get())
        current_t = max(0.001, float(self.t))
        t = np.linspace(0, current_t, 700)

        y_ref = np.array([self.recovery_at(ti, float(self.t1_s_var.get())) for ti in t])
        y_short = np.array([self.recovery_at(ti, self.short_t1_s()) for ti in t])
        y_long = np.array([self.recovery_at(ti, self.long_t1_s()) for ti in t])

        mode_title = "90° saturation recovery" if self.mode == "saturation" else "180° inversion recovery"
        ax.set_title(f"T1 longitudinal recovery — {mode_title}", fontsize=16, pad=10, weight="bold")

        ax.plot(t, y_ref, color="royalblue", linewidth=2.8, label="Reference T1")
        ax.plot(t, y_short, color="firebrick", linewidth=2.7, label="Short T1")
        ax.plot(t, y_long, color="navy", linewidth=2.7, label="Long T1")

        ax.scatter([t[-1]], [y_ref[-1]], color="royalblue", s=40, zorder=5)
        ax.scatter([t[-1]], [y_short[-1]], color="firebrick", s=40, zorder=5)
        ax.scatter([t[-1]], [y_long[-1]], color="navy", s=40, zorder=5)

        # RF pulse window
        pulse_color = "orange" if self.mode == "saturation" else "orangered"
        pulse_label = "90° RF saturation" if self.mode == "saturation" else "180° RF inversion"
        ax.axvspan(0, self.RF_DURATION_S, color=pulse_color, alpha=0.14)
        ax.text(0.03, 0.92, pulse_label, color=pulse_color, fontsize=12, va="top", weight="bold")

        # Reference T1 marker
        t1 = max(0.05, float(self.t1_s_var.get()))
        if self.RF_DURATION_S + t1 <= max_t:
            ax.axvline(self.RF_DURATION_S + t1, color="royalblue", linestyle="--", linewidth=1.2, alpha=0.45)
            ax.text(self.RF_DURATION_S + t1 + 0.05, 0.72, "T1", color="royalblue", fontsize=11)

        # Inversion null time
        if self.mode == "inversion":
            null_t = self.RF_DURATION_S + t1 * np.log(2.0)
            if null_t <= max_t:
                ax.axvline(null_t, color="green", linestyle="--", linewidth=1.4, alpha=0.65)
                ax.text(null_t + 0.05, 0.08, "TI null", color="green", fontsize=11)

        ax.axhline(0, color="black", linewidth=1.0, alpha=0.25)
        ax.axhline(1.0, color="lightgray", linestyle=":", linewidth=1.0)

        ax.set_xlim(0, max_t)
        ax.set_ylim(-1.08, 1.08)
        ax.set_xlabel("Time (s)", fontsize=13)
        ax.set_ylabel("Longitudinal magnetization Mx", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.20)
        ax.legend(loc="lower right" if self.mode == "inversion" else "upper right", fontsize=11, frameon=True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ============================================================
    # 3D helper functions
    # ============================================================

    def _make_sphere_xyz(self, center, radius, nu=42, nv=28):
        u = np.linspace(0, 2*np.pi, nu)
        v = np.linspace(0, np.pi, nv)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        return x, y, z

    def _plot_sphere(self, ax, center, radius, color, alpha=0.25):
        x, y, z = self._make_sphere_xyz(center, radius, nu=28, nv=18)
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=False)

    def _plot_shaded_sphere(self, ax, center, radius, color, alpha=0.34):
        import matplotlib.colors as mcolors

        x, y, z = self._make_sphere_xyz(center, radius, nu=54, nv=34)

        nx = (x - center[0]) / radius
        ny = (y - center[1]) / radius
        nz = (z - center[2]) / radius

        light = np.array([-0.55, -0.35, 0.75], dtype=float)
        light = light / np.linalg.norm(light)

        intensity = nx * light[0] + ny * light[1] + nz * light[2]
        intensity = np.clip(intensity, -1.0, 1.0)
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

        highlight = center + np.array([-0.08, -0.08, 0.10])
        ax.scatter([highlight[0]], [highlight[1]], [highlight[2]], s=18, color="white", alpha=0.55)

    def _plot_glow_sphere(self, ax, center, base_radius, color):
        for mul, alpha in [(1.0, 0.25), (1.6, 0.13), (2.4, 0.06)]:
            self._plot_sphere(ax, center, base_radius * mul, color, alpha=alpha)

    def _draw_b0_vector_3d(self, ax, origin, b0_strength, label="B0"):
        length = 0.55 + 0.28 * float(b0_strength)
        lw = 3.0 + 0.35 * float(b0_strength)
        ax.plot(
            [origin[0], origin[0] + length],
            [origin[1], origin[1]],
            [origin[2], origin[2]],
            color="red",
            linewidth=lw,
            alpha=0.92,
        )
        ax.quiver(
            origin[0] + length,
            origin[1],
            origin[2],
            0.16,
            0,
            0,
            color="red",
            linewidth=lw,
            arrow_length_ratio=0.55,
        )
        ax.text(origin[0] + length + 0.08, origin[1], origin[2], label, color="red", fontsize=12, weight="bold")

    def draw_3d_animation(self):
        ax = self.ax_3d
        ax.clear()
        ax.set_facecolor("white")

        mode_title = "90° saturation recovery" if self.mode == "saturation" else "180° inversion recovery"
        ax.set_title(f"3D animation: T1 longitudinal recovery after {mode_title}", fontsize=16, pad=10, weight="bold")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.15, 1.15)
        ax.set_zlim(-1.05, 1.05)
        ax.set_xlabel("X / B0 axis", fontsize=13)
        ax.set_ylabel("Y / transverse", fontsize=13)
        ax.set_zlabel("Z / transverse", fontsize=13)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_box_aspect((2.3, 1.6, 1.5))
        ax.view_init(elev=22, azim=self.azim_3d)

        short_pos = np.array([-0.05, -0.45, 0.0])
        long_pos = np.array([-0.05, +0.45, 0.0])

        for center, color, label in [
            (short_pos, "firebrick", ""),
            (long_pos, "royalblue", ""),
        ]:
            self._plot_shaded_sphere(ax, center, radius=0.20, color=color, alpha=0.36)
            ax.text(center[0] - 0.10, center[1] - 0.66, center[2] - 0.08, label, color=color, fontsize=10)

        spin_len = 0.70
        short_vec = spin_len * self.short_vec
        long_vec = spin_len * self.long_vec

        ax.quiver(
            short_pos[0], short_pos[1], short_pos[2],
            short_vec[0], short_vec[1], short_vec[2],
            color="firebrick",
            linewidth=3.6,
            arrow_length_ratio=0.18,
        )
        ax.quiver(
            long_pos[0], long_pos[1], long_pos[2],
            long_vec[0], long_vec[1], long_vec[2],
            color="royalblue",
            linewidth=3.6,
            arrow_length_ratio=0.18,
        )

        # Net vector
        net_origin = np.array([0.35, 0.0, 0.0])
        net_vec = spin_len * self.net_vec
        ax.quiver(
            net_origin[0], net_origin[1], net_origin[2],
            net_vec[0], net_vec[1], net_vec[2],
            color="black",
            linewidth=4.2,
            arrow_length_ratio=0.16,
        )
        ax.text(
            net_origin[0] + net_vec[0] + 0.02,
            net_origin[1] + net_vec[1] + 0.03,
            net_origin[2] + net_vec[2],
            "Net Mx",
            color="black",
            fontsize=11,
        )

        self._draw_b0_vector_3d(ax, np.array([-1.15, -0.95, -0.82]), self.b0_strength_var.get(), label="B0")

        # RF pulse direction + glow
        if self.rf_active():
            if self.mode == "saturation":
                b1_len = 0.48 + 0.34 * float(self.b1_90_strength_var.get())
                origin = np.array([-0.95, -0.02, -0.72])
                self._plot_glow_sphere(ax, origin + np.array([0.0, 0.0, b1_len * 0.55]), 0.10, "orange")
                ax.quiver(
                    origin[0],
                    origin[1],
                    origin[2],
                    0.0,
                    0.0,
                    b1_len,
                    color="orange",
                    linewidth=5.4,
                    alpha=1.0,
                    arrow_length_ratio=0.16,
                )
                ax.text(origin[0] + 0.02, origin[1], origin[2] + b1_len + 0.05, "90° RF B1: saturation", color="darkorange", fontsize=11, weight="bold")
                ax.text2D(0.72, 0.88, "90° RF SATURATION", transform=ax.transAxes, fontsize=16, color="darkorange", weight="bold")

            else:
                b1_len = 0.55 + 0.35 * float(self.b1_180_strength_var.get())
                origin = np.array([-0.95, 0.78, -0.55])
                self._plot_glow_sphere(ax, origin + np.array([0.0, b1_len * 0.55, 0.0]), 0.12, "orangered")
                ax.quiver(
                    origin[0],
                    origin[1],
                    origin[2],
                    0.0,
                    b1_len,
                    0.0,
                    color="orangered",
                    linewidth=5.8,
                    alpha=1.0,
                    arrow_length_ratio=0.16,
                )
                ax.text(origin[0] + 0.02, origin[1] + b1_len + 0.05, origin[2], "180° RF B1: inversion", color="orangered", fontsize=11, weight="bold")
                ax.text2D(0.72, 0.88, "180° RF INVERSION", transform=ax.transAxes, fontsize=16, color="orangered", weight="bold")

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

        ax.annotate("", xy=(0.82, 0.77), xytext=(0.18, 0.77), arrowprops=dict(arrowstyle="->", color="red", lw=3))
        ax.text(0.18, 0.81, "Main B0", fontsize=12, color="red", weight="bold")

        ax.text(0.16, 0.58, "90° RF saturation", fontsize=12, color="darkorange", weight="bold")
        ax.text(0.16, 0.52, "Mx → 0, then recovers", fontsize=10, color="black")

        ax.text(0.16, 0.42, "180° RF inversion", fontsize=12, color="orangered", weight="bold")
        ax.text(0.16, 0.36, "Mx → -M0, then recovers", fontsize=10, color="black")

        c1 = Circle((0.25, 0.22), 0.060, color="firebrick", alpha=0.28)
        c2 = Circle((0.25, 0.10), 0.060, color="royalblue", alpha=0.28)
        ax.add_patch(c1)
        ax.add_patch(c2)

        ax.text(0.34, 0.22, "Short T1: faster", color="firebrick", fontsize=11, va="center")
        ax.text(0.34, 0.10, "Long T1: slower", color="royalblue", fontsize=11, va="center")


def main():
    root = tk.Tk()
    app = T1RelaxationLab(root)
    root.mainloop()


if __name__ == "__main__":
    main()
