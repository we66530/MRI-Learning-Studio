
import tkinter as tk
from tkinter import ttk
import numpy as np

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MRISequentialT1T2DetectionDemo:
    """
    Single-sequence MRI receiver detection demo.

    Goal:
        Show T2-like detection and T1 readout on the SAME timeline.

    Sequence:
        t = 0:
            90° RF excitation
            M is tipped from B0 / +X into transverse plane.
            Receiver detects decaying oscillating voltage: T2 / FID.

        Between first FID and second readout:
            Transverse signal decays.
            Longitudinal Mx recovers silently.
            Receiver coil sees almost no signal because Mx is parallel to B0.

        t = TR:
            Second 90° RF readout pulse
            Recovered Mx is converted into Mxy.
            Receiver detects another voltage burst.
            This burst amplitude indirectly reflects T1 recovery.

    Coordinate:
        B0 axis = +X
        Transverse plane = Y-Z
    """

    RF1_DURATION = 0.12
    RF2_DURATION = 0.10

    def __init__(self, root):
        self.root = root
        self.root.title("MRI Sequential T2 and T1 Receiver Detection Demo")
        self.root.geometry("1740x980")

        # ----------------------------
        # Controls
        # ----------------------------
        self.t2_s_var = tk.DoubleVar(value=0.65)
        self.t1_s_var = tk.DoubleVar(value=1.45)
        self.tr_s_var = tk.DoubleVar(value=1.75)
        self.max_time_s_var = tk.DoubleVar(value=4.00)
        self.dt_s_var = tk.DoubleVar(value=0.012)
        self.precession_speed_var = tk.DoubleVar(value=28.0)
        self.b0_strength_var = tk.DoubleVar(value=1.5)
        self.auto_rotate_var = tk.BooleanVar(value=False)
        self.auto_restart_var = tk.BooleanVar(value=False)

        # ----------------------------
        # State
        # ----------------------------
        self.running = False
        self.t = 0.0
        self.azim_3d = -58

        self.m_vec = np.array([1.0, 0.0, 0.0])
        self.mx = 1.0
        self.mxy = 0.0
        self.receiver_voltage = 0.0
        self.coil_glow = 0.0
        self.rf_label = ""

        self.time_history = []
        self.voltage_history = []
        self.mxy_history = []
        self.mx_history = []

        self._build_ui()
        self.reset_simulation()
        self.redraw_all()
        self._animate_loop()

    # ============================================================
    # UI
    # ============================================================

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        main.columnconfigure(0, weight=2, minsize=330)
        main.columnconfigure(1, weight=8)
        main.rowconfigure(0, weight=1)

        controls = ttk.Frame(main, padding=10, width=330)
        controls.grid(row=0, column=0, sticky="nsew")
        controls.grid_propagate(False)

        plots = ttk.Frame(main)
        plots.grid(row=0, column=1, sticky="nsew")

        ttk.Label(controls, text="T1/T2 detection", font=("Arial", 18, "bold")).pack(anchor="w", pady=(0, 10))

        desc = ttk.Label(
            controls,
            text=(
                "One continuous sequence:\n\n"
                "1) First 90° RF creates Mxy.\n"
                "   Receiver detects T2/FID directly.\n\n"
                "2) Mx recovers silently along B0.\n"
                "   Receiver does not directly see it.\n\n"
                "3) Second 90° RF converts recovered Mx to Mxy.\n"
                "   Receiver burst indirectly reflects T1."
            ),
            justify=tk.LEFT,
            wraplength=295,
            font=("Arial", 10),
        )
        desc.pack(anchor="w", pady=(0, 12))

        ttk.Separator(controls).pack(fill=tk.X, pady=8)

        self._add_slider(controls, "T2 decay time", self.t2_s_var, 0.20, 2.50)
        self._add_slider(controls, "T1 recovery time", self.t1_s_var, 0.30, 4.00)
        self._add_slider(controls, "TR / second 90° time", self.tr_s_var, 0.60, 3.50)
        self._add_slider(controls, "Visual precession speed", self.precession_speed_var, 8.0, 70.0)
        self._add_slider(controls, "Main B0 strength", self.b0_strength_var, 0.3, 3.0)
        self._add_slider(controls, "dt per frame", self.dt_s_var, 0.004, 0.050)
        self._add_slider(controls, "Max time", self.max_time_s_var, 2.00, 8.00)

        ttk.Separator(controls).pack(fill=tk.X, pady=8)

        ttk.Button(controls, text="Start / Restart", command=self.start_animation).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(controls, text="Pause / Resume", command=self.toggle_pause).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(controls, text="Reset", command=self.reset_simulation).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(controls, text="Reset 3D View", command=self.reset_3d_view).pack(anchor="w", fill=tk.X, pady=2)

        ttk.Separator(controls).pack(fill=tk.X, pady=8)

        ttk.Checkbutton(controls, text="Auto rotate 3D", variable=self.auto_rotate_var).pack(anchor="w", pady=2)
        ttk.Checkbutton(controls, text="Auto restart", variable=self.auto_restart_var).pack(anchor="w", pady=2)

        self.status_label = ttk.Label(controls, text="", justify=tk.LEFT, wraplength=295, font=("Arial", 10))
        self.status_label.pack(anchor="w", pady=(12, 0))

        # Three simultaneous plots only.
        self.fig = Figure(figsize=(14.8, 9.2), dpi=100, facecolor="white")
        gs = self.fig.add_gridspec(
            3, 1,
            height_ratios=[1.45, 1.0, 1.0],
            hspace=0.38,
        )

        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection="3d")
        self.ax_m = self.fig.add_subplot(gs[1, 0])
        self.ax_scope = self.fig.add_subplot(gs[2, 0])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plots)
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

    def update_slider_labels(self):
        for var in [
            self.t2_s_var,
            self.t1_s_var,
            self.tr_s_var,
            self.precession_speed_var,
            self.b0_strength_var,
            self.dt_s_var,
            self.max_time_s_var,
        ]:
            name = var._label_name
            val = var.get()

            if name == "Visual precession speed":
                text = f"{name}: {val:.1f} rad/s"
            elif name == "Main B0 strength":
                text = f"{name}: {val:.2f}"
            else:
                text = f"{name}: {val:.3f} s"

            var._label_widget.config(text=text)

    def on_param_change(self):
        # Keep max time beyond second readout.
        min_max = float(self.tr_s_var.get()) + 1.2
        if self.max_time_s_var.get() < min_max:
            self.max_time_s_var.set(min_max)

        self.update_slider_labels()
        if not self.running:
            self.compute_state()
        self.redraw_all()

    # ============================================================
    # Timeline model
    # ============================================================

    def is_first_rf(self, t=None):
        if t is None:
            t = self.t
        return 0.0 <= t <= self.RF1_DURATION

    def is_second_rf(self, t=None):
        if t is None:
            t = self.t
        tr = float(self.tr_s_var.get())
        return tr <= t <= tr + self.RF2_DURATION

    def compute_values_at(self, t):
        """
        Returns:
            mx, mxy, voltage, m_vec, label
        """
        t2 = max(0.05, float(self.t2_s_var.get()))
        t1 = max(0.05, float(self.t1_s_var.get()))
        tr = float(self.tr_s_var.get())
        omega = float(self.precession_speed_var.get())

        # Stage 1: first RF excitation: M moves +X -> transverse +Y.
        if self.is_first_rf(t):
            p = np.clip(t / self.RF1_DURATION, 0.0, 1.0)
            angle = p * np.pi / 2.0
            mx = np.cos(angle)
            mxy = np.sin(angle)
            m_vec = np.array([mx, mxy, 0.0])
            voltage = 0.0
            label = "1st 90° RF: excitation"
            return mx, mxy, voltage, m_vec, label

        # Stage 2: after first RF, transverse precession / T2 FID.
        # We also let longitudinal recovery begin from near zero.
        if t < tr:
            elapsed = t - self.RF1_DURATION
            mxy = np.exp(-elapsed / t2)
            phase = omega * elapsed
            mx = 1.0 - np.exp(-elapsed / t1)

            # Show the actual animated vector mostly as transverse FID at first,
            # then gradually dominated by longitudinal recovery as Mxy decays.
            m_vec = np.array([mx, mxy * np.cos(phase), mxy * np.sin(phase)])
            voltage = mxy * np.cos(phase)
            label = "T2/FID detected directly; Mx recovers silently"
            return mx, mxy, voltage, m_vec, label

        # Stage 3: second RF readout converts recovered Mx into transverse Mxy.
        mx_at_tr = 1.0 - np.exp(-(tr - self.RF1_DURATION) / t1)

        if self.is_second_rf(t):
            p = np.clip((t - tr) / self.RF2_DURATION, 0.0, 1.0)
            angle = p * np.pi / 2.0
            mx = mx_at_tr * np.cos(angle)
            mxy = mx_at_tr * np.sin(angle)
            m_vec = np.array([mx, mxy, 0.0])
            voltage = 0.0
            label = "2nd 90° RF: recovered Mx → Mxy"
            return mx, mxy, voltage, m_vec, label

        # Stage 4: second received burst, amplitude depends on recovered Mx at TR.
        elapsed2 = t - (tr + self.RF2_DURATION)
        mxy = mx_at_tr * np.exp(-elapsed2 / t2)
        phase = omega * elapsed2
        mx = 1.0 - np.exp(-(t - self.RF1_DURATION) / t1)

        m_vec = np.array([0.0, mxy * np.cos(phase), mxy * np.sin(phase)])
        voltage = mxy * np.cos(phase)
        label = "T1 readout burst: amplitude reflects recovered Mx"
        return mx, mxy, voltage, m_vec, label

    def compute_state(self):
        self.mx, self.mxy, self.receiver_voltage, self.m_vec, self.rf_label = self.compute_values_at(float(self.t))
        self.coil_glow = min(1.0, 0.12 + abs(self.receiver_voltage) + 0.20 * self.mxy)

    # ============================================================
    # Run controls
    # ============================================================

    def start_animation(self):
        self.running = True
        self.t = 0.0
        self.time_history = []
        self.voltage_history = []
        self.mxy_history = []
        self.mx_history = []
        self.compute_state()
        self.redraw_all()

    def toggle_pause(self):
        self.running = not self.running
        self.redraw_all()

    def reset_simulation(self):
        self.running = False
        self.t = 0.0
        self.time_history = []
        self.voltage_history = []
        self.mxy_history = []
        self.mx_history = []
        self.compute_state()
        self.redraw_all()

    def reset_3d_view(self):
        self.azim_3d = -58
        self.redraw_all()

    def _step(self):
        if self.auto_rotate_var.get():
            self.azim_3d += 0.25

        if not self.running:
            return

        self.t += float(self.dt_s_var.get())

        if self.t > float(self.max_time_s_var.get()):
            self.running = False
            if self.auto_restart_var.get():
                self.start_animation()
            return

        self.compute_state()

        self.time_history.append(self.t)
        self.voltage_history.append(self.receiver_voltage)
        self.mxy_history.append(self.mxy)
        self.mx_history.append(self.mx)

    def _animate_loop(self):
        self._step()
        self.redraw_all()
        self.root.after(28, self._animate_loop)

    # ============================================================
    # Drawing
    # ============================================================

    def redraw_all(self):
        self.update_slider_labels()
        self.update_status()
        self.draw_3d_panel()
        self.draw_magnetization_plot()
        self.draw_receiver_plot()
        self.canvas.draw_idle()

    def update_status(self):
        self.status_label.config(
            text=(
                f"Running: {self.running}\n"
                f"Time: {self.t:.2f} s\n"
                f"Stage: {self.rf_label}\n\n"
                f"Mx longitudinal: {self.mx:+.3f}\n"
                f"Mxy transverse: {self.mxy:+.3f}\n"
                f"Receiver voltage: {self.receiver_voltage:+.3f}\n\n"
                "Key point:\n"
                "T2 is detected directly from Mxy decay.\n"
                "T1 is detected indirectly after a second 90° RF converts Mx into Mxy."
            )
        )

    def _make_sphere_xyz(self, center, radius, nu=42, nv=28):
        u = np.linspace(0, 2*np.pi, nu)
        v = np.linspace(0, np.pi, nv)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        return x, y, z

    def _plot_shaded_sphere(self, ax, center, radius, color, alpha=0.28):
        import matplotlib.colors as mcolors

        x, y, z = self._make_sphere_xyz(center, radius, nu=54, nv=34)
        nx = (x - center[0]) / radius
        ny = (y - center[1]) / radius
        nz = (z - center[2]) / radius

        light = np.array([-0.55, -0.35, 0.75], dtype=float)
        light /= np.linalg.norm(light)

        intensity = np.clip(nx * light[0] + ny * light[1] + nz * light[2], -1.0, 1.0)
        shade = 0.52 + 0.48 * (intensity + 1.0) / 2.0

        base = np.array(mcolors.to_rgb(color))
        rgb = np.clip(base[None, None, :] * shade[:, :, None] + 0.18 * (1.0 - shade[:, :, None]), 0, 1)

        rgba = np.zeros((*rgb.shape[:2], 4))
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = alpha

        ax.plot_surface(x, y, z, facecolors=rgba, linewidth=0.25, edgecolor=(0, 0, 0, 0.08), shade=False)

    def draw_3d_panel(self):
        ax = self.ax_3d
        ax.clear()
        ax.set_facecolor("white")

        ax.set_title("Shared 3D animation: same magnetization, same timeline", fontsize=16, weight="bold", pad=8)
        ax.set_xlim(-1.35, 1.45)
        ax.set_ylim(-1.15, 1.15)
        ax.set_zlim(-1.05, 1.05)
        ax.set_xlabel("X / B0 axis", fontsize=12)
        ax.set_ylabel("Y / transverse", fontsize=12)
        ax.set_zlabel("Z / transverse", fontsize=12)
        ax.set_box_aspect((2.3, 1.7, 1.5))
        ax.view_init(elev=22, azim=self.azim_3d)

        sample = np.array([-0.25, 0.0, 0.0])
        self._plot_shaded_sphere(ax, sample, 0.24, "royalblue", alpha=0.22)

        # B0 vector; length follows B0 strength.
        b0_len = 0.55 + 0.28 * float(self.b0_strength_var.get())
        ax.quiver(-1.15, -1.02, -0.88, b0_len, 0, 0, color="red", linewidth=3.6, arrow_length_ratio=0.15)
        ax.text(-1.15 + b0_len + 0.05, -1.02, -0.88, "B0", color="red", fontsize=12, weight="bold")

        # Transverse circle.
        th = np.linspace(0, 2*np.pi, 240)
        r = 0.55
        ax.plot(np.full_like(th, sample[0]), sample[1] + r*np.cos(th), sample[2] + r*np.sin(th),
                color="#d8d8d8", linewidth=1.1, alpha=0.9)

        # Magnetization vector.
        ax.quiver(sample[0], sample[1], sample[2],
                  self.m_vec[0], self.m_vec[1], self.m_vec[2],
                  color="black", linewidth=4.5, arrow_length_ratio=0.16)
        ax.text(sample[0] + self.m_vec[0] + 0.04,
                sample[1] + self.m_vec[1] + 0.04,
                sample[2] + self.m_vec[2],
                "M", color="black", fontsize=12, weight="bold")

        # Mx and Mxy components.
        ax.quiver(sample[0], sample[1], sample[2], self.mx, 0, 0,
                  color="crimson", linewidth=2.7, arrow_length_ratio=0.14, alpha=0.80)
        ax.text(sample[0] + self.mx + 0.04, sample[1] - 0.08, sample[2] - 0.06, "Mx", color="crimson", fontsize=11)

        if self.mxy > 0.02:
            trans = np.array([0, self.m_vec[1], self.m_vec[2]])
            ax.quiver(sample[0], sample[1], sample[2], trans[0], trans[1], trans[2],
                      color="navy", linewidth=2.7, arrow_length_ratio=0.14, alpha=0.80)
            ax.text(sample[0], sample[1] + trans[1] + 0.05, sample[2] + trans[2], "Mxy", color="navy", fontsize=11)

        # Receiver coil.
        coil_x = 0.92
        coil_r = 0.58
        glow = self.coil_glow
        coil_color = "orange" if glow > 0.23 else "gray"
        coil_lw = 2.0 + 4.0 * glow
        ax.plot(np.full_like(th, coil_x), coil_r*np.cos(th), coil_r*np.sin(th),
                color=coil_color, linewidth=coil_lw, alpha=0.55 + 0.45 * glow)
        ax.text(coil_x, -0.75, 0.75, "Receive coil", color=coil_color, fontsize=12, weight="bold")

        # RF cue.
        if self.is_first_rf():
            ax.text2D(0.73, 0.88, "1st 90° RF\nExcitation", transform=ax.transAxes,
                      color="darkorange", fontsize=16, weight="bold")
        elif self.is_second_rf():
            ax.text2D(0.68, 0.88, "2nd 90° RF\nT1 readout", transform=ax.transAxes,
                      color="darkorange", fontsize=16, weight="bold")

        ax.xaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        ax.yaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        ax.zaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))

    def draw_magnetization_plot(self):
        ax = self.ax_m
        ax.clear()
        ax.set_facecolor("white")

        ax.set_title("Magnetization components on the same timeline", fontsize=15, weight="bold", pad=8)
        ax.set_xlim(0, float(self.max_time_s_var.get()))
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Magnitude", fontsize=12)
        ax.grid(True, alpha=0.22)

        if len(self.time_history) > 1:
            ax.plot(self.time_history, self.mx_history, color="crimson", linewidth=2.4, label="Mx / longitudinal recovery")
            ax.plot(self.time_history, self.mxy_history, color="navy", linewidth=2.4, label="Mxy / transverse signal")
            ax.scatter([self.time_history[-1]], [self.mx_history[-1]], color="crimson", s=35)
            ax.scatter([self.time_history[-1]], [self.mxy_history[-1]], color="navy", s=35)

        # RF windows and labels, no moving scanner line.
        ax.axvspan(0, self.RF1_DURATION, color="orange", alpha=0.13)
        ax.text(0.03, 1.02, "1st 90° RF", color="darkorange", fontsize=11, weight="bold", va="top")

        tr = float(self.tr_s_var.get())
        ax.axvspan(tr, tr + self.RF2_DURATION, color="orange", alpha=0.13)
        ax.text(tr + 0.02, 1.02, "2nd 90° RF", color="darkorange", fontsize=11, weight="bold", va="top")

        ax.text(0.40, 0.78, "T2/FID detected here", color="navy", fontsize=11, weight="bold")
        ax.text(0.72, 0.20, "silent T1 recovery", color="dimgray", fontsize=11)
        ax.text(tr + 0.20, 0.70, "T1 readout after conversion", color="crimson", fontsize=11, weight="bold")

        ax.legend(loc="upper right", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def draw_receiver_plot(self):
        ax = self.ax_scope
        ax.clear()
        ax.set_facecolor("white")

        ax.set_title("Oscilloscope: receiver coil voltage", fontsize=15, weight="bold", pad=8)
        ax.set_xlim(0, float(self.max_time_s_var.get()))
        ax.set_ylim(-1.10, 1.10)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Voltage", fontsize=12)
        ax.grid(True, alpha=0.22)

        if len(self.time_history) > 1:
            ax.plot(self.time_history, self.voltage_history, color="navy", linewidth=2.4, label="Receiver voltage")
            ax.plot(self.time_history, self.mxy_history, color="gray", linestyle="--", linewidth=1.5, alpha=0.72, label="Mxy envelope")
            ax.scatter([self.time_history[-1]], [self.voltage_history[-1]], color="navy", s=45)

        ax.axvspan(0, self.RF1_DURATION, color="orange", alpha=0.13)
        ax.text(0.03, 0.96, "1st 90° RF", color="darkorange", fontsize=11, weight="bold", va="top")

        tr = float(self.tr_s_var.get())
        ax.axvspan(tr, tr + self.RF2_DURATION, color="orange", alpha=0.13)
        ax.text(tr + 0.02, 0.96, "2nd 90° RF", color="darkorange", fontsize=11, weight="bold", va="top")

        ax.text(0.42, 0.78, "T2/FID voltage burst", color="navy", fontsize=11, weight="bold")
        ax.text(0.55, -0.78, "No voltage from pure Mx", color="dimgray", fontsize=11)
        ax.text(tr + 0.22, 0.58, "T1 readout burst", color="crimson", fontsize=11, weight="bold")

        ax.legend(loc="upper right", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def main():
    root = tk.Tk()
    app = MRISequentialT1T2DetectionDemo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
