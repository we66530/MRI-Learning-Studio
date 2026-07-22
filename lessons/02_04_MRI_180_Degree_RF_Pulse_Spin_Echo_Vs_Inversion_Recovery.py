import tkinter as tk
from tkinter import ttk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# 3D comparison of two different roles of a 180° RF pulse
#
# Upper row:
#   Spin-Echo sequence
#   90° excitation
#   -> transverse precession and dephasing
#   -> 180° refocusing
#   -> spin echo
#
# Lower row:
#   Inversion-Recovery sequence
#   180° inversion
#   -> longitudinal T1 recovery
#   -> 90° excitation at TI
#   -> transverse precession / tissue signal
#
# Run:
#   pip install numpy matplotlib
#   python SE_vs_IR_sequence_3D.py
# ============================================================


class Compare180PulseRoles:
    TIMER_MS = 35

    # RF pulse durations are deliberately slowed for visualization.
    SE_90_DURATION = 0.18
    SE_180_DURATION = 0.18

    IR_180_DURATION = 0.20
    IR_90_DURATION = 0.16

    def __init__(self, root):
        self.root = root
        self.root.title("Different Roles of the 180° RF Pulse")
        self.root.geometry("1740x980")

        # --------------------------------------------------------
        # Control variables
        # --------------------------------------------------------
        self.speed_var = tk.DoubleVar(value=1.0)
        self.auto_rotate_var = tk.BooleanVar(value=False)

        # Spin-echo parameters
        self.se_te_var = tk.DoubleVar(value=1.80)
        self.se_t2_var = tk.DoubleVar(value=2.80)

        # Inversion-recovery parameters
        self.ir_ti_var = tk.DoubleVar(value=0.70)
        self.short_t1_var = tk.DoubleVar(value=1.00)
        self.long_t1_var = tk.DoubleVar(value=2.40)
        self.target_var = tk.StringVar(value="Short T1")

        # --------------------------------------------------------
        # Animation state
        # --------------------------------------------------------
        self.running = False
        self.progress = 0.0

        self.se_azim = -55.0
        self.ir_azim = -55.0

        self.n_spins = 19

        # Static B0 inhomogeneity / frequency offsets.
        self.frequency_offsets = np.linspace(
            -5.2,
            5.2,
            self.n_spins
        )

        self.ui_ready = False

        self._build_ui()

        self.ui_ready = True
        self.redraw_all()
        self._animation_loop()

    # ============================================================
    # UI
    # ============================================================

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        main.columnconfigure(0, weight=2, minsize=335)
        main.columnconfigure(1, weight=8)
        main.rowconfigure(0, weight=1)

        control = ttk.Frame(
            main,
            padding=12
        )
        control.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        plot_frame = ttk.Frame(main)
        plot_frame.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        ttk.Label(
            control,
            text="180° RF Pulse",
            font=("Arial", 18, "bold")
        ).pack(anchor="w", pady=(0, 12))

        ttk.Button(
            control,
            text="Start / Pause",
            command=self.toggle_animation
        ).pack(fill=tk.X, pady=3)

        ttk.Button(
            control,
            text="Restart",
            command=self.restart_animation
        ).pack(fill=tk.X, pady=3)

        ttk.Button(
            control,
            text="Reset",
            command=self.reset_animation
        ).pack(fill=tk.X, pady=3)

        ttk.Button(
            control,
            text="Reset 3D View",
            command=self.reset_3d_view
        ).pack(fill=tk.X, pady=3)

        ttk.Checkbutton(
            control,
            text="Auto rotate 3D view",
            variable=self.auto_rotate_var
        ).pack(anchor="w", pady=(7, 2))

        ttk.Separator(control).pack(
            fill=tk.X,
            pady=10
        )

        self._add_slider(
            control,
            "Animation speed",
            self.speed_var,
            0.25,
            3.0,
            "{:.2f}×"
        )

        ttk.Separator(control).pack(
            fill=tk.X,
            pady=10
        )

        ttk.Label(
            control,
            text="Spin Echo",
            font=("Arial", 13, "bold")
        ).pack(anchor="w")

        self._add_slider(
            control,
            "Echo time, TE",
            self.se_te_var,
            1.0,
            3.0,
            "{:.2f} s"
        )

        self._add_slider(
            control,
            "True T2",
            self.se_t2_var,
            1.5,
            5.0,
            "{:.2f} s"
        )

        ttk.Separator(control).pack(
            fill=tk.X,
            pady=10
        )

        ttk.Label(
            control,
            text="Inversion Recovery",
            font=("Arial", 13, "bold")
        ).pack(anchor="w")

        ttk.Label(
            control,
            text="Target tissue"
        ).pack(anchor="w", pady=(5, 0))

        target_combo = ttk.Combobox(
            control,
            textvariable=self.target_var,
            values=[
                "Short T1",
                "Long T1"
            ],
            state="readonly"
        )
        target_combo.pack(
            fill=tk.X,
            pady=(0, 7)
        )
        target_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.redraw_all()
        )

        self._add_slider(
            control,
            "Inversion time, TI",
            self.ir_ti_var,
            0.10,
            2.80,
            "{:.2f} s"
        )

        self._add_slider(
            control,
            "Short T1",
            self.short_t1_var,
            0.40,
            1.60,
            "{:.2f} s"
        )

        self._add_slider(
            control,
            "Long T1",
            self.long_t1_var,
            1.40,
            3.80,
            "{:.2f} s"
        )

        ttk.Button(
            control,
            text="Null Short T1",
            command=self.set_short_t1_null
        ).pack(fill=tk.X, pady=3)

        ttk.Button(
            control,
            text="Null Long T1",
            command=self.set_long_t1_null
        ).pack(fill=tk.X, pady=3)

        ttk.Separator(control).pack(
            fill=tk.X,
            pady=10
        )

        self.status_label = ttk.Label(
            control,
            text="",
            justify=tk.LEFT,
            wraplength=305,
            font=("Arial", 10)
        )
        self.status_label.pack(
            anchor="w",
            pady=(4, 0)
        )

        # --------------------------------------------------------
        # Figure
        # --------------------------------------------------------
        self.fig = Figure(
            figsize=(14.2, 9.1),
            dpi=100,
            facecolor="white"
        )

        gs = self.fig.add_gridspec(
            2,
            2,
            width_ratios=[1.12, 1.48],
            height_ratios=[1.0, 1.0],
            hspace=0.30,
            wspace=0.20
        )

        # Both left panels are now 3D.
        self.ax_se_3d = self.fig.add_subplot(
            gs[0, 0],
            projection="3d"
        )

        self.ax_se_signal = self.fig.add_subplot(
            gs[0, 1]
        )

        self.ax_ir_3d = self.fig.add_subplot(
            gs[1, 0],
            projection="3d"
        )

        self.ax_ir_curve = self.fig.add_subplot(
            gs[1, 1]
        )

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
        frame.pack(
            fill=tk.X,
            pady=6
        )

        value_label = ttk.Label(
            frame,
            text=(
                f"{name}: "
                f"{value_format.format(variable.get())}"
            )
        )
        value_label.pack(anchor="w")

        def slider_changed(_value=None):
            value_label.config(
                text=(
                    f"{name}: "
                    f"{value_format.format(variable.get())}"
                )
            )

            if self.ui_ready:
                self.redraw_all()

        slider = ttk.Scale(
            frame,
            from_=minimum,
            to=maximum,
            variable=variable,
            orient=tk.HORIZONTAL,
            command=slider_changed
        )
        slider.pack(fill=tk.X)

    # ============================================================
    # Controls
    # ============================================================

    def toggle_animation(self):
        if self.progress >= 1.0:
            self.progress = 0.0

        self.running = not self.running

    def restart_animation(self):
        self.progress = 0.0
        self.running = True
        self.redraw_all()

    def reset_animation(self):
        self.running = False
        self.progress = 0.0
        self.redraw_all()

    def reset_3d_view(self):
        self.se_azim = -55.0
        self.ir_azim = -55.0
        self.redraw_all()

    def set_short_t1_null(self):
        self.target_var.set("Short T1")

        self.ir_ti_var.set(
            float(self.short_t1_var.get())
            * np.log(2.0)
        )

        self.restart_animation()

    def set_long_t1_null(self):
        self.target_var.set("Long T1")

        self.ir_ti_var.set(
            float(self.long_t1_var.get())
            * np.log(2.0)
        )

        self.restart_animation()

    def _animation_loop(self):
        if self.auto_rotate_var.get():
            self.se_azim += 0.30
            self.ir_azim += 0.30

            if not self.running:
                self.redraw_all()

        if self.running:
            increment = (
                0.0045
                * float(self.speed_var.get())
            )

            self.progress += increment

            if self.progress >= 1.0:
                self.progress = 1.0
                self.running = False

            self.redraw_all()

        self.root.after(
            self.TIMER_MS,
            self._animation_loop
        )

    # ============================================================
    # Shared 3D drawing helpers
    # ============================================================

    @staticmethod
    def _sphere_xyz(
        center,
        radius,
        nu=34,
        nv=20
    ):
        u = np.linspace(
            0,
            2 * np.pi,
            nu
        )

        v = np.linspace(
            0,
            np.pi,
            nv
        )

        x = (
            center[0]
            + radius
            * np.outer(
                np.cos(u),
                np.sin(v)
            )
        )

        y = (
            center[1]
            + radius
            * np.outer(
                np.sin(u),
                np.sin(v)
            )
        )

        z = (
            center[2]
            + radius
            * np.outer(
                np.ones_like(u),
                np.cos(v)
            )
        )

        return x, y, z

    def _draw_bloch_sphere(
        self,
        ax,
        center,
        radius=1.0,
        color="gray",
        alpha=0.15
    ):
        x, y, z = self._sphere_xyz(
            center,
            radius
        )

        ax.plot_wireframe(
            x,
            y,
            z,
            color=color,
            linewidth=0.45,
            alpha=alpha
        )

        # Transverse precession circle.
        theta = np.linspace(
            0,
            2 * np.pi,
            180
        )

        ax.plot(
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            np.full_like(theta, center[2]),
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.35
        )

    @staticmethod
    def _draw_vector_3d(
        ax,
        origin,
        vector,
        color,
        linewidth=3.0,
        alpha=1.0,
        label=None
    ):
        vector = np.asarray(
            vector,
            dtype=float
        )

        if np.linalg.norm(vector) < 1e-6:
            return

        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            vector[0],
            vector[1],
            vector[2],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            arrow_length_ratio=0.15
        )

        if label:
            tip = (
                np.asarray(origin)
                + vector
            )

            ax.text(
                tip[0],
                tip[1],
                tip[2] + 0.05,
                label,
                color=color,
                fontsize=9
            )

    @staticmethod
    def _draw_b0_arrow(
        ax,
        origin,
        length=0.75
    ):
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            0,
            0,
            length,
            color="black",
            linewidth=3.0,
            arrow_length_ratio=0.18
        )

        ax.text(
            origin[0],
            origin[1],
            origin[2] + length + 0.05,
            "B0",
            color="black",
            fontsize=10,
            weight="bold"
        )

    @staticmethod
    def _draw_rf_arrow(
        ax,
        origin,
        direction,
        label,
        color
    ):
        direction = np.asarray(
            direction,
            dtype=float
        )

        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color=color,
            linewidth=5.0,
            arrow_length_ratio=0.22
        )

        tip = (
            np.asarray(origin)
            + direction
        )

        ax.text(
            tip[0],
            tip[1],
            tip[2] + 0.06,
            label,
            color=color,
            fontsize=10,
            weight="bold"
        )

    # ============================================================
    # Spin-echo model
    # ============================================================

    def spin_echo_times(self):
        excitation_start = 0.10

        excitation_end = (
            excitation_start
            + self.SE_90_DURATION
        )

        te = float(
            self.se_te_var.get()
        )

        refocus_center = (
            excitation_end
            + te / 2.0
        )

        refocus_start = (
            refocus_center
            - self.SE_180_DURATION / 2.0
        )

        refocus_end = (
            refocus_center
            + self.SE_180_DURATION / 2.0
        )

        echo_time = (
            excitation_end
            + te
        )

        end = (
            echo_time
            + 0.50
        )

        return (
            excitation_start,
            excitation_end,
            refocus_start,
            refocus_center,
            refocus_end,
            echo_time,
            end
        )

    def spin_echo_time(self):
        *_, end = self.spin_echo_times()
        return self.progress * end

    def spin_vectors_3d(self, t):
        (
            excitation_start,
            excitation_end,
            refocus_start,
            _refocus_center,
            refocus_end,
            _echo_time,
            _end
        ) = self.spin_echo_times()

        vectors = []

        # Before the 90° pulse: all spins along +z.
        if t < excitation_start:
            for _offset in self.frequency_offsets:
                vectors.append(
                    np.array([
                        0.0,
                        0.0,
                        1.0
                    ])
                )

            return np.array(vectors)

        # 90° excitation:
        # rotate +z toward +x around the y-axis.
        if t < excitation_end:
            progress = np.clip(
                (
                    t
                    - excitation_start
                )
                / self.SE_90_DURATION,
                0.0,
                1.0
            )

            angle = (
                np.pi
                / 2.0
                * progress
            )

            common_vector = np.array([
                np.sin(angle),
                0.0,
                np.cos(angle)
            ])

            for _offset in self.frequency_offsets:
                vectors.append(
                    common_vector.copy()
                )

            return np.array(vectors)

        t2 = max(
            0.10,
            float(self.se_t2_var.get())
        )

        amplitude = np.exp(
            -(
                t
                - excitation_end
            )
            / t2
        )

        # Free precession and dephasing before 180° pulse.
        if t < refocus_start:
            elapsed = (
                t
                - excitation_end
            )

            for offset in self.frequency_offsets:
                phase = offset * elapsed

                vectors.append(
                    amplitude
                    * np.array([
                        np.cos(phase),
                        np.sin(phase),
                        0.0
                    ])
                )

            return np.array(vectors)

        phase_duration_before_180 = (
            refocus_start
            - excitation_end
        )

        # During 180° pulse:
        # rotate each vector around the x-axis.
        if t < refocus_end:
            pulse_progress = np.clip(
                (
                    t
                    - refocus_start
                )
                / self.SE_180_DURATION,
                0.0,
                1.0
            )

            rotation_angle = (
                np.pi
                * pulse_progress
            )

            for offset in self.frequency_offsets:
                phase_before = (
                    offset
                    * phase_duration_before_180
                )

                x0 = (
                    amplitude
                    * np.cos(phase_before)
                )

                y0 = (
                    amplitude
                    * np.sin(phase_before)
                )

                # Rotation around x:
                # x stays unchanged,
                # y and z rotate.
                vectors.append(
                    np.array([
                        x0,
                        y0 * np.cos(rotation_angle),
                        y0 * np.sin(rotation_angle)
                    ])
                )

            return np.array(vectors)

        # After the 180° pulse:
        # phase order is reversed and spins rephase.
        elapsed_after_180 = (
            t
            - refocus_end
        )

        for offset in self.frequency_offsets:
            phase = (
                -offset
                * phase_duration_before_180
                + offset
                * elapsed_after_180
            )

            vectors.append(
                amplitude
                * np.array([
                    np.cos(phase),
                    np.sin(phase),
                    0.0
                ])
            )

        return np.array(vectors)

    def spin_echo_signal(self, t):
        vectors = self.spin_vectors_3d(t)

        transverse_complex = (
            vectors[:, 0]
            + 1j * vectors[:, 1]
        )

        return np.abs(
            np.mean(transverse_complex)
        )

    def spin_echo_stage(self, t):
        (
            excitation_start,
            excitation_end,
            refocus_start,
            _refocus_center,
            refocus_end,
            echo_time,
            _end
        ) = self.spin_echo_times()

        if t < excitation_start:
            return "Equilibrium along B0"

        if t < excitation_end:
            return "90° excitation"

        if t < refocus_start:
            return "Transverse precession and dephasing"

        if t < refocus_end:
            return "180° transverse refocusing"

        if t < echo_time:
            return "Spins rephasing"

        return "Spin echo"

    # ============================================================
    # Inversion-recovery model
    # ============================================================

    def inversion_times(self):
        inversion_start = 0.10

        recovery_start = (
            inversion_start
            + self.IR_180_DURATION
        )

        ti = float(
            self.ir_ti_var.get()
        )

        excitation_start = (
            recovery_start
            + ti
        )

        excitation_end = (
            excitation_start
            + self.IR_90_DURATION
        )

        end = (
            excitation_end
            + 0.70
        )

        return (
            inversion_start,
            recovery_start,
            excitation_start,
            excitation_end,
            end
        )

    def inversion_time(self):
        *_, end = self.inversion_times()
        return self.progress * end

    @staticmethod
    def inversion_recovery(
        elapsed,
        t1
    ):
        elapsed = max(
            0.0,
            elapsed
        )

        return (
            1.0
            - 2.0
            * np.exp(
                -elapsed
                / max(0.05, t1)
            )
        )

    def ir_vector_3d(
        self,
        t,
        t1,
        phase_offset=0.0
    ):
        (
            inversion_start,
            recovery_start,
            excitation_start,
            excitation_end,
            _end
        ) = self.inversion_times()

        # Before inversion.
        if t < inversion_start:
            return np.array([
                0.0,
                0.0,
                1.0
            ])

        # 180° inversion:
        # rotate +z to -z around x-axis.
        if t < recovery_start:
            progress = np.clip(
                (
                    t
                    - inversion_start
                )
                / self.IR_180_DURATION,
                0.0,
                1.0
            )

            angle = (
                np.pi
                * progress
            )

            return np.array([
                0.0,
                -np.sin(angle),
                np.cos(angle)
            ])

        # Longitudinal T1 recovery.
        if t < excitation_start:
            elapsed = (
                t
                - recovery_start
            )

            mz = self.inversion_recovery(
                elapsed,
                t1
            )

            return np.array([
                0.0,
                0.0,
                mz
            ])

        ti = float(
            self.ir_ti_var.get()
        )

        mz_at_ti = self.inversion_recovery(
            ti,
            t1
        )

        # 90° excitation at TI.
        if t < excitation_end:
            progress = np.clip(
                (
                    t
                    - excitation_start
                )
                / self.IR_90_DURATION,
                0.0,
                1.0
            )

            angle = (
                np.pi
                / 2.0
                * progress
            )

            return np.array([
                0.0,
                -mz_at_ti * np.sin(angle),
                mz_at_ti * np.cos(angle)
            ])

        # Transverse precession after the 90° pulse.
        elapsed_transverse = (
            t
            - excitation_end
        )

        decay = np.exp(
            -elapsed_transverse / 0.65
        )

        phase = (
            6.0
            * elapsed_transverse
            + phase_offset
        )

        # The sign of Mz at TI is retained.
        return decay * np.array([
            mz_at_ti * np.sin(phase),
            -mz_at_ti * np.cos(phase),
            0.0
        ])

    def inversion_stage(self, t):
        (
            inversion_start,
            recovery_start,
            excitation_start,
            excitation_end,
            _end
        ) = self.inversion_times()

        if t < inversion_start:
            return "Equilibrium along B0"

        if t < recovery_start:
            return "180° longitudinal inversion"

        if t < excitation_start:
            return "Different T1 recovery rates"

        if t < excitation_end:
            return "90° excitation at TI"

        return "Transverse signal and precession"

    # ============================================================
    # Redraw
    # ============================================================

    def redraw_all(self):
        if not self.ui_ready:
            return

        se_time = self.spin_echo_time()
        ir_time = self.inversion_time()

        self.draw_spin_echo_3d(
            se_time
        )

        self.draw_spin_echo_signal(
            se_time
        )

        self.draw_inversion_recovery_3d(
            ir_time
        )

        self.draw_inversion_curve(
            ir_time
        )

        self.update_status(
            se_time,
            ir_time
        )

        self.fig.suptitle(
            (
                "The role of a 180° RF pulse depends "
                "on its position in the sequence"
            ),
            fontsize=16,
            weight="bold",
            y=0.985
        )

        self.canvas.draw_idle()

    # ============================================================
    # Upper-left: 3D spin-echo animation
    # ============================================================

    def draw_spin_echo_3d(self, t):
        ax = self.ax_se_3d
        ax.clear()

        ax.set_title(
            "Spin Echo: 180° refocuses transverse phase",
            fontsize=12,
            weight="bold",
            pad=8
        )

        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_zlim(-1.25, 1.25)

        ax.set_box_aspect((1, 1, 0.95))

        ax.set_xlabel("Mx")
        ax.set_ylabel("My")
        ax.set_zlabel("Mz / B0")

        ax.view_init(
            elev=24,
            azim=self.se_azim
        )

        origin = np.array([
            0.0,
            0.0,
            0.0
        ])

        self._draw_bloch_sphere(
            ax,
            origin,
            radius=1.0
        )

        # Coordinate axes.
        ax.plot(
            [-1.15, 1.15],
            [0, 0],
            [0, 0],
            color="lightgray",
            linewidth=1
        )

        ax.plot(
            [0, 0],
            [-1.15, 1.15],
            [0, 0],
            color="lightgray",
            linewidth=1
        )

        ax.plot(
            [0, 0],
            [0, 0],
            [-1.15, 1.15],
            color="gray",
            linewidth=1.2
        )

        self._draw_b0_arrow(
            ax,
            np.array([
                -1.10,
                -1.05,
                -1.05
            ]),
            length=0.70
        )

        vectors = self.spin_vectors_3d(t)

        (
            excitation_start,
            excitation_end,
            refocus_start,
            _refocus_center,
            refocus_end,
            echo_time,
            _end
        ) = self.spin_echo_times()

        echo_near = (
            abs(t - echo_time) < 0.10
        )

        for index, vector in enumerate(vectors):
            alpha = (
                0.28
                + 0.60
                * index
                / max(1, self.n_spins - 1)
            )

            color = (
                "goldenrod"
                if echo_near
                else "royalblue"
            )

            self._draw_vector_3d(
                ax,
                origin,
                vector,
                color=color,
                linewidth=1.7,
                alpha=alpha
            )

        # Net transverse magnetization.
        net_vector = np.mean(
            vectors,
            axis=0
        )

        self._draw_vector_3d(
            ax,
            origin,
            net_vector,
            color="black",
            linewidth=4.2,
            alpha=1.0,
            label="Net M"
        )

        # 90° pulse vector.
        if excitation_start <= t <= excitation_end:
            self._draw_rf_arrow(
                ax,
                origin=np.array([
                    -1.15,
                    -0.95,
                    -0.85
                ]),
                direction=np.array([
                    0.0,
                    0.85,
                    0.0
                ]),
                label="90° excitation",
                color="orangered"
            )

        # 180° pulse vector.
        if refocus_start <= t <= refocus_end:
            self._draw_rf_arrow(
                ax,
                origin=np.array([
                    -1.15,
                    -0.95,
                    -0.85
                ]),
                direction=np.array([
                    0.85,
                    0.0,
                    0.0
                ]),
                label="180° refocus",
                color="red"
            )

        if echo_near:
            ax.text2D(
                0.42,
                0.90,
                "SPIN ECHO",
                transform=ax.transAxes,
                color="goldenrod",
                fontsize=15,
                weight="bold"
            )

        ax.xaxis.pane.set_facecolor(
            (0.98, 0.98, 0.98, 1.0)
        )
        ax.yaxis.pane.set_facecolor(
            (0.98, 0.98, 0.98, 1.0)
        )
        ax.zaxis.pane.set_facecolor(
            (0.98, 0.98, 0.98, 1.0)
        )

    # ============================================================
    # Upper-right: spin-echo signal
    # ============================================================

    def draw_spin_echo_signal(
        self,
        current_t
    ):
        ax = self.ax_se_signal
        ax.clear()

        (
            excitation_start,
            excitation_end,
            _refocus_start,
            refocus_center,
            _refocus_end,
            echo_time,
            end
        ) = self.spin_echo_times()

        full_time = np.linspace(
            0,
            end,
            750
        )

        full_signal = np.array([
            self.spin_echo_signal(t)
            for t in full_time
        ])

        visible = (
            full_time
            <= current_t
        )

        ax.plot(
            full_time,
            full_signal,
            color="lightsteelblue",
            linewidth=1.5,
            alpha=0.45
        )

        ax.plot(
            full_time[visible],
            full_signal[visible],
            color="royalblue",
            linewidth=3
        )

        current_signal = self.spin_echo_signal(
            current_t
        )

        ax.scatter(
            [current_t],
            [current_signal],
            color="black",
            s=45,
            zorder=6
        )

        ax.axvspan(
            excitation_start,
            excitation_end,
            color="orangered",
            alpha=0.10
        )

        ax.axvline(
            refocus_center,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.80
        )

        ax.axvline(
            echo_time,
            color="goldenrod",
            linestyle="--",
            linewidth=1.8,
            alpha=0.90
        )

        ax.text(
            excitation_start,
            1.02,
            "90°",
            color="orangered",
            fontsize=10
        )

        ax.text(
            refocus_center,
            1.02,
            "180°",
            color="red",
            ha="center",
            fontsize=10
        )

        ax.text(
            echo_time,
            1.02,
            "Echo",
            color="goldenrod",
            ha="center",
            fontsize=10
        )

        ax.set_title(
            "Transverse coherence and spin echo",
            fontsize=13,
            weight="bold"
        )

        ax.set_xlim(0, end)
        ax.set_ylim(0, 1.08)

        ax.set_xlabel("Time")
        ax.set_ylabel("Net transverse signal")

        ax.grid(
            True,
            alpha=0.22
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ============================================================
    # Lower-left: 3D inversion-recovery animation
    # ============================================================

    def draw_inversion_recovery_3d(
        self,
        t
    ):
        ax = self.ax_ir_3d
        ax.clear()

        ax.set_title(
            "Inversion Recovery: 180° inverts longitudinal Mz",
            fontsize=12,
            weight="bold",
            pad=8
        )

        ax.set_xlim(-2.10, 2.10)
        ax.set_ylim(-1.25, 1.25)
        ax.set_zlim(-1.25, 1.25)

        ax.set_box_aspect(
            (1.8, 1.0, 1.0)
        )

        ax.set_xlabel("Mx")
        ax.set_ylabel("My")
        ax.set_zlabel("Mz / B0")

        ax.view_init(
            elev=24,
            azim=self.ir_azim
        )

        short_center = np.array([
            -1.02,
            0.0,
            0.0
        ])

        long_center = np.array([
            1.02,
            0.0,
            0.0
        ])

        sphere_radius = 0.72

        target = self.target_var.get()

        short_alpha = (
            0.30
            if target == "Short T1"
            else 0.12
        )

        long_alpha = (
            0.30
            if target == "Long T1"
            else 0.12
        )

        self._draw_bloch_sphere(
            ax,
            short_center,
            radius=sphere_radius,
            color="firebrick",
            alpha=short_alpha
        )

        self._draw_bloch_sphere(
            ax,
            long_center,
            radius=sphere_radius,
            color="royalblue",
            alpha=long_alpha
        )

        ax.text(
            short_center[0],
            short_center[1] - 0.88,
            short_center[2] - 0.85,
            "Short T1",
            color="firebrick",
            fontsize=10,
            ha="center",
            weight=(
                "bold"
                if target == "Short T1"
                else "normal"
            )
        )

        ax.text(
            long_center[0],
            long_center[1] - 0.88,
            long_center[2] - 0.85,
            "Long T1",
            color="royalblue",
            fontsize=10,
            ha="center",
            weight=(
                "bold"
                if target == "Long T1"
                else "normal"
            )
        )

        short_vector = (
            sphere_radius
            * self.ir_vector_3d(
                t,
                float(self.short_t1_var.get()),
                phase_offset=0.0
            )
        )

        long_vector = (
            sphere_radius
            * self.ir_vector_3d(
                t,
                float(self.long_t1_var.get()),
                phase_offset=0.0
            )
        )

        self._draw_vector_3d(
            ax,
            short_center,
            short_vector,
            color="firebrick",
            linewidth=4.0,
            alpha=1.0
        )

        self._draw_vector_3d(
            ax,
            long_center,
            long_vector,
            color="royalblue",
            linewidth=4.0,
            alpha=1.0
        )

        self._draw_b0_arrow(
            ax,
            np.array([
                -1.90,
                -1.00,
                -1.05
            ]),
            length=0.70
        )

        (
            inversion_start,
            recovery_start,
            excitation_start,
            excitation_end,
            _end
        ) = self.inversion_times()

        # 180° inversion pulse.
        if inversion_start <= t <= recovery_start:
            self._draw_rf_arrow(
                ax,
                origin=np.array([
                    -0.45,
                    -1.05,
                    -0.90
                ]),
                direction=np.array([
                    0.90,
                    0.0,
                    0.0
                ]),
                label="180° inversion",
                color="orangered"
            )

        # 90° excitation pulse at TI.
        if excitation_start <= t <= excitation_end:
            self._draw_rf_arrow(
                ax,
                origin=np.array([
                    -0.45,
                    -1.05,
                    -0.90
                ]),
                direction=np.array([
                    0.90,
                    0.0,
                    0.0
                ]),
                label="90° at TI",
                color="purple"
            )

        ti = float(
            self.ir_ti_var.get()
        )

        short_mz = self.inversion_recovery(
            ti,
            float(self.short_t1_var.get())
        )

        long_mz = self.inversion_recovery(
            ti,
            float(self.long_t1_var.get())
        )

        ax.text2D(
            0.02,
            0.93,
            (
                f"Mz at TI:  "
                f"Short T1 = {short_mz:+.2f}   "
                f"Long T1 = {long_mz:+.2f}"
            ),
            transform=ax.transAxes,
            fontsize=9
        )

        ax.xaxis.pane.set_facecolor(
            (0.98, 0.98, 0.98, 1.0)
        )
        ax.yaxis.pane.set_facecolor(
            (0.98, 0.98, 0.98, 1.0)
        )
        ax.zaxis.pane.set_facecolor(
            (0.98, 0.98, 0.98, 1.0)
        )

    # ============================================================
    # Lower-right: inversion recovery curves
    # ============================================================

    def draw_inversion_curve(
        self,
        current_t
    ):
        ax = self.ax_ir_curve
        ax.clear()

        (
            _inversion_start,
            recovery_start,
            excitation_start,
            _excitation_end,
            _end
        ) = self.inversion_times()

        short_t1 = float(
            self.short_t1_var.get()
        )

        long_t1 = float(
            self.long_t1_var.get()
        )

        ti = float(
            self.ir_ti_var.get()
        )

        curve_end = max(
            3.2,
            ti + 0.70,
            long_t1 * 1.30
        )

        recovery_time = np.linspace(
            0,
            curve_end,
            750
        )

        short_curve = (
            1.0
            - 2.0
            * np.exp(
                -recovery_time
                / short_t1
            )
        )

        long_curve = (
            1.0
            - 2.0
            * np.exp(
                -recovery_time
                / long_t1
            )
        )

        elapsed_recovery = np.clip(
            current_t
            - recovery_start,
            0.0,
            curve_end
        )

        visible = (
            recovery_time
            <= elapsed_recovery
        )

        target = self.target_var.get()

        short_alpha = (
            1.0
            if target == "Short T1"
            else 0.38
        )

        long_alpha = (
            1.0
            if target == "Long T1"
            else 0.38
        )

        short_width = (
            3.2
            if target == "Short T1"
            else 2.0
        )

        long_width = (
            3.2
            if target == "Long T1"
            else 2.0
        )

        # Faint full trajectories.
        ax.plot(
            recovery_time,
            short_curve,
            color="firebrick",
            alpha=0.14,
            linewidth=1.4
        )

        ax.plot(
            recovery_time,
            long_curve,
            color="royalblue",
            alpha=0.14,
            linewidth=1.4
        )

        # Animated trajectories.
        ax.plot(
            recovery_time[visible],
            short_curve[visible],
            color="firebrick",
            alpha=short_alpha,
            linewidth=short_width,
            label="Short T1"
        )

        ax.plot(
            recovery_time[visible],
            long_curve[visible],
            color="royalblue",
            alpha=long_alpha,
            linewidth=long_width,
            label="Long T1"
        )

        ax.axhline(
            0,
            color="black",
            linewidth=1
        )

        ax.axhline(
            1,
            color="lightgray",
            linestyle=":"
        )

        ax.axvline(
            ti,
            color="purple",
            linestyle="--",
            linewidth=2.2
        )

        ax.text(
            ti,
            0.95,
            "TI / 90°",
            color="purple",
            fontsize=10,
            rotation=90,
            va="top",
            ha="right"
        )

        short_at_ti = self.inversion_recovery(
            ti,
            short_t1
        )

        long_at_ti = self.inversion_recovery(
            ti,
            long_t1
        )

        ax.scatter(
            [ti],
            [short_at_ti],
            color="firebrick",
            s=60,
            zorder=6
        )

        ax.scatter(
            [ti],
            [long_at_ti],
            color="royalblue",
            s=60,
            zorder=6
        )

        if current_t >= excitation_start:
            ax.axvspan(
                ti,
                curve_end,
                color="purple",
                alpha=0.035
            )

        ax.set_title(
            "Longitudinal recovery after inversion",
            fontsize=13,
            weight="bold"
        )

        ax.set_xlim(
            0,
            curve_end
        )

        ax.set_ylim(
            -1.08,
            1.08
        )

        ax.set_xlabel(
            "Time after 180° inversion"
        )

        ax.set_ylabel(
            "Longitudinal magnetization Mz"
        )

        ax.grid(
            True,
            alpha=0.22
        )

        ax.legend(
            loc="lower right",
            fontsize=10
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ============================================================
    # Status
    # ============================================================

    def update_status(
        self,
        se_time,
        ir_time
    ):
        target = self.target_var.get()

        if target == "Short T1":
            target_t1 = float(
                self.short_t1_var.get()
            )
        else:
            target_t1 = float(
                self.long_t1_var.get()
            )

        ti = float(
            self.ir_ti_var.get()
        )

        target_mz = self.inversion_recovery(
            ti,
            target_t1
        )

        target_signal = abs(
            target_mz
        )

        if target_signal < 0.05:
            null_status = "Target tissue is near null."
        elif target_mz < 0:
            null_status = "Target is still below Mz = 0."
        else:
            null_status = "Target has passed Mz = 0."

        self.status_label.config(
            text=(
                "Spin Echo\n"
                f"{self.spin_echo_stage(se_time)}\n\n"
                "Inversion Recovery\n"
                f"{self.inversion_stage(ir_time)}\n\n"
                f"Target: {target}\n"
                f"TI: {ti:.2f} s\n"
                f"Mz at TI: {target_mz:+.3f}\n"
                f"{null_status}"
            )
        )


def main():
    root = tk.Tk()
    Compare180PulseRoles(root)
    root.mainloop()


if __name__ == "__main__":
    main()