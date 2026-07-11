"""
MRI Learning Studio - Extensible Lesson Launcher

Recommended project structure
-----------------------------
MRI-Learning-Studio/
├─ run.py
├─ lessons/
│  ├─ 01_MRI_Fundamentals_Coil_and_Magnetization.py
│  ├─ 02_RF_Pulse_and_Flip_Angle.py
│  └─ ...
├─ assets/
└─ README.md

How it works
------------
1. Automatically scans the lessons/ folder for Python files.
2. Uses the filename to generate a readable lesson title.
3. Starts each lesson in a separate Python process.
4. Supports search, refresh, sorting, and process termination.
5. Optionally reads lesson metadata from the top-level constants:
       LESSON_TITLE = "MRI Coils and Magnetization"
       LESSON_DESCRIPTION = "Learn the roles of B0, RF, gradient, and receive coils."
       LESSON_CATEGORY = "MRI Fundamentals"
       LESSON_ORDER = 1

Notes
-----
Each lesson script should use:
    if __name__ == "__main__":
        main()

This prevents the lesson from starting accidentally when inspected by run.py.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional


APP_TITLE = "MRI Learning Studio"
BASE_DIR = Path(__file__).resolve().parent
LESSONS_DIR = BASE_DIR / "lessons"

WINDOW_SIZE = "1100x700"
MIN_WINDOW_SIZE = (850, 560)

IGNORED_FILENAMES = {
    "__init__.py",
    "run.py",
    "gui.py",
}

METADATA_KEYS = {
    "LESSON_TITLE",
    "LESSON_DESCRIPTION",
    "LESSON_CATEGORY",
    "LESSON_ORDER",
}


@dataclass(frozen=True)
class LessonInfo:
    path: Path
    filename: str
    title: str
    description: str
    category: str
    order: int

    @property
    def search_text(self) -> str:
        return " ".join(
            [
                self.filename,
                self.title,
                self.description,
                self.category,
            ]
        ).lower()


def natural_sort_key(text: str) -> list[object]:
    """Sort strings naturally: 2 before 10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def title_from_filename(path: Path) -> str:
    """
    Convert:
        01_MRI_Fundamentals_Coil_and_Magnetization.py
    into:
        01 · MRI Fundamentals Coil and Magnetization
    """
    stem = path.stem
    match = re.match(r"^(\d+)[_\-\s]*(.*)$", stem)

    if match:
        number = match.group(1)
        name = match.group(2)
        readable = re.sub(r"[_\-]+", " ", name).strip()
        readable = re.sub(r"\s+", " ", readable)
        return f"{number} · {readable}"

    readable = re.sub(r"[_\-]+", " ", stem).strip()
    return re.sub(r"\s+", " ", readable)


def order_from_filename(path: Path, default: int = 9999) -> int:
    match = re.match(r"^(\d+)", path.stem)
    return int(match.group(1)) if match else default


def read_lesson_metadata(path: Path) -> dict[str, object]:
    """
    Safely read simple top-level metadata constants using AST.
    The lesson script itself is not executed.
    """
    metadata: dict[str, object] = {}

    try:
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError):
        return metadata

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        else:
            targets = [node.target]
            value_node = node.value

        if value_node is None:
            continue

        for target in targets:
            if not isinstance(target, ast.Name):
                continue

            key = target.id
            if key not in METADATA_KEYS:
                continue

            try:
                value = ast.literal_eval(value_node)
            except (ValueError, TypeError):
                continue

            metadata[key] = value

    return metadata


def discover_lessons(directory: Path) -> list[LessonInfo]:
    directory.mkdir(parents=True, exist_ok=True)

    lessons: list[LessonInfo] = []

    for path in directory.glob("*.py"):
        if path.name.startswith("_") or path.name in IGNORED_FILENAMES:
            continue

        metadata = read_lesson_metadata(path)

        title = str(
            metadata.get("LESSON_TITLE")
            or title_from_filename(path)
        )

        description = str(
            metadata.get("LESSON_DESCRIPTION")
            or "No lesson description has been added yet."
        )

        category = str(
            metadata.get("LESSON_CATEGORY")
            or "Uncategorized"
        )

        raw_order = metadata.get("LESSON_ORDER")
        try:
            order = int(raw_order) if raw_order is not None else order_from_filename(path)
        except (TypeError, ValueError):
            order = order_from_filename(path)

        lessons.append(
            LessonInfo(
                path=path,
                filename=path.name,
                title=title,
                description=description,
                category=category,
                order=order,
            )
        )

    lessons.sort(
        key=lambda lesson: (
            lesson.order,
            natural_sort_key(lesson.filename),
        )
    )
    return lessons


class LessonLauncherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(*MIN_WINDOW_SIZE)

        self.lessons: list[LessonInfo] = []
        self.filtered_lessons: list[LessonInfo] = []
        self.running_processes: dict[Path, subprocess.Popen] = {}

        self.search_var = tk.StringVar()
        self.category_var = tk.StringVar(value="All categories")
        self.status_var = tk.StringVar(value="Ready")

        self._configure_style()
        self._build_ui()
        self.refresh_lessons()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(1000, self.poll_processes)

    def _configure_style(self) -> None:
        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Title.TLabel", font=("Arial", 22, "bold"))
        style.configure("Subtitle.TLabel", font=("Arial", 11))
        style.configure("LessonTitle.TLabel", font=("Arial", 15, "bold"))
        style.configure("Category.TLabel", font=("Arial", 10, "bold"))
        style.configure("Status.TLabel", font=("Arial", 10))
        style.configure("Accent.TButton", font=("Arial", 11, "bold"))

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        self._build_header(main)
        self._build_toolbar(main)
        self._build_content(main)
        self._build_status_bar(main)

    def _build_header(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, pady=(0, 14))

        ttk.Label(
            header,
            text=APP_TITLE,
            style="Title.TLabel",
        ).pack(anchor=tk.W)

        ttk.Label(
            header,
            text="Select a lesson to launch an interactive MRI learning module.",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(3, 0))

    def _build_toolbar(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(toolbar, text="Search:").pack(side=tk.LEFT)

        search_entry = ttk.Entry(
            toolbar,
            textvariable=self.search_var,
            width=32,
        )
        search_entry.pack(side=tk.LEFT, padx=(6, 14))
        search_entry.bind("<KeyRelease>", lambda _event: self.apply_filter())

        ttk.Label(toolbar, text="Category:").pack(side=tk.LEFT)

        self.category_combo = ttk.Combobox(
            toolbar,
            textvariable=self.category_var,
            state="readonly",
            width=22,
        )
        self.category_combo.pack(side=tk.LEFT, padx=(6, 14))
        self.category_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.apply_filter(),
        )

        ttk.Button(
            toolbar,
            text="Refresh lessons",
            command=self.refresh_lessons,
        ).pack(side=tk.RIGHT)

        ttk.Button(
            toolbar,
            text="Open lessons folder",
            command=self.open_lessons_folder,
        ).pack(side=tk.RIGHT, padx=(0, 8))

    def _build_content(self, parent: ttk.Frame) -> None:
        content = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(content, padding=(0, 0, 10, 0))
        right_panel = ttk.Frame(content, padding=(10, 0, 0, 0))

        content.add(left_panel, weight=3)
        content.add(right_panel, weight=2)

        self._build_lesson_list(left_panel)
        self._build_details_panel(right_panel)

    def _build_lesson_list(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        self.lesson_tree = ttk.Treeview(
            frame,
            columns=("category", "status"),
            show="tree headings",
            selectmode="browse",
        )

        self.lesson_tree.heading("#0", text="Lesson")
        self.lesson_tree.heading("category", text="Category")
        self.lesson_tree.heading("status", text="Status")

        self.lesson_tree.column("#0", width=390, minwidth=240)
        self.lesson_tree.column("category", width=150, minwidth=100)
        self.lesson_tree.column("status", width=90, minwidth=75, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(
            frame,
            orient=tk.VERTICAL,
            command=self.lesson_tree.yview,
        )
        self.lesson_tree.configure(yscrollcommand=scrollbar.set)

        self.lesson_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.lesson_tree.bind("<<TreeviewSelect>>", self.on_lesson_selected)
        self.lesson_tree.bind("<Double-1>", lambda _event: self.launch_selected())
        self.lesson_tree.bind("<Return>", lambda _event: self.launch_selected())

    def _build_details_panel(self, parent: ttk.Frame) -> None:
        detail_box = ttk.LabelFrame(parent, text="Lesson details", padding=16)
        detail_box.pack(fill=tk.BOTH, expand=True)

        self.detail_title = ttk.Label(
            detail_box,
            text="Select a lesson",
            style="LessonTitle.TLabel",
            wraplength=360,
            justify=tk.LEFT,
        )
        self.detail_title.pack(anchor=tk.W)

        self.detail_category = ttk.Label(
            detail_box,
            text="",
            style="Category.TLabel",
        )
        self.detail_category.pack(anchor=tk.W, pady=(8, 12))

        self.detail_description = ttk.Label(
            detail_box,
            text="Choose a lesson from the list to view its description.",
            wraplength=360,
            justify=tk.LEFT,
        )
        self.detail_description.pack(anchor=tk.W, fill=tk.X)

        ttk.Separator(detail_box).pack(fill=tk.X, pady=18)

        self.detail_filename = ttk.Label(
            detail_box,
            text="",
            wraplength=360,
            justify=tk.LEFT,
        )
        self.detail_filename.pack(anchor=tk.W)

        self.detail_path = ttk.Label(
            detail_box,
            text="",
            wraplength=360,
            justify=tk.LEFT,
        )
        self.detail_path.pack(anchor=tk.W, pady=(5, 0))

        button_frame = ttk.Frame(detail_box)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(18, 0))

        self.launch_button = ttk.Button(
            button_frame,
            text="Launch lesson",
            style="Accent.TButton",
            command=self.launch_selected,
            state=tk.DISABLED,
        )
        self.launch_button.pack(fill=tk.X, pady=(0, 8))

        self.stop_button = ttk.Button(
            button_frame,
            text="Stop selected lesson",
            command=self.stop_selected,
            state=tk.DISABLED,
        )
        self.stop_button.pack(fill=tk.X)

    def _build_status_bar(self, parent: ttk.Frame) -> None:
        ttk.Separator(parent).pack(fill=tk.X, pady=(12, 8))

        footer = ttk.Frame(parent)
        footer.pack(fill=tk.X)

        ttk.Label(
            footer,
            textvariable=self.status_var,
            style="Status.TLabel",
        ).pack(side=tk.LEFT)

        self.lesson_count_label = ttk.Label(
            footer,
            text="0 lessons",
            style="Status.TLabel",
        )
        self.lesson_count_label.pack(side=tk.RIGHT)

    def refresh_lessons(self) -> None:
        self.lessons = discover_lessons(LESSONS_DIR)

        categories = sorted(
            {lesson.category for lesson in self.lessons},
            key=str.lower,
        )
        category_values = ["All categories", *categories]
        self.category_combo["values"] = category_values

        if self.category_var.get() not in category_values:
            self.category_var.set("All categories")

        self.apply_filter()
        self.status_var.set(f"Scanned: {LESSONS_DIR}")

    def apply_filter(self) -> None:
        query = self.search_var.get().strip().lower()
        selected_category = self.category_var.get()

        self.filtered_lessons = [
            lesson
            for lesson in self.lessons
            if (not query or query in lesson.search_text)
            and (
                selected_category == "All categories"
                or lesson.category == selected_category
            )
        ]

        self.populate_tree()

    def populate_tree(self) -> None:
        selected_path = self.get_selected_lesson_path()

        for item in self.lesson_tree.get_children():
            self.lesson_tree.delete(item)

        item_to_select: Optional[str] = None

        for index, lesson in enumerate(self.filtered_lessons):
            status = self.get_process_status(lesson.path)
            item_id = str(index)

            self.lesson_tree.insert(
                "",
                tk.END,
                iid=item_id,
                text=lesson.title,
                values=(lesson.category, status),
            )

            if selected_path == lesson.path:
                item_to_select = item_id

        if item_to_select is not None:
            self.lesson_tree.selection_set(item_to_select)
            self.lesson_tree.focus(item_to_select)
        elif self.filtered_lessons:
            self.lesson_tree.selection_set("0")
            self.lesson_tree.focus("0")
        else:
            self.clear_details()

        self.lesson_count_label.config(
            text=f"{len(self.filtered_lessons)} / {len(self.lessons)} lessons"
        )

        self.on_lesson_selected()

    def get_selected_lesson(self) -> Optional[LessonInfo]:
        selection = self.lesson_tree.selection()
        if not selection:
            return None

        try:
            index = int(selection[0])
        except (TypeError, ValueError):
            return None

        if not 0 <= index < len(self.filtered_lessons):
            return None

        return self.filtered_lessons[index]

    def get_selected_lesson_path(self) -> Optional[Path]:
        lesson = self.get_selected_lesson()
        return lesson.path if lesson else None

    def on_lesson_selected(self, _event=None) -> None:
        lesson = self.get_selected_lesson()

        if lesson is None:
            self.clear_details()
            return

        self.detail_title.config(text=lesson.title)
        self.detail_category.config(text=lesson.category)
        self.detail_description.config(text=lesson.description)
        self.detail_filename.config(text=f"File: {lesson.filename}")
        self.detail_path.config(text=f"Path: {lesson.path}")

        self.launch_button.config(state=tk.NORMAL)

        is_running = self.is_process_running(lesson.path)
        self.stop_button.config(
            state=tk.NORMAL if is_running else tk.DISABLED
        )

    def clear_details(self) -> None:
        self.detail_title.config(text="No lesson selected")
        self.detail_category.config(text="")
        self.detail_description.config(
            text="No lessons match the current filter."
        )
        self.detail_filename.config(text="")
        self.detail_path.config(text="")
        self.launch_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)

    def launch_selected(self) -> None:
        lesson = self.get_selected_lesson()
        if lesson is None:
            return

        existing = self.running_processes.get(lesson.path)
        if existing is not None and existing.poll() is None:
            bring_forward = messagebox.askyesno(
                "Lesson already running",
                (
                    f'"{lesson.title}" is already running.\n\n'
                    "Start another instance?"
                ),
            )
            if not bring_forward:
                return

        try:
            process = subprocess.Popen(
                [sys.executable, str(lesson.path)],
                cwd=str(lesson.path.parent),
            )
        except OSError as exc:
            messagebox.showerror(
                "Unable to launch lesson",
                f"Could not start:\n{lesson.path}\n\n{exc}",
            )
            return

        self.running_processes[lesson.path] = process
        self.status_var.set(f"Started: {lesson.title}")
        self.populate_tree()

    def stop_selected(self) -> None:
        lesson = self.get_selected_lesson()
        if lesson is None:
            return

        process = self.running_processes.get(lesson.path)
        if process is None or process.poll() is not None:
            self.running_processes.pop(lesson.path, None)
            self.populate_tree()
            return

        should_stop = messagebox.askyesno(
            "Stop lesson",
            f'Stop "{lesson.title}"?',
        )
        if not should_stop:
            return

        self.terminate_process(process)
        self.running_processes.pop(lesson.path, None)
        self.status_var.set(f"Stopped: {lesson.title}")
        self.populate_tree()

    def terminate_process(self, process: subprocess.Popen) -> None:
        try:
            process.terminate()
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        except OSError:
            pass

    def is_process_running(self, path: Path) -> bool:
        process = self.running_processes.get(path)
        return process is not None and process.poll() is None

    def get_process_status(self, path: Path) -> str:
        return "Running" if self.is_process_running(path) else "Ready"

    def poll_processes(self) -> None:
        changed = False

        for path, process in list(self.running_processes.items()):
            if process.poll() is not None:
                self.running_processes.pop(path, None)
                changed = True

        if changed:
            self.populate_tree()

        self.root.after(1000, self.poll_processes)

    def open_lessons_folder(self) -> None:
        LESSONS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            if sys.platform.startswith("win"):
                os.startfile(LESSONS_DIR)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(LESSONS_DIR)])
            else:
                subprocess.Popen(["xdg-open", str(LESSONS_DIR)])
        except OSError as exc:
            messagebox.showerror(
                "Unable to open folder",
                f"Could not open:\n{LESSONS_DIR}\n\n{exc}",
            )

    def on_close(self) -> None:
        active_processes = [
            process
            for process in self.running_processes.values()
            if process.poll() is None
        ]

        if active_processes:
            close_all = messagebox.askyesnocancel(
                "Close application",
                (
                    f"{len(active_processes)} lesson window(s) are still running.\n\n"
                    "Yes: close the launcher and all lessons\n"
                    "No: close only the launcher\n"
                    "Cancel: return to the launcher"
                ),
            )

            if close_all is None:
                return

            if close_all:
                for process in active_processes:
                    self.terminate_process(process)

        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    LessonLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
