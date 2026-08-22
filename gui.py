#!/usr/bin/env python3
"""Desktop GUI for AutoMatExtract.

Thin wrapper around main.run_extraction() (main.py) — the GUI and the CLI (`python main.py`)
always run the exact same pipeline code, so behavior never drifts between the two.

Usage:
    python gui.py
    python main.py --gui
"""

import logging
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import List, Optional

from main import _CREDIT_TEXT, _PROJECT_DESCRIPTION, run_extraction
from models import PaperData

_DONE_SENTINEL = object()

# (category attribute on PaperData, property name) highlighted for every processed paper —
# activation energy plus the other properties most commonly asked about across categories.
_HIGHLIGHT_PROPERTIES = [
    ("electrochemical", "Activation_Energy"),
    ("electrochemical", "Ionic_Conductivity"),
    ("electrochemical", "Capacity"),
    ("electrochemical", "Voltage"),
    ("electronic", "Band_Gap"),
    ("thermodynamic", "Formation_Energy"),
    ("mechanical", "Bulk_Modulus"),
    ("structural", "Density"),
]


def _summarize_paper(paper: PaperData) -> str:
    """One readable block per processed paper: materials found plus every headline property
    that was actually extracted (activation energy, conductivity, band gap, ...)."""
    lines: List[str] = []

    title = paper.metadata.get("title")
    if title and not title.is_empty():
        lines.append(f"    Title: {title.normalized_value}")

    top_materials = [m.normalized_formula for m in paper.materials[:5]]
    lines.append(f"    Materials ({len(paper.materials)}): "
                 f"{', '.join(top_materials) if top_materials else 'none found'}")

    found_any_property = False
    for cat_field, prop in _HIGHLIGHT_PROPERTIES:
        candidates = (getattr(paper, cat_field, {}) or {}).get(prop, [])
        if candidates:
            found_any_property = True
            best = candidates[0]
            lines.append(f"    {prop}: {best.normalized_value} (confidence {best.confidence:.2f})")
    if not found_any_property:
        lines.append("    No headline properties (activation energy, conductivity, ...) matched.")

    lines.append(f"    Relations: {len(paper.relations)} | Tables: {len(paper.tables)} | "
                 f"Figures: {len(paper.figures)} | Equations: {len(paper.equations)}")
    return "\n".join(lines)


def _final_summary(papers: List[PaperData]) -> str:
    if not papers:
        return "No papers were successfully processed."

    total_materials = sum(len(p.materials) for p in papers)
    total_relations = sum(len(p.relations) for p in papers)
    with_activation_energy = sum(
        1 for p in papers if p.electrochemical.get("Activation_Energy"))

    lines = [
        "",
        "==================== Summary ====================",
        f"Papers processed:              {len(papers)}",
        f"Materials found (total):       {total_materials}",
        f"Property relations found:      {total_relations}",
        f"Papers with Activation Energy: {with_activation_energy}",
        "===================================================",
    ]
    return "\n".join(lines)


class QueueLogHandler(logging.Handler):
    """Routes stdlib logging records into a thread-safe queue the GUI drains on the main thread
    (Tkinter widgets may only be touched from the main thread, but pipeline.py logs from
    worker threads)."""

    def __init__(self, log_queue: "queue.Queue"):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self.log_queue.put(self.format(record))


class AutoMatExtractGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AutoMatExtract")
        self.root.geometry("720x580")
        self.root.minsize(560, 440)

        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.log_queue: "queue.Queue" = queue.Queue()
        self._processed_papers: List[PaperData] = []

        self._build_widgets()
        self._setup_logging()
        self.root.protocol("WM_DELETE_WINDOW", self._on_end_program)
        self.root.after(100, self._drain_log_queue)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        pad = {"padx": 8, "pady": 6}

        header = ttk.Frame(self.root, padding=(8, 10, 8, 4))
        header.pack(fill="x")

        ttk.Label(header, text="AutoMatExtract", font=("TkDefaultFont", 15, "bold")).pack(anchor="w")
        ttk.Label(header, text=_PROJECT_DESCRIPTION, wraplength=680, justify="left",
                  foreground="#555555").pack(anchor="w", pady=(2, 4))
        ttk.Label(header, text=_CREDIT_TEXT, font=("TkDefaultFont", 9),
                  foreground="gray").pack(anchor="w")
        ttk.Separator(header, orient="horizontal").pack(fill="x", pady=(8, 0))

        form = ttk.Frame(self.root)
        form.pack(fill="x", **pad)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Input folder (PDFs):").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar(value="papers")
        ttk.Entry(form, textvariable=self.input_var).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(form, text="Browse…", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(form, text="Output folder:").grid(row=1, column=0, sticky="w")
        self.output_var = tk.StringVar(value="output")
        ttk.Entry(form, textvariable=self.output_var).grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(form, text="Browse…", command=self._browse_output).grid(row=1, column=2)

        ttk.Label(form, text="PDFs processed at once:").grid(row=2, column=0, sticky="w")
        self.workers_var = tk.IntVar(value=4)
        ttk.Spinbox(form, from_=1, to=32, textvariable=self.workers_var, width=6).grid(
            row=2, column=1, sticky="w", padx=4)

        buttons = ttk.Frame(self.root)
        buttons.pack(fill="x", **pad)

        self.start_btn = ttk.Button(buttons, text="Start Processing", command=self._on_start)
        self.start_btn.pack(side="left", padx=4)

        self.stop_btn = ttk.Button(buttons, text="Stop", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        self.end_btn = ttk.Button(buttons, text="End Program", command=self._on_end_program)
        self.end_btn.pack(side="right", padx=4)

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(self.root, textvariable=self.status_var).pack(fill="x", padx=8)

        self.log_box = scrolledtext.ScrolledText(self.root, state="disabled", height=22)
        self.log_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _setup_logging(self) -> None:
        handler = QueueLogHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

    # ------------------------------------------------------------------
    # Folder pickers
    # ------------------------------------------------------------------
    def _browse_input(self) -> None:
        path = filedialog.askdirectory(title="Select folder containing PDFs")
        if path:
            self.input_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

    # ------------------------------------------------------------------
    # Start / Stop / End
    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return  # already running — Start is disabled while running, but guard anyway

        input_dir = self.input_var.get().strip()
        output_dir = self.output_var.get().strip()

        if not os.path.isdir(input_dir):
            messagebox.showerror("AutoMatExtract", f"Input directory not found:\n{input_dir}")
            return
        if not output_dir:
            messagebox.showerror("AutoMatExtract", "Please choose an output folder.")
            return

        try:
            max_workers = int(self.workers_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("AutoMatExtract", "\"PDFs processed at once\" must be a whole number.")
            return
        if max_workers < 1:
            messagebox.showerror("AutoMatExtract", "\"PDFs processed at once\" must be at least 1.")
            return

        self.stop_event.clear()
        self._processed_papers = []
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Processing…")

        self.worker_thread = threading.Thread(
            target=self._run_pipeline,
            args=(input_dir, output_dir, max_workers),
            daemon=True,
        )
        self.worker_thread.start()

    def _on_progress(self, filename: str, paper: Optional[PaperData]) -> None:
        """Called by pipeline.execute() on its own thread as soon as each PDF finishes."""
        if paper is None:
            self.log_queue.put(f"[FAILED] {filename}")
            return
        self._processed_papers.append(paper)
        self.log_queue.put(f"[DONE] {filename}\n{_summarize_paper(paper)}")

    def _run_pipeline(self, input_dir: str, output_dir: str, max_workers: int) -> None:
        try:
            run_extraction(input_dir, output_dir, max_workers=max_workers,
                            stop_event=self.stop_event, progress_callback=self._on_progress)
            self.log_queue.put(_final_summary(self._processed_papers))
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
        finally:
            self.log_queue.put(_DONE_SENTINEL)

    def _on_stop(self) -> None:
        # Lets any PDFs already in flight finish (so their work isn't wasted) but cancels
        # everything still queued — see the stop_event check in pipeline.py's execute().
        self.stop_event.set()
        self.status_var.set("Stopping — finishing in-flight PDFs…")
        self.stop_btn.config(state="disabled")

    def _on_end_program(self) -> None:
        self.stop_event.set()
        self.root.after(150, self.root.destroy)

    # ------------------------------------------------------------------
    # Log panel
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        self.log_box.config(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                message = self.log_queue.get_nowait()
                if message is _DONE_SENTINEL:
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.status_var.set("Idle.")
                else:
                    self._log(message)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log_queue)


def main() -> None:
    root = tk.Tk()
    AutoMatExtractGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
