#!/usr/bin/env python3
"""CLI entry point for AutoMatExtract.

Usage:
	author - Younes Moukhlij
	python main.py
	python main.py --input papers --output output --workers 4
	python main.py --input /path/to/pdfs --output /path/to/database -v
	python main.py --gui
"""

import argparse
import logging
import os
import sys
import threading
from typing import Callable, Optional

from models import PaperData
from pipeline import NaLiXPipeline

_CREDIT_TEXT = ("AutoMatExtract — Created by Younes Moukhlij "
				"(Software Engineering student at 1337 Coding School, "
				"PhD researcher at FPK, USMS)")

# Shown alongside _CREDIT_TEXT at startup, in both the CLI banner and the GUI's header —
# kept here (rather than duplicated in gui.py) so the two never drift apart.
_PROJECT_DESCRIPTION = (
	"Turns a folder of battery/materials-science PDFs into a structured, queryable database: "
	"parses text, tables, figures and equations (with OCR fallback for scanned pages), extracts "
	"materials and their properties, and exports linked CSV, Excel and JSON."
)


def _print_banner() -> None:
	"""Prints a small decorated header to the console: project name, one-line description, and
	author credit (see _CREDIT_TEXT for the plain-text version used in --help, where box-drawing
	characters would look out of place)."""
	import textwrap

	title = "AutoMatExtract"
	body_lines = (
		textwrap.wrap(_PROJECT_DESCRIPTION, width=58)
		+ [""]
		+ [
			"Younes Moukhlij",
			"Software Engineering student · 1337 Coding School",
			"PhD Researcher · FPK, USMS",
		]
	)
	width = max(len(title), *(len(l) for l in body_lines)) + 4

	print("╭" + "─" * width + "╮")
	print("│ " + title.center(width - 2) + " │")
	print("├" + "─" * width + "┤")
	for l in body_lines:
		print("│ " + l.center(width - 2) + " │")
	print("╰" + "─" * width + "╯")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Extract structured battery-materials data (materials, properties, DFT/AIMD/ML "
					"parameters, relations, tables, figures, equations, workflow) from PDFs.",
		epilog=_CREDIT_TEXT,
	)
	parser.add_argument("--input", "-i", default="papers",
						help="Directory containing input PDFs (default: papers)")
	parser.add_argument("--output", "-o", default="output",
						help="Directory to write the extracted database into (default: output)")
	parser.add_argument("--workers", "-w", type=int, default=4,
						help="Number of PDFs to process in parallel (default: 4)")
	parser.add_argument("--verbose", "-v", action="store_true",
						help="Enable debug-level logging")
	parser.add_argument("--gui", action="store_true",
						help="Launch the desktop GUI instead of running from the command line")
	return parser.parse_args()


def run_extraction(input_dir: str, output_dir: str, max_workers: int = 4,
					stop_event: Optional[threading.Event] = None,
					progress_callback: Optional[Callable[[str, Optional[PaperData]], None]] = None) -> int:
	"""Shared pipeline entry point used by both the CLI (`main()`) and the GUI (`gui.py`), so
	the two never drift into two different ways of running the same extraction.

	`stop_event`/`progress_callback` are optional and unused by the plain CLI path — they exist
	so the GUI can request an early stop and get live per-paper feedback.
	"""
	if not os.path.isdir(input_dir):
		logging.error(f"Input directory not found: {input_dir}")
		return 1

	pipeline = NaLiXPipeline(input_dir=input_dir, output_dir=output_dir)
	pipeline.execute(max_workers=max_workers, stop_event=stop_event, progress_callback=progress_callback)
	return 0


def main() -> int:
	args = parse_args()

	logging.basicConfig(
		level=logging.DEBUG if args.verbose else logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
		datefmt="%H:%M:%S",
	)

	_print_banner()

	if args.gui:
		from gui import main as gui_main
		gui_main()
		return 0

	print(f"--> Processing papers from {args.input} to {args.output}")
	print(f"Using {args.workers} parallel workers")

	return run_extraction(args.input, args.output, max_workers=args.workers)


if __name__ == "__main__":
	sys.exit(main())
