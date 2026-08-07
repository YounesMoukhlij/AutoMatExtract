#!/usr/bin/env python3
"""CLI entry point for AutoMatExtract.

Usage:
    python main.py
    python main.py --input papers --output output --workers 4
    python main.py --input /path/to/pdfs --output /path/to/database -v
"""

import argparse
import logging
import os
import sys

from pipeline import NaLiXPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured battery-materials data (materials, properties, DFT/AIMD/ML "
                    "parameters, relations, tables, figures, equations, workflow) from PDFs.",
    )
    parser.add_argument("--input", "-i", default="papers",
                        help="Directory containing input PDFs (default: papers)")
    parser.add_argument("--output", "-o", default="output",
                        help="Directory to write the extracted database into (default: output)")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of PDFs to process in parallel (default: 4)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug-level logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.path.isdir(args.input):
        logging.error(f"Input directory not found: {args.input}")
        return 1

    pipeline = NaLiXPipeline(input_dir=args.input, output_dir=args.output)
    pipeline.execute(max_workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
