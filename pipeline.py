# pipeline.py

import concurrent.futures
import logging
import os
import threading
from typing import Callable, Optional, Set

import pandas as pd

from exporter import StrictExporter
from extractor import ExtractorEngine
from models import ExtractedField, PaperData
from parser import DocumentParser
from ranking import Ranker

class NaLiXPipeline:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def get_processed_files(self) -> Set[str]:
        """
        Reads the filenames already present in papers.csv so re-running the pipeline only
        processes new PDFs. Fails open (returns an empty set) if the database doesn't exist
        yet or is unreadable, so new papers are never silently skipped.
        """
        path = os.path.join(self.output_dir, "papers.csv")
        if not os.path.exists(path):
            return set()
        try:
            df = pd.read_csv(path, usecols=["filename"])
            return set(df["filename"].dropna().astype(str))
        except Exception as e:
            logging.warning(f"Could not read existing papers.csv ({e}); treating all PDFs as new.")
            return set()

    def process_document(self, filename: str) -> Optional[PaperData]:
        filepath = os.path.join(self.input_dir, filename)

        try:
            doc_parser = DocumentParser(filepath)
            raw_meta, blocks, references = doc_parser.parse()

            # Format Metadata into ExtractedFields
            metadata_fields = {
                k: ExtractedField(v, v, 1.0, "1", "Metadata", v, "Metadata") if v != "NONE" else ExtractedField.empty()
                for k, v in raw_meta.items()
            }

            extracted_raw = ExtractorEngine.extract_all(blocks)
            ranked = Ranker.process(extracted_raw)

            return PaperData(
                filename=filename,
                metadata=metadata_fields,
                materials=ranked["materials"],
                electrochemical=ranked["ELECTROCHEMICAL"],
                thermodynamic=ranked["THERMODYNAMIC"],
                mechanical=ranked["MECHANICAL"],
                structural=ranked["STRUCTURAL"],
                electronic=ranked["ELECTRONIC"],
                thermal=ranked["THERMAL"],
                spectroscopy=ranked["SPECTROSCOPY"],
                dft=ranked["DFT"],
                aimd=ranked["AIMD"],
                ml=ranked["ML"],
                synthesis=ranked["SYNTHESIS"],
                characterization=ranked["CHARACTERIZATION"],
                battery_perf={},
                interfaces={},
                relations=ranked["relations"],
                workflow=ranked["workflow"],
                figures=ranked["figures"],
                tables=ranked["tables"],
                equations=ranked["equations"],
                references=references,
            )
        except Exception as e:
            logging.error(f"Failed {filename}: {str(e)}")
            return None

    def execute(self, max_workers: int = 4, stop_event: Optional[threading.Event] = None,
                progress_callback: Optional[Callable[[str, Optional[PaperData]], None]] = None) -> None:
        """Runs extraction over every not-yet-processed PDF and exports the results.

        `stop_event` (optional): when set between files, cancels every PDF still queued and
        stops after the ones already running finish — lets a caller (e.g. the GUI) request an
        early stop without losing in-flight work.
        `progress_callback` (optional): called once per PDF as soon as it finishes, as
        `callback(filename, paper_or_none)` — `None` on failure — so a caller can show live
        per-paper feedback instead of waiting for the whole batch to end.
        """
        # 1. Identify which files have already been processed
        processed_files = self.get_processed_files()

        # 2. Filter the folder for only NEW, unprocessed PDFs
        all_pdfs = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.pdf')]
        new_pdfs = [f for f in all_pdfs if f not in processed_files]

        if not new_pdfs:
            if not all_pdfs:
                logging.info(f"No PDFs found in {self.input_dir}. Add some and re-run.")
            else:
                logging.info("All PDFs in the folder have already been processed. Database is up to date.")
            return

        logging.info(f"Skipping {len(processed_files)} existing papers.")
        logging.info(f"Starting extraction for {len(new_pdfs)} new papers...")

        # 3. Process only the new files (spaCy/pint/pymatgen are safe to share read-only
        # across these worker threads — see the module-level notes in extractor.py/normalizer.py).
        # Submitted (rather than mapped) so a stop request can cancel PDFs that haven't started yet.
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_document, f): f for f in new_pdfs}
            for future in concurrent.futures.as_completed(futures):
                filename = futures[future]
                r = future.result()
                if r:
                    results.append(r)
                else:
                    logging.warning(f"No data extracted from {filename} (parse/extraction failed).")
                if progress_callback is not None:
                    try:
                        progress_callback(filename, r)
                    except Exception as e:
                        logging.warning(f"progress_callback raised for {filename}: {e}")
                if stop_event is not None and stop_event.is_set():
                    for pending in futures:
                        pending.cancel()
                    logging.info(f"Stop requested — cancelled remaining queued PDFs "
                                 f"({len(results)}/{len(new_pdfs)} finished).")
                    break

        # 4. Export (CSV/Excel/JSON all support incremental merge with an existing database)
        if results:
            exporter = StrictExporter(self.output_dir)
            exporter.export(results)
            logging.info(f"Successfully added {len(results)} new papers to the database.")
