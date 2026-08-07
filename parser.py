# parser.py

import datetime
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import fitz
import pdfplumber

from config import SchemaConfig
from normalizer import normalize_scientific_text
from ocr import run_ocr

_CAPTION_REF_REGEX = re.compile(r'^(Fig(?:ure)?s?|Table)\.?\s*(\d+)', re.IGNORECASE)
_REFERENCE_START_REGEX = re.compile(r'^\[?\d{1,3}\]?\.?\s+\S')
_DOI_REGEX = re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+')
_YEAR_CONTEXT_REGEX = re.compile(r'(?:©|Received|Published|Accepted|Copyright)[^\d]{0,20}(\d{4})', re.IGNORECASE)
_YEAR_REGEX = re.compile(r'\b(19|20)\d{2}\b')
_EQUATION_NUMBERED_REGEX = re.compile(r'=.*\(\s*\d+\s*\)\s*$')
_PROSE_STARTERS = ('the ', 'this ', 'in ', 'for ', 'we ', 'it ', 'these', 'when ', 'if ',
                    'a ', 'an ', 'as ', 'to ', 'that ')

# Generous, non-maintained sanity ceiling for extracted publication years.
_MAX_PLAUSIBLE_YEAR = datetime.date.today().year + 1


class DocumentParser:
    """
    Intelligently parses scientific PDFs.
    Extracts text, identifies structural sections, isolates tables/captions/equations,
    and falls back to font-size analysis if metadata is missing.
    """

    # Flattened synonym -> canonical section lookup, longest synonym first so e.g. "results and
    # discussion" matches before the bare "results" prefix would.
    _SECTION_LOOKUP: List[Tuple[str, str]] = sorted(
        ((syn, canon) for canon, syns in SchemaConfig.SECTION_SYNONYMS.items() for syn in syns),
        key=lambda pair: len(pair[0]), reverse=True,
    )

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.current_section = "Metadata/Header"
        self._in_references = False
        self._reference_lines: List[str] = []

    def parse(self) -> Tuple[Dict[str, str], List[Dict[str, Any]], List[str]]:
        """Returns structured metadata, a list of contextual blocks, and split reference entries."""
        doc = fitz.open(self.filepath)

        metadata = self._extract_advanced_metadata(doc)

        blocks: List[Dict[str, Any]] = []
        first_page_text, last_page_text = "", ""

        for page_num, page in enumerate(doc, start=1):
            text = self._extract_reading_order_text(page)

            # Trigger OCR if the page appears to be a scanned image
            if len(text.strip()) < 50:
                logging.info(f"Low text density on page {page_num} of {self.filepath}. Triggering OCR...")
                ocr_text = run_ocr(page)
                text = ocr_text if ocr_text != "NONE" else text

            if page_num == 1:
                first_page_text = text
            last_page_text = text

            blocks.extend(self._process_page_text(text, page_num))

        doc.close()

        metadata["doi"] = self._extract_doi(metadata, first_page_text, last_page_text)
        metadata["publication_year"] = self._extract_year(metadata, first_page_text, last_page_text)

        # Dedicated Table Extraction via pdfplumber (needs the Caption blocks above for numbering).
        blocks.extend(self._extract_tables(blocks))

        references = self._split_references(self._reference_lines)

        return metadata, blocks, references

    # ------------------------------------------------------------------
    # Reading order / multi-column layout (schema section 20: robust to multi-column layouts)
    # ------------------------------------------------------------------
    def _extract_reading_order_text(self, page: "fitz.Page") -> str:
        """
        Reconstructs a plausible reading order for single- and two-column academic layouts by
        bucketing text blocks into left/right halves via x-position and sorting each bucket
        top-to-bottom, instead of trusting fitz's raw block order (which can interleave columns).
        """
        try:
            raw_blocks = page.get_text("blocks")
        except Exception:
            return page.get_text("text")

        midpoint = page.rect.width / 2
        text_blocks = [b for b in raw_blocks if len(b) >= 7 and b[6] == 0 and b[4].strip()]
        text_blocks.sort(key=lambda b: (0 if b[0] < midpoint else 1, round(b[1], 1)))

        return "\n".join(b[4] for b in text_blocks)

    # ------------------------------------------------------------------
    # Page text -> typed blocks
    # ------------------------------------------------------------------
    def _process_page_text(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Processes text line-by-line using a state machine to track the current section and
        group lines into paragraphs, figure/table captions, equation candidates, and (inside
        the References section) individual reference entries.
        """
        text = normalize_scientific_text(text)
        page_blocks = []
        lines = text.split('\n')
        current_paragraph = ""

        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue

            # A. Section Header Detection (checked first so a "References" header itself
            # doesn't get swallowed as a reference entry or body text).
            if len(clean_line) < 60:
                matched_section = self._detect_section(clean_line)
                if matched_section:
                    if current_paragraph:
                        page_blocks.append(self._create_block(current_paragraph, page_num, "Text"))
                        current_paragraph = ""
                    self.current_section = matched_section
                    self._in_references = (matched_section == "References")
                    continue

            # B. Inside the References section: accumulate raw lines for later splitting.
            if self._in_references:
                self._reference_lines.append(clean_line)
                continue

            # C. Figure and Table Caption Detection
            caption_match = _CAPTION_REF_REGEX.match(clean_line)
            if caption_match:
                if current_paragraph:
                    page_blocks.append(self._create_block(current_paragraph, page_num, "Text"))
                    current_paragraph = ""
                ref_kind = "Table" if caption_match.group(1).lower().startswith("table") else "Figure"
                block = self._create_block(clean_line, page_num, "Caption")
                block["ref_kind"] = ref_kind
                block["ref_number"] = caption_match.group(2)
                page_blocks.append(block)
                continue

            # D. Equation candidate: short, symbol-dense, not ordinary prose.
            if self._looks_like_equation(clean_line):
                if current_paragraph:
                    page_blocks.append(self._create_block(current_paragraph, page_num, "Text"))
                    current_paragraph = ""
                page_blocks.append(self._create_block(clean_line, page_num, "EquationCandidate"))
                continue

            # E. Accumulate standard text lines into a continuous paragraph
            current_paragraph += " " + clean_line

        if current_paragraph:
            page_blocks.append(self._create_block(current_paragraph, page_num, "Text"))

        return page_blocks

    @staticmethod
    def _looks_like_equation(line: str) -> bool:
        if '=' not in line or len(line) > 80:
            return False
        if line.lower().startswith(_PROSE_STARTERS):
            return False
        if _EQUATION_NUMBERED_REGEX.search(line):
            return True
        words = line.split()
        if not words:
            return False
        long_words = sum(1 for w in words if len(w) > 3 and w.isalpha())
        return long_words <= 1 and len(words) <= 12

    def _create_block(self, text: str, page_num: int, block_type: str) -> Dict[str, Any]:
        """Helper to standardize block dictionaries."""
        return {
            "text": text.strip(),
            "page": page_num,
            "section": self.current_section,
            "type": block_type,
        }

    def _detect_section(self, line: str) -> str:
        """Determines if a line is an academic section header, resolving synonyms (§12)."""
        lower_line = line.lower()
        clean_line = re.sub(r'^([ivx\d]+\.?)+\s*', '', lower_line).strip()
        for synonym, canonical in self._SECTION_LOOKUP:
            if clean_line.startswith(synonym):
                return canonical
        return ""

    # ------------------------------------------------------------------
    # References (schema section 1: references extraction)
    # ------------------------------------------------------------------
    @staticmethod
    def _split_references(lines: List[str]) -> List[str]:
        entries: List[str] = []
        current = ""
        for line in lines:
            if _REFERENCE_START_REGEX.match(line):
                if current:
                    entries.append(current.strip())
                current = line
            else:
                current += " " + line
        if current:
            entries.append(current.strip())
        return entries

    # ------------------------------------------------------------------
    # Metadata: title/authors/journal (existing) + DOI/publication year (schema section 1)
    # ------------------------------------------------------------------
    def _extract_advanced_metadata(self, doc: "fitz.Document") -> Dict[str, str]:
        """
        Extracts metadata natively, falling back to font-size analysis on the first page
        to locate the Title if the embedded PDF metadata is missing or corrupted.
        """
        meta = doc.metadata
        title = meta.get("title", "") or ""
        authors = meta.get("author", "") or ""
        journal = meta.get("subject", "") or ""

        if not title or "untitled" in title.lower():
            try:
                page_dict = doc[0].get_text("dict")
                largest_text = ""
                max_size = 0
                for block in page_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["size"] > max_size and len(span["text"].strip()) > 5:
                                    max_size = span["size"]
                                    largest_text = span["text"]
                if largest_text:
                    title = largest_text.strip()
            except Exception as e:
                logging.debug(f"Font-size title extraction failed: {e}")

        return {
            "title": title.strip() if title and title.strip() else "NONE",
            "authors": authors.strip() if authors and authors.strip() else "NONE",
            "journal": journal.strip() if journal and journal.strip() else "NONE",
            "creation_date": meta.get("creationDate", "NONE") or "NONE",
        }

    @staticmethod
    def _extract_doi(metadata: Dict[str, str], first_page_text: str, last_page_text: str) -> str:
        for source in (first_page_text, last_page_text, metadata.get("title", ""), metadata.get("journal", "")):
            m = _DOI_REGEX.search(source or "")
            if m:
                return m.group().rstrip('.,;)')
        return "NONE"

    @staticmethod
    def _extract_year(metadata: Dict[str, str], first_page_text: str, last_page_text: str) -> str:
        for source in (first_page_text, last_page_text):
            m = _YEAR_CONTEXT_REGEX.search(source or "")
            if m and 1950 <= int(m.group(1)) <= _MAX_PLAUSIBLE_YEAR:
                return m.group(1)
        for source in (first_page_text, last_page_text):
            for m in _YEAR_REGEX.finditer(source or ""):
                year = int(m.group())
                if 1950 <= year <= _MAX_PLAUSIBLE_YEAR:
                    return str(year)
        m = re.search(r'(19|20)\d{2}', metadata.get("creation_date", "") or "")
        if m and 1950 <= int(m.group()) <= _MAX_PLAUSIBLE_YEAR:
            return m.group()
        return "NONE"

    # ------------------------------------------------------------------
    # Tables: structure-preserving (schema section 8), not flattened to joined strings.
    # ------------------------------------------------------------------
    def _extract_tables(self, existing_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        captions_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for b in existing_blocks:
            if b["type"] == "Caption" and b.get("ref_kind") == "Table":
                captions_by_page.setdefault(b["page"], []).append(b)

        blocks = []
        try:
            with pdfplumber.open(self.filepath) as pdf:
                for page_idx, page in enumerate(pdf.pages, 1):
                    extracted = page.extract_tables()
                    for t_idx, table in enumerate(extracted, 1):
                        block = self._build_table_block(table, page_idx, t_idx, captions_by_page.get(page_idx, []))
                        if block:
                            blocks.append(block)
        except Exception as e:
            logging.warning(f"Table extraction failed for {self.filepath} via pdfplumber: {e}")

        return blocks

    @staticmethod
    def _build_table_block(table: List[List[Any]], page_idx: int, t_idx: int,
                            page_captions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not table or not table[0]:
            return None

        # Forward-fill None/blank header cells (merged-cell spans) before cleaning.
        headers: List[str] = []
        last = "Column"
        for cell in table[0]:
            val = normalize_scientific_text(str(cell).replace('\n', ' ').strip()) if cell is not None else ""
            if val:
                last = val
            headers.append(last or "Column")
        # De-duplicate repeated header names (e.g. a merged multi-column header) with a suffix.
        seen: Dict[str, int] = {}
        dedup_headers = []
        for h in headers:
            seen[h] = seen.get(h, 0) + 1
            dedup_headers.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
        headers = dedup_headers

        rows: List[Dict[str, str]] = []
        raw_lines = [" | ".join(headers)]
        for raw_row in table[1:]:
            cleaned = [normalize_scientific_text(str(c).replace('\n', ' ').strip()) if c is not None else ""
                       for c in raw_row]
            if not any(cleaned):
                continue
            row_dict = {(headers[i] if i < len(headers) else f"Column_{i}"): cleaned[i]
                        for i in range(len(cleaned))}
            rows.append(row_dict)
            raw_lines.append(" | ".join(cleaned))

        caption_text, table_number = "NONE", str(t_idx)
        if page_captions:
            match = next((c for c in page_captions if c.get("ref_number") == str(t_idx)), page_captions[0])
            caption_text = match["text"]
            table_number = match.get("ref_number", table_number)

        material_column = "NONE"
        for h in headers:
            if h.lower() in ("material", "materials", "sample", "composition", "compound", "electrolyte"):
                material_column = h
                break

        return {
            "text": "\n".join(raw_lines),
            "page": page_idx,
            "section": "Results",  # Tables inherently bias toward Results
            "type": "Table",
            "table_number": table_number,
            "caption": caption_text,
            "headers": headers,
            "rows": rows,
            "material_column": material_column,
        }
