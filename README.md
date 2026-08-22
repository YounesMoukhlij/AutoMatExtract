# AutoMatExtract

**Author:** Younes Moukhlij — Software Engineering student at 1337 Coding School,
PhD researcher at USMS.

AutoMatExtract turns a folder of battery/materials-science PDFs into a structured, queryable
database. It parses each paper (text, tables, figures, equations, OCR fallback for scanned
pages), pulls out materials and their properties with SpaCy/regex/PyMatgen, ranks and
deduplicates candidate values, links materials to properties, and exports everything as linked
CSV files, a multi-sheet Excel workbook, and nested JSON.

Run it from the command line or from a small Tkinter desktop GUI — both use the exact same
pipeline, so results never differ between the two.

## Features

- **PDF parsing** — text, tables, and captions via PyMuPDF/pdfplumber, with per-block page and
  section tracking so every extracted value keeps its provenance.
- **OCR fallback** — scanned/image-only pages are rendered at 400 DPI, cleaned up with OpenCV
  (denoise, sharpen, adaptive threshold), and read with Tesseract.
- **NLP + rule-based extraction** — SpaCy dependency parsing and regex patterns pull materials,
  properties (electrochemical, thermodynamic, mechanical, structural, electronic, thermal,
  spectroscopy), and computational parameters (DFT, AIMD, ML) out of running text and tables.
- **Chemistry-aware normalization** — formulas and units are normalized and validated with
  PyMatgen and Pint (e.g. unit conversions, composition parsing).
- **Ranking & deduplication** — multiple candidate values for the same property are scored and
  reduced to the best-supported one, with a confidence score and full source trace.
- **Material–property relations** — links each property back to the material it describes
  (via dependency parsing, proximity, or table-row alignment).
- **Incremental runs** — re-running the pipeline on the same output folder skips papers that are
  already in `papers.csv` and only processes new PDFs; every export format supports merging into
  an existing database.
- **Multiple export formats** — normalized relational CSVs (`papers`, `materials`, `properties`,
  `relations`, `tables`, `figures`, `equations`, `workflow`), a multi-sheet Excel workbook, nested
  JSON, and a flat one-row-per-paper summary CSV for quick scanning.
- **CLI and GUI** — a scriptable command-line interface and a desktop GUI with worker-count
  control and Start/Stop/live progress.

## How it works

```
papers/*.pdf
    │
    ▼
parser.py        — PyMuPDF/pdfplumber: text blocks, tables, sections, references
    │ (falls back to ocr.py for image-only pages)
    ▼
extractor.py     — SpaCy NLP + regex + PyMatgen: candidate materials & properties
    │
    ▼
ranking.py       — dedupe candidates, score confidence, keep the best value per property
    │
    ▼
pipeline.py      — orchestrates parsing/extraction/ranking across PDFs (ThreadPoolExecutor)
    │
    ▼
exporter.py      — writes output/{database.json, database.xlsx, *.csv}
```

`models.py` defines the shared dataclasses (`PaperData`, `Material`, `ExtractedField`,
`Relation`, ...) that flow through every stage, and `normalizer.py` / `config.py` hold the
scientific value/unit normalization and regex/context configuration used by the extractor.

## Platform support

Linux, macOS, and Windows are all supported — the codebase uses only `os.path`, UTF-8 I/O, and
cross-platform libraries throughout (no OS-specific paths or shell calls). A few
platform-specific setup steps:

- **Tesseract OCR** is installed differently per OS (see Installation below). On Windows, the
  installer doesn't reliably add `tesseract.exe` to `PATH`; `ocr.py` auto-detects the default
  install location (`C:\Program Files\Tesseract-OCR\tesseract.exe`), or you can set the
  `TESSERACT_CMD` environment variable to its full path if installed elsewhere.
- **GUI (`gui.py`)** uses Tkinter, which ships with the standard Python installer on Windows and
  macOS. On Linux, it's often a separate OS package — e.g. `sudo apt install python3-tk`.

## Installation

Requires **Python 3.9+** and a system install of **Tesseract OCR** (used for scanned pages).

```bash
# Tesseract (system dependency)
brew install tesseract              # macOS
sudo apt install tesseract-ocr      # Debian/Ubuntu
sudo apt install python3-tk         # Debian/Ubuntu, only if using --gui
# Windows: download the installer from https://github.com/UB-Mannheim/tesseract/wiki

# Python dependencies
pip install -r requirements.txt

# SpaCy language model (not pinned in requirements.txt)
python -m spacy download en_core_web_sm
```

## Usage

### Command line

```bash
# Process every PDF in ./papers, write results to ./output
python main.py

# Custom input/output directories and worker count
python main.py --input papers --output output --workers 4

# Verbose (debug) logging
python main.py -v
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | `papers` | Directory containing input PDFs |
| `--output` | `-o` | `output` | Directory to write the extracted database into |
| `--workers` | `-w` | `4` | Number of PDFs to process in parallel |
| `--verbose` | `-v` | off | Enable debug-level logging |
| `--gui` | | off | Launch the desktop GUI instead of running from the CLI |

Re-running the command on the same `--output` directory is safe and incremental: papers already
recorded in `output/papers.csv` are skipped, and only new PDFs are processed and merged in.

### GUI

```bash
python main.py --gui
# or
python gui.py
```

Pick an input folder of PDFs and an output folder, choose the worker count, and hit Start. The
GUI shows live per-paper progress (materials found, headline properties extracted) and can be
stopped early without losing already-completed work.

## Output

Running the pipeline populates the output directory with:

| File | Contents |
|---|---|
| `papers.csv` | One row per paper: title, authors, journal, DOI, year, counts |
| `materials.csv` | Every material/composition found, with normalized formula and elements |
| `properties.csv` | Every extracted property value, with unit, confidence, page, section |
| `relations.csv` | Material ↔ property links, with the extraction method used |
| `activation_energy.csv` | Every ion-migration energy-barrier value found — under any name the paper used for it (activation energy, Ea, migration/hopping/diffusion barrier, energy barrier) — one row per candidate, with `matched_as` recording which term matched, material (when resolved), value, unit, confidence, page/section/sentence. A focused view so this commonly-needed property doesn't have to be filtered out of `properties.csv` |
| `tables.csv` | Extracted table records (caption, headers, rows) |
| `figures.csv` | Figure captions and cross-references to materials/properties/tables |
| `equations.csv` | Extracted equations with type, variables, and context sentence |
| `workflow.csv` | Ordered experimental/computational workflow steps per paper |
| `database.xlsx` | All of the above as sheets in one workbook |
| `database.json` | All of the above as nested JSON, one record per paper |
| `database_summary.csv` | Flat, one-row-per-paper legacy summary for quick scanning |

Every extracted value carries provenance (source page, section, sentence, source type) and a
confidence score, so results can be traced back to the exact place in the PDF they came from.

## Known limitations

Verified by running the pipeline end-to-end against real PDFs (text-based papers and a
synthetic scanned page) and the papers in `papers/`:

- **Extraction is regex/NLP-based, not perfect.** Expect occasional false positives: common
  words that coincide with material/software aliases (e.g. "crystal" matched as a DFT software
  name), or roman numerals in running text picked up as element symbols. Treat `confidence`
  scores in `properties.csv`/`materials.csv` as a filter, not a guarantee — spot-check anything
  used downstream.
- **Table extraction can misfire on complex PDF layouts.** pdfplumber sometimes extracts
  non-tabular page content (e.g. multi-column figure text) as a spurious low-confidence table
  with empty rows; real tables in the same papers extract correctly.
- **OCR quality depends on Tesseract being installed** (`brew install tesseract` /
  `sudo apt install tesseract-ocr`); without it, image-only/scanned pages silently fall back to
  the (near-empty) native text layer instead of erroring.

## Project structure

```
AutoMatExtract/
├── main.py          # CLI entry point, shared by CLI and GUI
├── gui.py           # Tkinter desktop interface
├── pipeline.py      # Orchestrates parsing → extraction → ranking → export
├── parser.py        # PyMuPDF/pdfplumber: text, tables, sections, references
├── ocr.py           # OpenCV preprocessing + Tesseract OCR for scanned pages
├── extractor.py     # SpaCy NLP, regex matching, PyMatgen integration
├── normalizer.py    # Scientific value & unit normalization/conversion
├── chemistry.py     # Chemistry/formula helpers
├── ranking.py       # Deduplication and confidence scoring
├── exporter.py      # CSV / Excel / JSON generation
├── models.py        # Shared dataclasses (PaperData, Material, ExtractedField, ...)
├── config.py        # Regex patterns and NLP context lists
├── requirements.txt
├── papers/          # Default input directory for PDFs
└── output/          # Default output directory for the extracted database
```

## Contributing

Contributions are welcome.

1. Fork the repo and create a feature branch.
2. Keep changes focused — the pipeline stages (`parser` → `extractor` → `ranking` → `exporter`)
   are intentionally decoupled, so a change to one stage shouldn't need to touch the others.
3. Preserve the "strict defaults" convention used throughout the codebase: missing/unparsed
   values are represented as `"NONE"` (or `ExtractedField.empty()`), not `None` or omitted keys,
   so downstream exports stay schema-consistent.
4. Test against a small set of real PDFs in `papers/` and check the generated files in `output/`
   before opening a PR.
5. Open a pull request describing the change and, if it affects extraction quality, include a
   before/after comparison on a sample paper.

Bug reports and feature requests are welcome via GitHub issues.

## License

No license file is currently included in this repository. Contact the maintainer before reusing
this code outside of personal/evaluation purposes.
