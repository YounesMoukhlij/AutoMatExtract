# exporter.py

import json
import logging
import os
import re
from dataclasses import asdict
from typing import Any, Dict, List

import pandas as pd

from models import ExtractedField, PaperData

# Characters XML (and therefore .xlsx) cannot store.
_ILLEGAL_XML_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Paper-level (not tied to a single material) categorized property dicts on PaperData.
CATEGORY_FIELDS = [
    "electrochemical", "thermodynamic", "mechanical", "structural", "electronic", "thermal",
    "spectroscopy", "dft", "aimd", "ml", "synthesis", "characterization", "battery_perf", "interfaces",
]


class StrictExporter:
    """Builds the normalized relational schema (schema section 16: papers/materials/properties/
    relations/tables/figures/equations/workflow) and writes it as linked CSV files, a multi-sheet
    Excel workbook, and nested JSON — plus the original flat one-row-per-paper summary CSV kept
    for backward-compatible quick scanning (schema rule: don't remove working functionality)."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Normalized table builders
    # ------------------------------------------------------------------
    @staticmethod
    def _field_to_material_map(paper: PaperData) -> Dict[int, str]:
        """Maps id(ExtractedField) -> owning Material.material_id, for the properties table's
        optional material_id foreign key (schema section 16: every material's own properties)."""
        mapping: Dict[int, str] = {}
        for m in paper.materials:
            for p in m.properties:
                mapping[id(p)] = m.material_id
        return mapping

    @classmethod
    def _build_rows(cls, papers: List[PaperData]) -> Dict[str, List[Dict[str, Any]]]:
        tables: Dict[str, List[Dict[str, Any]]] = {
            "papers": [], "materials": [], "properties": [], "relations": [],
            "tables": [], "figures": [], "equations": [], "workflow": [], "activation_energy": [],
        }

        for paper in papers:
            pid = paper.filename
            field_to_material = cls._field_to_material_map(paper)
            meta = paper.metadata

            def meta_val(key: str) -> str:
                return meta.get(key, ExtractedField.empty()).normalized_value

            tables["papers"].append({
                "paper_id": pid, "filename": paper.filename,
                "title": meta_val("title"), "authors": meta_val("authors"), "journal": meta_val("journal"),
                "doi": meta_val("doi"), "publication_year": meta_val("publication_year"),
                "n_materials": len(paper.materials), "n_relations": len(paper.relations),
                "n_references": len(paper.references),
            })

            for m in paper.materials:
                tables["materials"].append({
                    "paper_id": pid, "material_id": m.material_id, "raw_formula": m.raw_formula,
                    "normalized_formula": m.normalized_formula,
                    "elements": json.dumps(m.elements, ensure_ascii=False),
                    "is_known_alias": m.is_known_alias, "confidence": round(m.confidence, 3),
                    "page": m.source_page, "section": m.source_section, "source_type": m.source_type,
                })

            for cat_field in CATEGORY_FIELDS:
                for prop, cands in (getattr(paper, cat_field, {}) or {}).items():
                    for c in cands:
                        tables["properties"].append({
                            "paper_id": pid, "material_id": field_to_material.get(id(c), "NONE"),
                            "category": cat_field.upper(), "property_name": prop,
                            "raw_value": c.raw_value, "normalized_value": c.normalized_value,
                            "unit": c.unit, "ea_type": c.ea_type, "confidence": round(c.confidence, 3),
                            "page": c.source_page, "section": c.source_section,
                            "sentence": c.source_sentence, "source_type": c.source_type,
                        })

            # Dedicated activation-energy table. Papers describe the same physical quantity (the
            # hop/migration energy barrier for ion transport) under several names — "activation
            # energy", "Ea", "migration barrier", "hopping barrier", "diffusion barrier", "energy
            # barrier" — which config.py's regexes already split into two schema properties
            # (Activation_Energy / Migration_Barrier). Pool both here so this table is the single
            # place to look regardless of which term the paper happened to use; "matched_as"
            # records which one it was. material_id is "NONE" only when nothing resolved this
            # candidate to a specific material (schema section 17); material_formula relies on
            # the same id()-based link Ranker.py uses (see ranking.py).
            material_by_field_id = {id(r.property_value): (r.material_id, r.material_formula)
                                     for r in paper.relations}
            activation_energy_sources = [
                ("Activation Energy", paper.electrochemical.get("Activation_Energy", [])),
                ("Migration Barrier", paper.electrochemical.get("Migration_Barrier", [])),
            ]
            for matched_as, candidates in activation_energy_sources:
                for c in candidates:
                    material_id, material_formula = material_by_field_id.get(id(c), ("NONE", "NONE"))
                    tables["activation_energy"].append({
                        "paper_id": pid, "material_id": material_id, "material_formula": material_formula,
                        "matched_as": matched_as, "ea_type": c.ea_type,
                        "raw_value": c.raw_value, "normalized_value": c.normalized_value, "unit": c.unit,
                        "confidence": round(c.confidence, 3), "page": c.source_page, "section": c.source_section,
                        "sentence": c.source_sentence, "source_type": c.source_type,
                    })

            for r in paper.relations:
                tables["relations"].append({
                    "paper_id": pid, "material_id": r.material_id, "material_formula": r.material_formula,
                    "category": r.category, "property_name": r.property_name,
                    "value": r.property_value.normalized_value, "unit": r.property_value.unit,
                    "ea_type": r.property_value.ea_type,
                    "confidence": round(r.confidence, 3), "extraction_method": r.extraction_method,
                    "sentence": r.source_sentence, "page": r.source_page,
                })

            for t in paper.tables:
                tables["tables"].append({
                    "paper_id": pid, "table_number": t.table_number, "caption": t.caption,
                    "page": t.page, "section": t.section,
                    "headers": json.dumps(t.headers, ensure_ascii=False),
                    "rows": json.dumps(t.rows, ensure_ascii=False),
                    "material_column": t.material_column,
                })

            for f in paper.figures:
                tables["figures"].append({
                    "paper_id": pid, "figure_number": f.figure_number, "caption": f.caption,
                    "page": f.page, "section": f.section,
                    "referenced_materials": " | ".join(f.referenced_materials) or "NONE",
                    "referenced_properties": " | ".join(f.referenced_properties) or "NONE",
                    "referenced_tables": " | ".join(f.referenced_tables) or "NONE",
                    "referenced_equations": " | ".join(f.referenced_equations) or "NONE",
                    "mentioned_methods": " | ".join(f.mentioned_methods) or "NONE",
                })

            for e in paper.equations:
                tables["equations"].append({
                    "paper_id": pid, "equation_type": e.equation_type, "equation_text": e.equation_text,
                    "page": e.page, "section": e.section,
                    "variables": " | ".join(e.variables) or "NONE", "context_sentence": e.context_sentence,
                })

            for w in paper.workflow:
                tables["workflow"].append({
                    "paper_id": pid, "step_order": w.order, "stage": w.stage,
                    "evidence_sentence": w.evidence_sentence, "page": w.page, "section": w.section,
                })

        return tables

    # ------------------------------------------------------------------
    # Legacy flat one-row-per-paper summary (kept so nothing existing is removed)
    # ------------------------------------------------------------------
    @staticmethod
    def to_flat_dict(paper: PaperData) -> Dict[str, Any]:
        row: Dict[str, Any] = {"Filename": paper.filename}

        for k, v in paper.metadata.items():
            row.update(v.to_flat_dict(f"Meta_{k}"))

        for cat_field in CATEGORY_FIELDS:
            prefix = cat_field[:5].capitalize()
            for prop_name, cands in (getattr(paper, cat_field, {}) or {}).items():
                best = cands[0] if cands else ExtractedField.empty()
                row.update(best.to_flat_dict(f"{prefix}_{prop_name}"))
                for i in range(1, 3):
                    cand = cands[i] if len(cands) > i else ExtractedField.empty()
                    row[f"{prefix}_{prop_name}_Cand{i + 1}"] = cand.normalized_value

        top_mats = [m.normalized_formula for m in paper.materials[:5]]
        row["Top_Materials"] = " | ".join(top_mats) if top_mats else "NONE"

        rels = [f"{r.material_formula}[{r.property_name}={r.property_value.normalized_value}]"
                for r in paper.relations[:5]]
        row["Top_Relations"] = " | ".join(rels) if rels else "NONE"

        return row

    # ------------------------------------------------------------------
    # Export: CSV (one file per table), Excel (one sheet per table), JSON (nested per paper)
    # ------------------------------------------------------------------
    def export(self, new_dataset: List[PaperData]) -> None:
        if not new_dataset:
            return

        tables = self._build_rows(new_dataset)
        self._export_csv(tables)
        self._export_excel(tables)
        self._export_json(new_dataset)
        self._export_legacy_summary(new_dataset)

    def _export_csv(self, tables: Dict[str, List[Dict[str, Any]]]) -> None:
        for name, rows in tables.items():
            path = os.path.join(self.output_dir, f"{name}.csv")
            new_df = self._strict_none(pd.DataFrame(rows))
            final_df = self._merge_csv(path, new_df)
            final_df.to_csv(path, index=False, encoding="utf-8-sig")

    def _export_excel(self, tables: Dict[str, List[Dict[str, Any]]]) -> None:
        path = os.path.join(self.output_dir, "database.xlsx")
        existing: Dict[str, pd.DataFrame] = {}
        if os.path.exists(path):
            try:
                existing = pd.read_excel(path, sheet_name=None)
            except Exception as e:
                logging.warning(f"Could not read existing Excel workbook, overwriting: {e}")

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, rows in tables.items():
                new_df = self._strict_none(pd.DataFrame(rows))
                combined = pd.concat([existing[name], new_df], ignore_index=True) if name in existing else new_df
                combined = self._strict_none(combined)
                combined.to_excel(writer, sheet_name=name[:31], index=False)

    def _export_json(self, new_dataset: List[PaperData]) -> None:
        path = os.path.join(self.output_dir, "database.json")
        existing_papers: List[Dict[str, Any]] = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    existing_papers = existing.get("papers", []) if isinstance(existing, dict) else existing
            except Exception as e:
                logging.warning(f"Could not read existing JSON database, overwriting: {e}")

        existing_filenames = {p.get("filename") for p in existing_papers}
        new_papers = [self._paper_to_nested_dict(p) for p in new_dataset if p.filename not in existing_filenames]

        with open(path, "w", encoding="utf-8") as f:
            json.dump({"papers": existing_papers + new_papers}, f, indent=2, ensure_ascii=False)

    def _export_legacy_summary(self, new_dataset: List[PaperData]) -> None:
        path = os.path.join(self.output_dir, "database_summary.csv")
        new_df = self._strict_none(pd.DataFrame([self.to_flat_dict(p) for p in new_dataset]))
        final_df = self._merge_csv(path, new_df)
        final_df.to_csv(path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_csv(path: str, new_df: pd.DataFrame) -> pd.DataFrame:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return new_df
        try:
            old_df = pd.read_csv(path)
            return StrictExporter._strict_none(pd.concat([old_df, new_df], ignore_index=True))
        except pd.errors.EmptyDataError:
            return new_df
        except Exception as e:
            logging.warning(f"Could not merge with existing {path}: {e}")
            return new_df

    @staticmethod
    def _strict_none(df: pd.DataFrame) -> pd.DataFrame:
        """Enforces the strict-NONE policy (schema section 17): never NaN/null/empty string."""
        if df.empty:
            return df
        df = df.fillna("NONE").replace("", "NONE")
        # Control characters left over from PDF extraction make openpyxl raise IllegalCharacterError
        # and abort the whole workbook export, so they are stripped from every text cell.
        return df.apply(lambda col: col.map(
            lambda v: _ILLEGAL_XML_CHARS.sub("", v) if isinstance(v, str) else v))

    @classmethod
    def _paper_to_nested_dict(cls, paper: PaperData) -> Dict[str, Any]:
        def field_dict(f: ExtractedField) -> Dict[str, Any]:
            return {
                "raw_value": f.raw_value, "normalized_value": f.normalized_value, "unit": f.unit,
                "confidence": round(f.confidence, 3), "page": f.source_page, "section": f.source_section,
                "sentence": f.source_sentence, "source_type": f.source_type,
                **({"ea_type": f.ea_type} if f.ea_type != "NONE" else {}),
            }

        return {
            "filename": paper.filename,
            "metadata": {k: field_dict(v) for k, v in paper.metadata.items()},
            "materials": [
                {
                    "material_id": m.material_id, "raw_formula": m.raw_formula,
                    "normalized_formula": m.normalized_formula, "elements": m.elements,
                    "is_known_alias": m.is_known_alias, "confidence": round(m.confidence, 3),
                    "page": m.source_page, "section": m.source_section,
                    "properties": [field_dict(p) for p in m.properties],
                }
                for m in paper.materials
            ],
            "paper_level_properties": {
                cat_field.upper(): {prop: [field_dict(c) for c in cands] for prop, cands in
                                     (getattr(paper, cat_field, {}) or {}).items() if cands}
                for cat_field in CATEGORY_FIELDS
            },
            "relations": [
                {
                    "material_id": r.material_id, "material_formula": r.material_formula,
                    "category": r.category, "property_name": r.property_name,
                    "value": r.property_value.normalized_value, "unit": r.property_value.unit,
                    "ea_type": r.property_value.ea_type,
                    "confidence": round(r.confidence, 3), "extraction_method": r.extraction_method,
                    "sentence": r.source_sentence, "page": r.source_page,
                }
                for r in paper.relations
            ],
            "tables": [asdict(t) for t in paper.tables],
            "figures": [asdict(f) for f in paper.figures],
            "equations": [asdict(e) for e in paper.equations],
            "workflow": [asdict(w) for w in paper.workflow],
            "references": paper.references,
        }
