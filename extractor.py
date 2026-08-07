# extractor.py

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import spacy

from chemistry import ChemistryEngine
from config import NUM, SchemaConfig
from models import EquationRecord, ExtractedField, FigureRecord, Material, Relation, TableRecord, WorkflowStep
from normalizer import DataNormalizer

# SciSpaCy is preferred for scientific literature; falls back to the general English model.
# Loaded once at import time and shared read-only across ThreadPoolExecutor workers — spaCy's
# documented pattern (inference calls don't mutate the shared pipeline).
try:
    nlp = spacy.load("en_core_sci_sm")
except OSError:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

_HAS_PARSER = "parser" in nlp.pipe_names or "senter" in nlp.pipe_names

_EQUATION_TYPE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(pattern), etype) for pattern, etype in SchemaConfig.EQUATION_PATTERNS.items()
]
_WORKFLOW_STAGE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(pattern), stage) for pattern, stage in SchemaConfig.WORKFLOW_STAGES
]
_VAR_TOKEN_REGEX = re.compile(r'\b([a-zA-Zσαβγμδρω∂][a-zA-Z0-9]{0,2})\b')
_VAR_STOPWORDS = {"the", "of", "in", "is", "at", "to", "for", "and", "or", "eq", "we"}

_CELL_NUM_UNIT = re.compile(rf'{NUM}\s*([A-Za-zΩ°μ%][A-Za-zΩ°μ%/\.\d⁻¹²³⁰⁴⁵⁶⁷⁸⁹\^-]*)?')
_HEADER_UNIT = re.compile(r'\(([^)]+)\)')
_EQ_REF_REGEX = re.compile(r'(?i)\bEq(?:uation)?s?\.?\s*(\d+)')
_TABLE_REF_REGEX = re.compile(r'(?i)\bTables?\.?\s*(\d+)')

# Table header keyword -> schema (category, property). Checked longest-keyword-first so
# "ionic conductivity" matches before the bare "conductivity" fallback.
_HEADER_PROPERTY_KEYWORDS: List[Tuple[str, str, str]] = sorted([
    ("ionic conductivity", "ELECTROCHEMICAL", "Ionic_Conductivity"),
    ("electronic conductivity", "ELECTROCHEMICAL", "Electronic_Conductivity"),
    ("conductivity", "ELECTROCHEMICAL", "Ionic_Conductivity"),
    ("activation energy", "ELECTROCHEMICAL", "Activation_Energy"),
    ("migration barrier", "ELECTROCHEMICAL", "Migration_Barrier"),
    ("diffusion coefficient", "ELECTROCHEMICAL", "Diffusion_Coefficient"),
    ("diffusivity", "ELECTROCHEMICAL", "Diffusion_Coefficient"),
    ("capacity retention", "ELECTROCHEMICAL", "Capacity_Retention"),
    ("specific capacity", "ELECTROCHEMICAL", "Specific_Capacity"),
    ("capacity", "ELECTROCHEMICAL", "Capacity"),
    ("coulombic efficiency", "ELECTROCHEMICAL", "Coulombic_Efficiency"),
    ("voltage", "ELECTROCHEMICAL", "Voltage"),
    ("energy density", "ELECTROCHEMICAL", "Energy_Density"),
    ("power density", "ELECTROCHEMICAL", "Power_Density"),
    ("current density", "ELECTROCHEMICAL", "Current_Density"),
    ("band gap", "ELECTRONIC", "Band_Gap"),
    ("formation energy", "THERMODYNAMIC", "Formation_Energy"),
    ("bulk modulus", "MECHANICAL", "Bulk_Modulus"),
    ("young's modulus", "MECHANICAL", "Young_Modulus"),
    ("cell volume", "STRUCTURAL", "Cell_Volume"),
    ("density", "STRUCTURAL", "Density"),
    ("space group", "STRUCTURAL", "Space_Group"),
    ("crystal system", "STRUCTURAL", "Crystal_System"),
    ("grain size", "STRUCTURAL", "Grain_Size"),
    ("particle size", "STRUCTURAL", "Particle_Size"),
    ("porosity", "STRUCTURAL", "Porosity"),
    ("melting point", "THERMAL", "Melting_Point"),
    ("thermal conductivity", "THERMAL", "Thermal_Conductivity"),
    ("cutoff energy", "DFT", "Cutoff_Energy"),
    ("temperature", "AIMD", "Temperature"),
], key=lambda x: len(x[0]), reverse=True)


class ExtractorEngine:

    # ------------------------------------------------------------------
    # Materials (delegated to chemistry.py — schema section 2)
    # ------------------------------------------------------------------
    @staticmethod
    def extract_materials(text: str) -> List[Dict[str, Any]]:
        return ChemistryEngine.find_materials(text)

    # ------------------------------------------------------------------
    # Dependency-parse relation linking (schema section 7)
    # ------------------------------------------------------------------
    @staticmethod
    def _dep_distance(tok_a, tok_b) -> int:
        if tok_a is None or tok_b is None:
            return 10_000
        chain_a = [tok_a] + list(tok_a.ancestors)
        chain_b = [tok_b] + list(tok_b.ancestors)
        pos_b = {t.i: d for d, t in enumerate(chain_b)}
        for d_a, t in enumerate(chain_a):
            if t.i in pos_b:
                return d_a + pos_b[t.i]
        return 10_000

    @staticmethod
    def _token_at(doc, start: int, end: int):
        span = doc.char_span(start, end, alignment_mode="expand")
        if span is not None and len(span) > 0:
            return span.root
        candidates = [t for t in doc if t.idx <= start]
        return candidates[-1] if candidates else (doc[0] if len(doc) else None)

    @classmethod
    def _link_property_to_material(cls, doc, materials: List[Dict], prop_start: int, prop_end: int) -> Tuple[Dict, str]:
        """Returns (best_material, extraction_method). Only ever links to ONE material —
        the syntactically (or, as a fallback, positionally) closest one in the sentence."""
        if len(materials) == 1:
            return materials[0], "proximity"

        if _HAS_PARSER and doc is not None:
            prop_tok = cls._token_at(doc, prop_start, prop_end)
            best, best_dist = None, None
            for mat in materials:
                mat_tok = cls._token_at(doc, mat["start"], mat["end"])
                dist = cls._dep_distance(prop_tok, mat_tok)
                if best_dist is None or dist < best_dist:
                    best, best_dist = mat, dist
            if best is not None and best_dist is not None and best_dist < 10_000:
                return best, "dependency_parse"

        # Fallback: nearest material by character distance to the property value.
        best = min(materials, key=lambda m: min(abs(m["start"] - prop_end), abs(prop_start - m["end"])))
        return best, "proximity"

    # ------------------------------------------------------------------
    # Sentence-level scoring (lightweight — ranking.py does the heavier multi-factor pass)
    # ------------------------------------------------------------------
    @staticmethod
    def _score_context(sentence: str, section: str, b_type: str) -> float:
        if section == "References":
            return 0.0
        score = 0.5
        sent_lower = sentence.lower()
        if b_type == "Table":
            score += 0.35
        elif b_type == "Caption":
            score += 0.25
        if section in ("Results", "Abstract", "Conclusion"):
            score += 0.2
        elif section == "Boilerplate":
            score -= 0.3
        elif section == "Supplementary":
            score -= 0.1
        if any(c in sent_lower for c in SchemaConfig.POSITIVE_CUES):
            score += 0.15
        if any(c in sent_lower for c in SchemaConfig.NEGATIVE_CUES):
            score -= 0.35
        return max(0.01, min(score, 0.99))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    @classmethod
    def extract_all(cls, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            cat: {prop: [] for prop in SchemaConfig.SCHEMA[cat]} for cat in SchemaConfig.SCHEMA
        }
        results["relations"] = []
        results["equations"] = []
        results["figures"] = []
        results["tables"] = []

        materials_by_formula: Dict[str, Material] = {}
        material_counter = [0]
        seen_equation_types = set()

        def get_or_create_material(cand: Dict[str, Any], score: float, page: str, section: str,
                                    sentence: str, b_type: str) -> Material:
            key = cand["formula"]
            existing = materials_by_formula.get(key)
            if existing:
                existing.confidence = max(existing.confidence, score)
                return existing
            material_counter[0] += 1
            mat = Material(
                material_id=f"MAT_{material_counter[0]}",
                raw_formula=cand["raw"], normalized_formula=cand["formula"],
                elements=cand["elements"], is_known_alias=cand["is_known_alias"],
                confidence=score, source_page=page, source_section=section,
                source_sentence=sentence, source_type=b_type,
            )
            materials_by_formula[key] = mat
            return mat

        for block in blocks:
            try:
                cls._process_block(block, results, get_or_create_material, seen_equation_types)
            except Exception as e:
                # One malformed block (bad OCR text, unusual table shape, ...) must not lose the
                # rest of the paper's extraction — schema section 20: robust against malformed input.
                logging.warning(f"Skipping block on page {block.get('page', '?')} after error: {e}")

        results["materials"] = list(materials_by_formula.values())
        results["workflow"] = cls._build_workflow(blocks)
        return results

    @classmethod
    def _process_block(cls, block: Dict[str, Any], results: Dict[str, Any], get_or_create_material,
                        seen_equation_types: set) -> None:
        if len(block.get("text", "")) < 3:
            return

        if block["type"] == "EquationCandidate":
            cls._record_equation(results, block["text"], block["text"], str(block["page"]),
                                  block["section"], seen_equation_types, force=True)
            return

        if block["type"] == "Table":
            cls._process_table_block(block, results, get_or_create_material)
            return

        if block["type"] == "Caption" and block.get("ref_kind") == "Figure":
            results["figures"].append(cls._build_figure(block))

        sentences = [s.text for s in nlp(block["text"]).sents] if block["type"] in ("Text", "Caption") \
            else [block["text"]]

        for sent in sentences:
            if len(sent.strip()) < 4:
                continue
            score = cls._score_context(sent, block["section"], block["type"])
            if score == 0.0:
                continue
            cls._process_sentence(sent, score, str(block["page"]), block["section"], block["type"],
                                   results, get_or_create_material, seen_equation_types)

    # ------------------------------------------------------------------
    # Workflow (schema section 11): ordered first-occurrence scan for pipeline-stage keywords,
    # preserving the paper's own order rather than forcing a fixed template.
    # ------------------------------------------------------------------
    @classmethod
    def _build_workflow(cls, blocks: List[Dict[str, Any]]) -> List[WorkflowStep]:
        steps: List[WorkflowStep] = []
        seen_stages = set()
        for block in blocks:
            if block["type"] not in ("Text", "Caption", "EquationCandidate") or block["section"] == "References":
                continue
            text = block["text"]
            for pattern, stage in _WORKFLOW_STAGE_PATTERNS:
                if stage in seen_stages:
                    continue
                if pattern.search(text):
                    seen_stages.add(stage)
                    steps.append(WorkflowStep(
                        order=len(steps) + 1, stage=stage, evidence_sentence=text[:200],
                        page=str(block["page"]), section=block["section"],
                    ))
        return steps

    @classmethod
    def _process_sentence(cls, sent: str, score: float, page: str, section: str, b_type: str,
                           results: Dict[str, Any], get_or_create_material, seen_equation_types: set) -> None:
        doc = nlp(sent) if _HAS_PARSER else None
        materials = ChemistryEngine.find_materials(sent)

        prop_matches = []  # (category, prop, field, start, end)
        for cat, props in SchemaConfig.COMPILED_SCHEMA.items():
            for prop, (pattern, norm_type) in props.items():
                for match in pattern.finditer(sent):
                    raw = match.groups() if match.groups() else match.group()
                    normalized = DataNormalizer.normalize(raw, norm_type)
                    if normalized == "NONE":
                        continue
                    unit = normalized.split(" ", 1)[1] if " " in normalized and norm_type not in ("string",) else "NONE"
                    field = ExtractedField(
                        raw_value=str(raw), normalized_value=normalized, confidence=score,
                        source_page=page, source_section=section, source_sentence=sent, source_type=b_type,
                        unit=unit, category=cat, property_name=prop,
                    )
                    results[cat][prop].append(field)
                    grp = match.lastindex or 0
                    start, end = (match.start(grp), match.end(grp)) if grp else (match.start(), match.end())
                    prop_matches.append((cat, prop, field, start, end))

        cls._scan_equation_mentions(sent, page, section, seen_equation_types, results)

        if not materials or not prop_matches:
            return

        for cat, prop, field, start, end in prop_matches:
            best_mat, method = cls._link_property_to_material(doc, materials, start, end)
            material = get_or_create_material(best_mat, score, page, section, sent, b_type)
            material.properties.append(field)
            results["relations"].append(Relation(
                material_id=material.material_id, material_formula=material.normalized_formula,
                category=cat, property_name=prop, property_value=field,
                confidence=min(0.99, score + 0.1), source_sentence=sent, source_page=page,
                extraction_method=method,
            ))

        for cand in materials:
            get_or_create_material(cand, score, page, section, sent, b_type)

    # ------------------------------------------------------------------
    # Tables (schema section 8): structured row/column relations, deterministic — no NLP needed.
    # ------------------------------------------------------------------
    @classmethod
    def _process_table_block(cls, block: Dict[str, Any], results: Dict[str, Any], get_or_create_material) -> None:
        page, section = str(block["page"]), block["section"]
        results["tables"].append(TableRecord(
            table_number=block.get("table_number", "NONE"), caption=block.get("caption", "NONE"),
            page=page, section=section, headers=block.get("headers", []), rows=block.get("rows", []),
            material_column=block.get("material_column", "NONE"), raw_text=block.get("text", "NONE"),
        ))

        material_column = block.get("material_column", "NONE")
        headers = [h for h in block.get("headers", []) if h != material_column]
        header_props = {h: cls._match_header_property(h) for h in headers}

        for row in block.get("rows", []):
            row_material = None
            if material_column != "NONE":
                cell_materials = ChemistryEngine.find_materials(row.get(material_column, ""))
                if cell_materials:
                    row_material = get_or_create_material(cell_materials[0], 0.85, page, section,
                                                            block.get("caption", "NONE"), "Table")
            for header, cell_value in row.items():
                if header == material_column or not cell_value:
                    continue
                prop_info = header_props.get(header)
                if not prop_info:
                    continue
                cat, prop = prop_info
                field = cls._extract_cell_value(cell_value, header, cat, prop, page, section, block)
                if field is None:
                    continue
                results[cat][prop].append(field)
                if row_material is not None:
                    row_material.properties.append(field)
                    results["relations"].append(Relation(
                        material_id=row_material.material_id, material_formula=row_material.normalized_formula,
                        category=cat, property_name=prop, property_value=field,
                        confidence=0.95, source_sentence=block.get("caption", "NONE"), source_page=page,
                        extraction_method="table_row",
                    ))

    @staticmethod
    def _match_header_property(header: str) -> Optional[Tuple[str, str]]:
        lower = header.lower()
        for keyword, cat, prop in _HEADER_PROPERTY_KEYWORDS:
            if keyword in lower:
                return cat, prop
        return None

    @staticmethod
    def _extract_cell_value(cell_value: str, header: str, cat: str, prop: str, page: str, section: str,
                             block: Dict[str, Any]) -> Optional[ExtractedField]:
        _, norm_type = SchemaConfig.SCHEMA[cat][prop]
        m = _CELL_NUM_UNIT.match(cell_value.strip())
        if not m:
            return None
        value_str, unit_str = m.group(1), (m.group(2) or "").strip()
        if not unit_str:
            header_unit_match = _HEADER_UNIT.search(header)
            unit_str = header_unit_match.group(1).strip() if header_unit_match else ""
        normalized = DataNormalizer.normalize((value_str, unit_str), norm_type)
        if normalized == "NONE":
            return None
        unit = normalized.split(" ", 1)[1] if " " in normalized else "NONE"
        return ExtractedField(
            raw_value=cell_value, normalized_value=normalized, confidence=0.9,
            source_page=page, source_section=section, source_sentence=block.get("caption", "NONE"),
            source_type="Table", unit=unit, category=cat, property_name=prop,
        )

    # ------------------------------------------------------------------
    # Figures (schema section 9)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_figure(block: Dict[str, Any]) -> FigureRecord:
        caption = block["text"]
        materials = [m["formula"] for m in ChemistryEngine.find_materials(caption)]
        properties = []
        for cat, props in SchemaConfig.COMPILED_SCHEMA.items():
            for prop, (pattern, _) in props.items():
                if pattern.search(caption):
                    properties.append(prop)
        techniques = SchemaConfig.COMPILED_SCHEMA["CHARACTERIZATION"]["Technique"][0].findall(caption)
        return FigureRecord(
            figure_number=block.get("ref_number", "NONE"), caption=caption,
            page=str(block["page"]), section=block["section"],
            referenced_materials=list(dict.fromkeys(materials)),
            referenced_properties=list(dict.fromkeys(properties)),
            referenced_tables=list(dict.fromkeys(_TABLE_REF_REGEX.findall(caption))),
            referenced_equations=list(dict.fromkeys(_EQ_REF_REGEX.findall(caption))),
            mentioned_methods=list(dict.fromkeys(techniques)),
        )

    # ------------------------------------------------------------------
    # Equations (schema section 10)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_variables(text: str) -> List[str]:
        out = []
        for tok in _VAR_TOKEN_REGEX.findall(text):
            if len(tok) <= 3 and tok.lower() not in _VAR_STOPWORDS and tok not in out:
                out.append(tok)
        return out[:8]

    @classmethod
    def _classify_equation(cls, text: str) -> str:
        for pattern, etype in _EQUATION_TYPE_PATTERNS:
            if pattern.search(text):
                return etype
        return "Unclassified"

    @classmethod
    def _record_equation(cls, results: Dict[str, Any], equation_text: str, context: str, page: str,
                          section: str, seen: set, force: bool = False) -> None:
        etype = cls._classify_equation(equation_text)
        if not force and etype == "Unclassified":
            return
        key = etype if etype != "Unclassified" else f"unclassified:{page}:{equation_text[:30]}"
        if key in seen:
            return
        seen.add(key)
        results["equations"].append(EquationRecord(
            equation_text=equation_text, equation_type=etype, page=page, section=section,
            variables=cls._extract_variables(equation_text), context_sentence=context,
        ))

    @classmethod
    def _scan_equation_mentions(cls, sent: str, page: str, section: str, seen: set, results: Dict[str, Any]) -> None:
        for pattern, etype in _EQUATION_TYPE_PATTERNS:
            if etype in seen:
                continue
            if pattern.search(sent):
                cls._record_equation(results, sent, sent, page, section, seen)
