# models.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedField:
    """A single extracted datum with full provenance.

    Kept as the one universal "value" type used across materials, properties,
    metadata, equations, etc. `unit`/`value_numeric`/`category`/`property_name`
    are optional enrichments (default "NONE"/None) so every existing positional
    call site (7 required args) keeps working unchanged.
    """
    raw_value: str
    normalized_value: str
    confidence: float
    source_page: str
    source_section: str
    source_sentence: str
    source_type: str  # Text, Table, Caption, Equation, Metadata
    unit: str = "NONE"
    value_numeric: Optional[float] = None
    category: str = "NONE"
    property_name: str = "NONE"

    @classmethod
    def empty(cls) -> "ExtractedField":
        return cls("NONE", "NONE", 0.0, "NONE", "NONE", "NONE", "NONE")

    def is_empty(self) -> bool:
        return self.raw_value == "NONE"

    def to_flat_dict(self, prefix: str) -> Dict[str, Any]:
        """Flattens this field into prefixed columns for the legacy flat summary export."""
        return {
            f"{prefix}_Raw": self.raw_value,
            f"{prefix}_Value": self.normalized_value,
            f"{prefix}_Unit": self.unit,
            f"{prefix}_Confidence": round(self.confidence, 3) if self.confidence else 0.0,
            f"{prefix}_Page": self.source_page,
            f"{prefix}_Section": self.source_section,
            f"{prefix}_SourceType": self.source_type,
        }


@dataclass
class Relation:
    """Links one material to one property value, with the method used to establish the link."""
    material_id: str
    material_formula: str
    category: str
    property_name: str
    property_value: ExtractedField
    confidence: float
    source_sentence: str
    source_page: str
    extraction_method: str  # dependency_parse | proximity | table_row


@dataclass
class Material:
    """A material/composition found in a paper, with its own attached properties (schema section 16)."""
    material_id: str
    raw_formula: str
    normalized_formula: str
    elements: Dict[str, float]
    is_known_alias: bool
    confidence: float
    source_page: str
    source_section: str
    source_sentence: str
    source_type: str
    properties: List[ExtractedField] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "Material":
        return cls("NONE", "NONE", "NONE", {}, False, 0.0, "NONE", "NONE", "NONE", "NONE")


@dataclass
class TableRecord:
    table_number: str
    caption: str
    page: str
    section: str
    headers: List[str] = field(default_factory=list)
    rows: List[Dict[str, str]] = field(default_factory=list)
    material_column: str = "NONE"
    raw_text: str = "NONE"


@dataclass
class FigureRecord:
    figure_number: str
    caption: str
    page: str
    section: str
    referenced_materials: List[str] = field(default_factory=list)
    referenced_properties: List[str] = field(default_factory=list)
    referenced_tables: List[str] = field(default_factory=list)
    referenced_equations: List[str] = field(default_factory=list)
    mentioned_methods: List[str] = field(default_factory=list)


@dataclass
class EquationRecord:
    equation_text: str
    equation_type: str
    page: str
    section: str
    variables: List[str] = field(default_factory=list)
    context_sentence: str = "NONE"


@dataclass
class WorkflowStep:
    order: int
    stage: str
    evidence_sentence: str
    page: str
    section: str


@dataclass
class PaperData:
    filename: str
    metadata: Dict[str, ExtractedField]

    # Per-material extraction (schema section 16): each material carries its own properties.
    materials: List[Material] = field(default_factory=list)

    # Paper-level (not tied to one material) categorized candidate lists — DFT/AIMD/ML/synthesis
    # settings usually describe the whole computational/experimental study, not a single material.
    electrochemical: Dict[str, List[ExtractedField]] = field(default_factory=dict)
    thermodynamic: Dict[str, List[ExtractedField]] = field(default_factory=dict)
    mechanical: Dict[str, List[ExtractedField]] = field(default_factory=dict)
    structural: Dict[str, List[ExtractedField]] = field(default_factory=dict)
    electronic: Dict[str, List[ExtractedField]] = field(default_factory=dict)
    thermal: Dict[str, List[ExtractedField]] = field(default_factory=dict)
    spectroscopy: Dict[str, List[ExtractedField]] = field(default_factory=dict)
    dft: Dict[str, List[ExtractedField]] = field(default_factory=dict)
    aimd: Dict[str, List[ExtractedField]] = field(default_factory=dict)
    ml: Dict[str, List[ExtractedField]] = field(default_factory=dict)
    synthesis: Dict[str, List[ExtractedField]] = field(default_factory=dict)
    characterization: Dict[str, List[ExtractedField]] = field(default_factory=dict)
    battery_perf: Dict[str, List[ExtractedField]] = field(default_factory=dict)
    interfaces: Dict[str, List[ExtractedField]] = field(default_factory=dict)

    relations: List[Relation] = field(default_factory=list)
    workflow: List[WorkflowStep] = field(default_factory=list)
    figures: List[FigureRecord] = field(default_factory=list)
    tables: List[TableRecord] = field(default_factory=list)
    equations: List[EquationRecord] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
