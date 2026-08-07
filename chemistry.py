# chemistry.py
"""Chemistry-aware material detection (schema section 2).

Generic across any battery chemistry (Li/Na/K/Mg/Ca/Zn/Al/H, solid-state or liquid
electrolytes, cathodes/anodes/coatings/separators/...). Combines:
  1. A formula tokenizer regex that understands decimal/non-stoichiometric subscripts,
     parenthesized groups, hyphen/en-dash-joined solid solutions ("Li2S-P2S5"), and
     middot-joined hydrates ("Li3InCl6·2H2O").
  2. pymatgen.Composition validation of every candidate token.
  3. A curated alias table for common battery-material shorthand that isn't a literal
     chemical formula (LFP, LCO, NMC811, graphite, ...).
  4. An acronym blacklist to reject valid-looking-but-wrong tokens (DFT, XRD, PBE, CV, ...).
"""

import re
from typing import Any, Dict, List, Optional

from pymatgen.core import Composition
from pymatgen.core.composition import CompositionError
from pymatgen.core.periodic_table import DummySpecies

from normalizer import normalize_scientific_text

_ELEMENT = r'[A-Z][a-z]?'
# Subscript: integer/decimal count, optionally with a non-stoichiometric "−x"/"+y" suffix
# (e.g. "0.3", "7", "7−x"). Deliberately only the unicode minus (U+2212) triggers this, never
# a plain ASCII hyphen/en-dash — those are reserved for the solid-solution joiner below, and
# without that split "Li7-xLa3Zr2O12" (ASCII hyphen) would be ambiguous with a joined formula.
_SUBSCRIPT = r'(?:\d+(?:\.\d+)?(?:\s?[−+]\s?[a-z])?)?'
_PAREN_GROUP = rf'\((?:{_ELEMENT}{_SUBSCRIPT})+\){_SUBSCRIPT}'
_UNIT = rf'(?:{_PAREN_GROUP}|{_ELEMENT}{_SUBSCRIPT})'
_RUN = rf'(?:{_UNIT}){{2,}}'

# A material formula token: at least two element/paren units, optionally hyphen-joined
# (solid solution, e.g. "Li2S-P2S5") and/or middot-joined (hydrate, e.g. "Li3InCl6·2H2O").
# Built as ONE regex (rather than found-then-merged) so a hydrate's leading digit ("·2H2O")
# doesn't sit on a \w/\w boundary that would block a second \b-anchored match from ever starting.
TOKEN_REGEX = re.compile(rf'\b{_RUN}(?:[-–]{_RUN})?(?:·\d*{_RUN})?')

_NONSTOICH_SUFFIX = re.compile(r'(\d)\s?[−+]\s?[a-z](?![a-z])')

# Common shorthand that is not itself a parseable chemical formula string.
ALIASES: Dict[str, str] = {
    "lfp": "LiFePO4", "lithium iron phosphate": "LiFePO4",
    "lco": "LiCoO2", "lithium cobalt oxide": "LiCoO2",
    "lmo": "LiMn2O4", "lithium manganese oxide": "LiMn2O4",
    "nca": "LiNi0.8Co0.15Al0.05O2",
    "llzo": "Li7La3Zr2O12",
    "lgps": "Li10GeP2S12",
    "lpscl": "Li6PS5Cl", "lpsbr": "Li6PS5Br", "lpsi": "Li6PS5I",
    "lto": "Li4Ti5O12",
    "lisicon": "Li14ZnGe4O16",
    "nasicon": "Na3Zr2Si2PO12",
    "na3ps4": "Na3PS4",
    "graphite": "C", "hard carbon": "C", "soft carbon": "C",
    "silicon": "Si", "lithium metal": "Li", "sodium metal": "Na",
}
_ALIAS_REGEX = re.compile(
    r'\b(' + '|'.join(sorted((re.escape(k) for k in ALIASES), key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)
_NMC_REGEX = re.compile(r'(?i)\bNMC\s?-?(\d)(\d)(\d)\b')

# Valid-looking element-symbol strings that are actually acronyms/units/methods, not materials.
ACRONYM_BLACKLIST = {
    "DFT", "XRD", "SEM", "TEM", "EDS", "EDX", "XPS", "FTIR", "EIS", "GITT", "CV", "PBE", "PBE0",
    "GGA", "LDA", "SCAN", "HSE", "NVT", "NPT", "NVE", "SOC", "PAW", "NEB", "VASP", "MSD", "RMSE",
    "MAE", "ROC", "AUC", "SHAP", "LIME", "CIF", "ICSD", "DOS", "PDOS", "COHP", "ESI", "ESW",
    "BET", "TGA", "DSC", "SAXS", "NMR", "SVM", "GNN", "PLD", "CVD", "SI", "IR", "UV", "AI", "ML",
    "NO", "IN", "AS", "IS", "AT", "OF", "OR", "ON", "BY", "TO", "PO", "SE", "BE", "PC", "CO",
}


def _clean_for_validation(raw: str) -> str:
    """Strips non-stoichiometric "-x"/"+y" subscript suffixes so pymatgen can parse the rest."""
    return _NONSTOICH_SUFFIX.sub(r'\1', raw)


def _validate(formula: str) -> Optional[Composition]:
    formula = formula.strip('.,;:)( ')
    if not formula or formula.upper() in ACRONYM_BLACKLIST:
        return None
    try:
        comp = Composition(_clean_for_validation(formula))
        if len(comp) == 0:
            return None
        # pymatgen silently accepts unrecognized tokens (e.g. "L", "M", "A") as placeholder
        # DummySpecies rather than raising — that would let acronym fragments like "NCA"/"LFP"
        # masquerade as valid compositions. Real materials must be built entirely of real elements.
        if any(isinstance(el, DummySpecies) for el in comp.elements):
            return None
        return comp
    except (CompositionError, ValueError, KeyError):
        return None


class ChemistryEngine:
    """Generic, chemistry-agnostic material finder used by extractor.py."""

    @staticmethod
    def find_materials(text: str) -> List[Dict[str, Any]]:
        text = normalize_scientific_text(text)
        results: List[Dict[str, Any]] = []
        claimed_spans: List[tuple] = []

        # 1. Raw formula tokens (hyphen/hydrate joins are already part of TOKEN_REGEX itself).
        for m in TOKEN_REGEX.finditer(text):
            material = ChemistryEngine._resolve_token(m.group())
            if material:
                material["start"], material["end"] = m.start(), m.end()
                results.append(material)
                claimed_spans.append((m.start(), m.end()))

        # 2. Curated aliases (LFP, NMC811, graphite, ...).
        for m in _ALIAS_REGEX.finditer(text):
            if any(s <= m.start() < e for s, e in claimed_spans):
                continue
            key = m.group(1).lower()
            canonical = ALIASES.get(key)
            comp = _validate(canonical) if canonical else None
            results.append({
                "raw": m.group(1),
                "formula": comp.reduced_formula if comp else canonical or m.group(1),
                "elements": dict(comp.get_el_amt_dict()) if comp else {},
                "is_known_alias": True,
                "start": m.start(),
                "end": m.end(),
            })

        # 3. Generic NMCxyz stoichiometry (NMC811 -> LiNi0.8Mn0.1Co0.1O2, etc.).
        for m in _NMC_REGEX.finditer(text):
            if any(s <= m.start() < e for s, e in claimed_spans):
                continue
            ni, mn, co = (int(d) / 10 for d in m.groups())
            formula = f"LiNi{ni:g}Mn{mn:g}Co{co:g}O2"
            comp = _validate(formula)
            results.append({
                "raw": m.group(0),
                "formula": comp.reduced_formula if comp else formula,
                "elements": dict(comp.get_el_amt_dict()) if comp else {},
                "is_known_alias": True,
                "start": m.start(),
                "end": m.end(),
            })

        return results

    @staticmethod
    def _resolve_token(raw: str) -> Optional[Dict[str, Any]]:
        raw = raw.strip('.,;:)( ')
        if len(raw) < 2:
            return None

        joiner_kind = "hydrate" if "·" in raw else ("solid_solution" if re.search(r'[-–]', raw) else None)

        if joiner_kind == "solid_solution":
            parts = re.split(r'[-–]', raw)
            comps = [_validate(p) for p in parts]
            if not all(comps):
                return None
            elements: Dict[str, float] = {}
            for c in comps:
                for el, amt in c.get_el_amt_dict().items():
                    elements[el] = elements.get(el, 0.0) + amt
            return {
                "raw": raw,
                "formula": "-".join(c.reduced_formula for c in comps),
                "elements": elements,
                "is_known_alias": False,
            }

        if joiner_kind == "hydrate":
            anhydrous_raw, _, hydrate_raw = raw.partition("·")
            hydrate_count = re.match(r'^(\d*)', hydrate_raw).group(1)
            hydrate_formula = hydrate_raw[len(hydrate_count):]
            comp1, comp2 = _validate(anhydrous_raw), _validate(hydrate_formula)
            if not comp1 or not comp2:
                return None
            elements = dict(comp1.get_el_amt_dict())
            mult = int(hydrate_count) if hydrate_count.isdigit() else 1
            for el, amt in comp2.get_el_amt_dict().items():
                elements[el] = elements.get(el, 0.0) + amt * mult
            return {
                "raw": raw,
                "formula": f"{comp1.reduced_formula}·{hydrate_count}{comp2.reduced_formula}",
                "elements": elements,
                "is_known_alias": False,
            }

        comp = _validate(raw)
        if not comp:
            return None
        # A single-element "formula" that's just a common short English word risks false
        # positives (e.g. "As", "In"); the acronym blacklist already screens the worst offenders.
        return {
            "raw": raw,
            "formula": comp.reduced_formula,
            "elements": dict(comp.get_el_amt_dict()),
            "is_known_alias": False,
        }
