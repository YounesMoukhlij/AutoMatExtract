# normalizer.py

import re
import unicodedata
from typing import Optional, Tuple, Union

import pint

# ---------------------------------------------------------------------------
# Shared scientific-text normalization (schema sections 13 & 14). Used by
# parser.py, ocr.py, chemistry.py and this module so unicode minus signs,
# Greek letters, and compatibility glyphs are handled in exactly one place.
# ---------------------------------------------------------------------------
_UNICODE_MINUS = '−'  # −
_EN_DASH = '–'        # –
_EM_DASH = '—'        # —


def normalize_scientific_text(text: str) -> str:
    """NFKC-normalizes text while preserving Greek letters, sub/superscripts, and symbols."""
    if not text:
        return text
    return unicodedata.normalize("NFKC", text)


def _parse_scientific_number(raw: str) -> Optional[float]:
    """Parses '1.2', '1.2e-3', '1.2x10^-3', '1.2×10−3' (incl. unicode minus) into a float."""
    if raw is None:
        return None
    s = str(raw).strip().replace(_UNICODE_MINUS, '-').replace(_EN_DASH, '-').replace(_EM_DASH, '-')
    s = s.replace(' ', '')
    m = re.match(r'^([+-]?\d*\.?\d+)(?:[eE]([+-]?\d+)|[xX×\*]10\^?([+-]?\d+))?$', s)
    if not m:
        return None
    try:
        coeff = float(m.group(1))
    except ValueError:
        return None
    exponent = m.group(2) or m.group(3)
    if exponent:
        coeff *= 10.0 ** int(exponent)
    return coeff


# ---------------------------------------------------------------------------
# Unit engine (schema section 14). Built once at import time and shared
# read-only across threads.
# ---------------------------------------------------------------------------
_ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
# pint's default registry already ships 'rydberg'/'Ry', 'hartree', 'Ah' and 'Wh' — only the
# common short alias 'Ha' for hartree is missing, so that's the only addition needed.
_ureg.define('@alias hartree = Ha')
Q_ = _ureg.Quantity

# Exact captured-unit-text -> pint-parseable unit string. Kept as an explicit lookup (rather
# than generic regex munging) since every possible unit token is enumerated in config.py's
# SCHEMA patterns and controlled by this same codebase.
UNIT_ALIASES = {
    'S/cm': 'S/cm', 'mS/cm': 'mS/cm', 'μS/cm': 'microsiemens/cm', 'uS/cm': 'microsiemens/cm',
    'cm2/s': 'cm**2/s', 'cm^2/s': 'cm**2/s', 'cm2/?s': 'cm**2/s', 'm2/s': 'm**2/s', 'm^2/s': 'm**2/s',
    'eV': 'eV', 'meV': 'meV', 'eV/atom': 'eV', 'meV/atom': 'meV', 'eV/f.u.': 'eV',
    'Ry': 'Ry', 'Ha': 'Ha', 'kJ/mol': 'kJ/mol', 'kcal/mol': 'kcal/mol',
    'V': 'V', 'mV': 'mV',
    'mAh/g': 'mAh/g', 'Ah/kg': 'Ah/kg', 'mAh/cm2': 'mAh/cm**2',
    'Wh/kg': 'Wh/kg', 'Wh/L': 'Wh/L', 'kWh/kg': 'kWh/kg',
    'W/kg': 'W/kg', 'kW/kg': 'kW/kg',
    'mA/cm2': 'mA/cm**2', 'A/cm2': 'A/cm**2', 'mA/g': 'mA/g', 'A/g': 'A/g',
    'GPa': 'GPa', 'MPa': 'MPa', 'bar': 'bar', 'atm': 'atm',
    'Å': 'angstrom', 'Angstrom': 'angstrom', 'nm': 'nm', 'pm': 'pm',
    'Å3': 'angstrom**3', 'Å^3': 'angstrom**3', 'nm3': 'nm**3',
    'g/cm3': 'g/cm**3', 'kg/m3': 'kg/m**3', 'g/cc': 'g/cm**3',
    'K': 'kelvin', '°C': 'degC', 'C': 'degC',
    'ps': 'ps', 'ns': 'ns', 'fs': 'fs',
    'h': 'hour', 'hours': 'hour', 'min': 'minute', 'minutes': 'minute',
    'cm-1': 'cm**-1', 'cm^-1': 'cm**-1',
    '°': 'degree', 'deg': 'degree', 'degrees': 'degree',
    'W/mK': 'W/m/K', 'W/m/K': 'W/m/K', 'W/(m·K)': 'W/m/K',
    'ohm cm2': 'ohm*cm**2', 'Ω cm2': 'ohm*cm**2', 'Ω⋅cm2': 'ohm*cm**2', 'ohm*cm2': 'ohm*cm**2',
    'J/m2': 'J/m**2', 'eV/Å2': 'eV/angstrom**2', 'eV/A2': 'eV/angstrom**2',
    '%': 'percent', 'percent': 'percent',
}

# Canonical per-category output unit (a valid pint unit expression).
CANONICAL_UNITS = {
    'energy': 'eV', 'energy_large': 'eV', 'conductivity': 'S/cm', 'diffusion': 'cm**2/s', 'voltage': 'V',
    'capacity': 'mAh/g', 'energy_density': 'Wh/kg', 'power_density': 'W/kg',
    'current_density': 'mA/cm**2', 'area_resistance': 'ohm*cm**2', 'pressure': 'GPa',
    'length': 'angstrom', 'volume': 'angstrom**3', 'density': 'g/cm**3',
    'temperature': 'kelvin', 'time': 'ps', 'wavenumber': 'cm**-1', 'angle': 'degree',
    'thermal_conductivity': 'W/m/K', 'surface_energy': 'J/m**2',
}

# (min, max) physical-validation bounds, expressed in the category's canonical unit.
BOUNDS = {
    # 'energy' covers physically small quantities (activation energy, band gap, formation
    # energy, ...); 'energy_large' covers DFT numerical-parameter energies (plane-wave cutoffs,
    # SCF thresholds), which routinely run into the hundreds of eV — a genuinely different scale
    # that would otherwise get wrongly rejected by the tight 'energy' bound.
    'energy': (-10.0, 10.0), 'energy_large': (0.0, 5000.0),
    'conductivity': (1e-12, 100.0), 'diffusion': (1e-15, 1e-3),
    'voltage': (0.0, 10.0), 'capacity': (0.0, 5000.0), 'energy_density': (0.0, 2000.0),
    'power_density': (0.0, 100000.0), 'current_density': (0.0, 10000.0),
    'area_resistance': (0.0, 1e7), 'pressure': (-50.0, 1000.0), 'length': (0.0, 1e7),
    'volume': (0.0, 1e7), 'density': (0.0, 30.0), 'temperature': (0.0, 4000.0),
    'time': (0.0, 1e9), 'wavenumber': (0.0, 5000.0), 'angle': (0.0, 180.0),
    'thermal_conductivity': (0.0, 3000.0), 'surface_energy': (0.0, 20.0),
    'percent': (0.0, 150.0), 'count': (0.0, 1e9), 'dimensionless': (-1e6, 1e6),
}


class DataNormalizer:
    """Generic pint-backed unit normalization + physical validation.

    `normalize(raw_match, norm_type)` keeps the exact call signature already used across the
    codebase (extractor passes `match.groups()` or `match.group()`); `norm_type` now names a
    physical category resolved dynamically through pint rather than a hand-written per-type
    branch, so adding a new unit is a one-line addition to UNIT_ALIASES, not a new code path.
    """

    @classmethod
    def normalize(cls, raw_match: Union[Tuple, str], norm_type: str) -> str:
        if not raw_match:
            return "NONE"

        if norm_type == 'string':
            value = raw_match[0] if isinstance(raw_match, tuple) else raw_match
            value = normalize_scientific_text(str(value)).strip()
            return value if value else "NONE"

        if not isinstance(raw_match, tuple) or len(raw_match) < 1:
            return "NONE"

        value_str = raw_match[0]
        unit_str = raw_match[1] if len(raw_match) > 1 and raw_match[1] else ""
        num = _parse_scientific_number(value_str)
        if num is None:
            return "NONE"

        if norm_type in ('count', 'dimensionless'):
            lo, hi = BOUNDS[norm_type]
            if not (lo <= num <= hi):
                return "NONE"
            return f"{num:g}"

        if norm_type == 'percent':
            lo, hi = BOUNDS['percent']
            if not (lo <= num <= hi):
                return "NONE"
            return f"{num:.2f} %"

        if norm_type not in CANONICAL_UNITS:
            return "NONE"

        # NFKC-normalization turns superscript minus signs (cm⁻¹) into the unicode minus U+2212,
        # not an ASCII hyphen — but every UNIT_ALIASES key uses a plain hyphen ("cm-1"). Fold it
        # back so lookups still hit.
        unit_str = normalize_scientific_text(unit_str).strip().replace(_UNICODE_MINUS, '-')
        pint_unit = UNIT_ALIASES.get(unit_str, unit_str)
        if not pint_unit:
            return "NONE"

        target = CANONICAL_UNITS[norm_type]
        try:
            qty = Q_(num, pint_unit)
        except Exception:
            return "NONE"

        try:
            converted = qty.to(target)
            out_val, out_unit = converted.magnitude, target
            lo, hi = BOUNDS[norm_type]
            if not (lo <= out_val <= hi):
                return "NONE"
        except pint.errors.DimensionalityError:
            # Same category but incompatible dimension (e.g. areal vs. gravimetric capacity) —
            # keep the pint-validated original unit instead of silently dropping the value.
            if not (num == num and abs(num) < float('inf')):  # NaN/inf guard
                return "NONE"
            out_val, out_unit = qty.magnitude, pint_unit
        except Exception:
            return "NONE"

        return f"{out_val:.4g} {out_unit}"

    @staticmethod
    def parse_number(raw: str) -> Optional[float]:
        """Public helper for other modules (e.g. chemistry.py) needing plain numeric parsing."""
        return _parse_scientific_number(raw)
