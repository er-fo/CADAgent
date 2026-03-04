"""
Thread specification database for tapped-hole feature operations.

Provides lookup helpers for common ISO metric, UNC, and UNF threads so the
Fusion add-in can translate high-level requests into concrete thread settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Conversion helpers ---------------------------------------------------------

_MM_PER_UNIT = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
}


def _inch_to_mm(value: float) -> float:
    return round(value * _MM_PER_UNIT["in"], 4)


def _round(value: float, digits: int = 4) -> float:
    return round(value, digits)


@dataclass(frozen=True)
class ThreadSpecification:
    """Thread specification with all parameters needed for Fusion API."""

    thread_type: str  # ISO, UNC, UNF
    designation: str  # e.g., "M6x1.0", "1/4-20 UNC"
    nominal_diameter: float  # mm
    pitch: float  # mm (metric pitch or TPI converted to mm)
    tap_drill_diameter: float  # mm
    minor_diameter: float  # mm
    major_diameter: float  # mm
    is_metric: bool
    fusion_thread_type: str
    fusion_size: str
    fusion_designation: str

    def min_thread_depth_mm(self) -> float:
        """Recommended minimum thread depth (1.5 × diameter)."""
        return self.nominal_diameter * 1.5

    def recommended_thread_depth_mm(self) -> float:
        """Recommended depth for full strength (2.5 × diameter)."""
        return self.nominal_diameter * 2.5


def _metric_threads() -> Dict[str, ThreadSpecification]:
    specs: Dict[str, ThreadSpecification] = {}
    metric_definitions = [
        ("M3", 0.5),
        ("M4", 0.7),
        ("M5", 0.8),
        ("M6", 1.0),
        ("M8", 1.25),
        ("M10", 1.5),
        ("M12", 1.75),
        ("M16", 2.0),
        ("M20", 2.5),
    ]

    for size, pitch in metric_definitions:
        nominal = float(size[1:])
        major = nominal
        # Format pitch with one decimal place for Fusion compatibility (e.g., "M6x1.0")
        pitch_text = f"{pitch:.1f}"
        designation = f"{size}x{pitch_text}"
        tap_drill = _round(major - pitch, 3)
        minor = _round(major - (1.0825 * pitch), 3)
        specs[size] = ThreadSpecification(
            thread_type="ISO",
            designation=designation,
            nominal_diameter=nominal,
            pitch=pitch,
            tap_drill_diameter=tap_drill,
            minor_diameter=minor,
            major_diameter=major,
            is_metric=True,
            fusion_thread_type="ISO Metric profile",
            fusion_size=designation,
            fusion_designation=designation,
        )
    return specs


def _imperial_threads() -> Tuple[Dict[str, ThreadSpecification], Dict[str, ThreadSpecification]]:
    unc_specs: Dict[str, ThreadSpecification] = {}
    unf_specs: Dict[str, ThreadSpecification] = {}

    unc_definitions = [
        ("#6-32", 0.1380, 32, 0.1065, "#6"),
        ("#8-32", 0.1640, 32, 0.1360, "#8"),
        ("#10-24", 0.1900, 24, 0.1495, "#10"),
        ("1/4-20", 0.2500, 20, 0.2010, "1/4"),
        ("5/16-18", 0.3125, 18, 0.2570, "5/16"),
        ("3/8-16", 0.3750, 16, 0.3125, "3/8"),
        ("1/2-13", 0.5000, 13, 0.4219, "1/2"),
    ]

    unf_definitions = [
        ("#6-40", 0.1380, 40, 0.1130, "#6"),
        ("#8-36", 0.1640, 36, 0.1405, "#8"),
        ("#10-32", 0.1900, 32, 0.1590, "#10"),
        ("1/4-28", 0.2500, 28, 0.2130, "1/4"),
        ("5/16-24", 0.3125, 24, 0.2720, "5/16"),
        ("3/8-24", 0.3750, 24, 0.3320, "3/8"),
        ("1/2-20", 0.5000, 20, 0.4531, "1/2"),
    ]

    def build_spec(
        table: Dict[str, ThreadSpecification],
        definition: Tuple[str, float, int, float, str],
        unified_type: str,
    ) -> None:
        size, major_in, tpi, tap_in, _bare_size = definition
        pitch_mm = _round(_MM_PER_UNIT["in"] / tpi, 4)
        major_mm = _round(_inch_to_mm(major_in), 4)
        tap_mm = _round(_inch_to_mm(tap_in), 4)
        minor_mm = _round(major_mm - (1.0825 * pitch_mm), 4)
        # Full designation includes type suffix for display/documentation
        designation = f"{size} {unified_type}"
        # Fusion expects just the size (e.g., "1/4-20") not "1/4-20 UNC"
        # The thread type is already specified separately
        fusion_designation = size
        table[size.upper()] = ThreadSpecification(
            thread_type=unified_type,
            designation=designation,
            nominal_diameter=major_mm,
            pitch=pitch_mm,
            tap_drill_diameter=tap_mm,
            minor_diameter=minor_mm,
            major_diameter=major_mm,
            is_metric=False,
            fusion_thread_type="Unified Screw Threads",
            fusion_size=size,
            fusion_designation=fusion_designation,
        )

    for definition in unc_definitions:
        build_spec(unc_specs, definition, "UNC")
    for definition in unf_definitions:
        build_spec(unf_specs, definition, "UNF")

    return unc_specs, unf_specs


METRIC_THREADS = _metric_threads()
UNC_THREADS, UNF_THREADS = _imperial_threads()

THREAD_TABLES: Dict[str, Dict[str, ThreadSpecification]] = {
    "metric": METRIC_THREADS,
    "unc": UNC_THREADS,
    "unf": UNF_THREADS,
}

_THREAD_TYPE_ALIASES = {
    "metric": "metric",
    "iso": "metric",
    "isometric": "metric",
    "iso-metric": "metric",
    "iso_metric": "metric",
    "unc": "unc",
    "unifiedcoarse": "unc",
    "unifiednationalcoarse": "unc",
    "unf": "unf",
    "unifiedfine": "unf",
    "unifiednationalfine": "unf",
}


def _normalize_thread_type(thread_type: str) -> Optional[str]:
    if not thread_type:
        return None
    key = "".join(ch for ch in thread_type.lower() if not ch.isspace()).replace("_", "").replace("-", "")
    return _THREAD_TYPE_ALIASES.get(key)


def _normalize_size(thread_type: str, size: str) -> str:
    token = size.strip().upper()
    token = token.replace(" ", "")
    token = token.replace("UNC", "").replace("UNF", "")

    if thread_type == "metric":
        if "X" in token:
            token = token.split("X", 1)[0]
        if not token.startswith("M"):
            token = f"M{token}"
        return token

    # Imperial (UNC/UNF)
    return token


def lookup_thread(thread_type: str, size: str) -> Optional[ThreadSpecification]:
    """
    Look up thread specification by type and size.
    """
    normalized_type = _normalize_thread_type(thread_type or "")
    if not normalized_type:
        return None

    if not size or not size.strip():
        return None

    normalized_size = _normalize_size(normalized_type, size)
    table = THREAD_TABLES.get(normalized_type, {})
    spec = table.get(normalized_size)
    if spec:
        return spec

    # Fallback: direct designation match
    target = size.strip().upper()
    for candidate in table.values():
        if candidate.designation.upper() == target or candidate.fusion_designation.upper() == target:
            return candidate

    return None


def get_tap_drill_diameter(thread_spec: ThreadSpecification) -> float:
    """Return recommended tap drill diameter (mm)."""
    return thread_spec.tap_drill_diameter


def list_available_threads(thread_type: Optional[str] = None) -> List[str]:
    """List thread designations filtered by optional type."""
    if thread_type:
        normalized_type = _normalize_thread_type(thread_type)
        if not normalized_type:
            return []
        return sorted(spec.designation for spec in THREAD_TABLES[normalized_type].values())

    all_specs: List[str] = []
    for table in THREAD_TABLES.values():
        all_specs.extend(spec.designation for spec in table.values())
    return sorted(all_specs)


def parse_thread_designation(designation: str) -> Tuple[str, str]:
    """
    Parse thread designation into normalized type and size token.
    """
    if not designation or not designation.strip():
        raise ValueError("Thread designation cannot be empty.")

    text = designation.strip()
    lowered = text.lower()

    if lowered.startswith("m"):
        size_token = text.upper().split()[0]
        if "X" in size_token:
            size_token = size_token.split("X", 1)[0]
        return "metric", size_token

    if "unf" in lowered:
        cleaned = text.upper().replace("UNF", "").strip()
        cleaned = cleaned.replace(" ", "")
        return "unf", cleaned

    if "unc" in lowered:
        cleaned = text.upper().replace("UNC", "").strip()
        cleaned = cleaned.replace(" ", "")
        return "unc", cleaned

    normalized = text.upper().replace(" ", "")
    if normalized.startswith("M"):
        size_token = normalized.split("X", 1)[0]
        return "metric", size_token

    for thread_type, table in THREAD_TABLES.items():
        for key in table.keys():
            if normalized == key.upper().replace(" ", ""):
                return thread_type, key

    raise ValueError(f"Unrecognized thread designation '{designation}'.")


def validate_thread_depth(thread_spec: ThreadSpecification, depth_cm: float) -> Tuple[bool, str]:
    """
    Validate thread depth relative to nominal diameter.

    Returns:
        Tuple[bool, str]: (is_within_minimum, diagnostic message)
    """
    if depth_cm <= 0:
        return False, "Thread depth must be positive."

    depth_mm = depth_cm * _MM_PER_UNIT["cm"]
    min_mm = thread_spec.min_thread_depth_mm()
    recommended_mm = thread_spec.recommended_thread_depth_mm()
    max_reasonable_mm = thread_spec.nominal_diameter * 5.0

    if depth_mm < min_mm:
        return False, (
            f"Thread depth {depth_cm:.2f} cm is below minimum engagement "
            f"({min_mm / _MM_PER_UNIT['cm']:.2f} cm for {thread_spec.designation})."
        )

    if depth_mm < recommended_mm:
        return True, (
            f"Thread depth {depth_cm:.2f} cm is below the recommended "
            f"{recommended_mm / _MM_PER_UNIT['cm']:.2f} cm but above minimum."
        )

    if depth_mm > max_reasonable_mm:
        return True, (
            f"Thread depth {depth_cm:.2f} cm exceeds 5× diameter "
            f"({max_reasonable_mm / _MM_PER_UNIT['cm']:.2f} cm); verify necessity."
        )

    return True, "Thread depth is within recommended range."


def suggest_thread_for_bolt(bolt_designation: str) -> Optional[ThreadSpecification]:
    """
    Suggest tapped hole thread for a given bolt designation.
    """
    if not bolt_designation or not bolt_designation.strip():
        return None

    try:
        thread_type, size = parse_thread_designation(bolt_designation)
        return lookup_thread(thread_type, size)
    except ValueError:
        normalized = bolt_designation.strip().upper()
        for table in THREAD_TABLES.values():
            if normalized in table:
                return table[normalized]
    return None


__all__ = [
    "ThreadSpecification",
    "lookup_thread",
    "get_tap_drill_diameter",
    "list_available_threads",
    "parse_thread_designation",
    "validate_thread_depth",
    "suggest_thread_for_bolt",
    "METRIC_THREADS",
    "UNC_THREADS",
    "UNF_THREADS",
]
