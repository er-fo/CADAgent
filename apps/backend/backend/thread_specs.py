"""Centralized thread specification catalogs and normalization helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Tuple

THREAD_SIZES_BY_TYPE: Dict[str, Tuple[str, ...]] = {
    "metric": ("M3", "M4", "M5", "M6", "M8", "M10", "M12", "M16", "M20"),
    "unc": ("#6-32", "#8-32", "#10-24", "1/4-20", "5/16-18", "3/8-16", "1/2-13"),
    "unf": ("#6-40", "#8-36", "#10-32", "1/4-28", "5/16-24", "3/8-24", "1/2-20"),
}

ALL_THREAD_SIZES: Tuple[str, ...] = tuple(
    size
    for thread_type in ("metric", "unc", "unf")
    for size in THREAD_SIZES_BY_TYPE[thread_type]
)

_THREAD_TYPE_ALIAS_KEYS: Dict[str, str] = {
    "metric": "metric",
    "iso": "metric",
    "isometric": "metric",
    "m": "metric",
    "unc": "unc",
    "unifiedcoarse": "unc",
    "unf": "unf",
    "unifiedfine": "unf",
}


def _clean_alias_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def normalize_thread_type(value: Any) -> Optional[str]:
    """Normalize a thread type to one of: metric, unc, unf."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _THREAD_TYPE_ALIAS_KEYS.get(_clean_alias_key(text))


def _format_metric_nominal(raw: str) -> str:
    if "." in raw:
        stripped = raw.rstrip("0").rstrip(".")
        return stripped or "0"
    return str(int(raw))


def normalize_thread_size(value: Any) -> Optional[str]:
    """Normalize thread size formatting to canonical catalog representation."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    squashed = re.sub(r"\s+", "", text)

    metric_match = re.fullmatch(r"[mM](\d+(?:\.\d+)?)(?:[xX]\d+(?:\.\d+)?)?", squashed)
    if metric_match:
        return f"M{_format_metric_nominal(metric_match.group(1))}"

    number_match = re.fullmatch(r"#?(6|8|10)-(\d+)", squashed)
    if number_match:
        return f"#{number_match.group(1)}-{int(number_match.group(2))}"

    fraction_match = re.fullmatch(r"(\d+)/(\d+)-(\d+)", squashed)
    if fraction_match:
        numerator = int(fraction_match.group(1))
        denominator = int(fraction_match.group(2))
        tpi = int(fraction_match.group(3))
        return f"{numerator}/{denominator}-{tpi}"

    lowered = squashed.lower()
    for candidate in ALL_THREAD_SIZES:
        if lowered == candidate.lower():
            return candidate
    return text.upper() if squashed.lower().startswith("m") else squashed


def thread_type_for_size(size: Any) -> Optional[str]:
    """Return canonical thread type for a canonicalized size."""
    normalized_size = normalize_thread_size(size)
    if not normalized_size:
        return None
    for thread_type, sizes in THREAD_SIZES_BY_TYPE.items():
        if normalized_size in sizes:
            return thread_type
    return None


def validate_thread_spec(thread_type: Any, thread_size: Any) -> Tuple[str, str]:
    """
    Validate and canonicalize a thread specification.

    Returns:
        (canonical_thread_type, canonical_thread_size)
    Raises:
        ValueError if the pair is not supported by the runtime catalog.
    """
    canonical_type = normalize_thread_type(thread_type)
    if not canonical_type:
        raise ValueError(
            "thread_type must be one of metric, unc, unf (aliases like 'iso' normalize to 'metric')."
        )

    canonical_size = normalize_thread_size(thread_size)
    if not canonical_size:
        raise ValueError("thread_size cannot be empty.")

    owning_type = thread_type_for_size(canonical_size)
    if owning_type is None:
        raise ValueError(
            f"thread_size '{canonical_size}' is not supported. "
            f"Allowed sizes: {format_catalog_inline()}."
        )
    if owning_type != canonical_type:
        raise ValueError(
            f"thread_size '{canonical_size}' belongs to '{owning_type}', not '{canonical_type}'. "
            f"Allowed {canonical_type} sizes: {', '.join(THREAD_SIZES_BY_TYPE[canonical_type])}."
        )
    return canonical_type, canonical_size


def format_catalog_inline() -> str:
    """Human-readable one-line catalog string."""
    return "; ".join(
        f"{thread_type.upper()}: {', '.join(sizes)}"
        for thread_type, sizes in THREAD_SIZES_BY_TYPE.items()
    )


def thread_size_markdown_table() -> str:
    """Markdown table used in prompts/docs."""
    rows = [
        "| Type   | Available Sizes |",
        "|--------|-----------------|",
    ]
    for thread_type in ("metric", "unc", "unf"):
        label = thread_type.upper() if thread_type != "metric" else "Metric"
        rows.append(f"| {label} | {', '.join(THREAD_SIZES_BY_TYPE[thread_type])} |")
    return "\n".join(rows)


def thread_sizes_for_type(thread_type: str) -> Sequence[str]:
    """Get supported sizes for a canonical thread type."""
    return THREAD_SIZES_BY_TYPE[thread_type]
