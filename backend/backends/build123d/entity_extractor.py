"""Extract real geometry/entity metadata from build123d shapes."""

from __future__ import annotations

from typing import Any, Dict


def _count(callable_attr: Any) -> int:
    if callable(callable_attr):
        try:
            return len(callable_attr())
        except Exception:
            return 0
    return 0


def extract_entities(part: Any) -> Dict[str, Any]:
    """Return topology + metric metadata from a built shape."""
    bbox = None
    try:
        bb = part.bounding_box()
        bbox = {
            "min": [float(bb.min.X), float(bb.min.Y), float(bb.min.Z)],
            "max": [float(bb.max.X), float(bb.max.Y), float(bb.max.Z)],
            "size": [float(bb.size.X), float(bb.size.Y), float(bb.size.Z)],
        }
    except Exception:
        bbox = None

    volume = None
    try:
        volume = float(part.volume)
    except Exception:
        volume = None

    return {
        "faces": _count(getattr(part, "faces", None)),
        "edges": _count(getattr(part, "edges", None)),
        "vertices": _count(getattr(part, "vertices", None)),
        "volume_mm3": volume,
        "bounding_box": bbox,
    }
