"""Extract real geometry/entity metadata from build123d shapes."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional


def _count(callable_attr: Any) -> int:
    if callable(callable_attr):
        try:
            return len(callable_attr())
        except Exception:
            return 0
    return 0


def _items(callable_attr: Any) -> List[Any]:
    if callable(callable_attr):
        try:
            return list(callable_attr())
        except Exception:
            return []
    return []


def _number(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return round(float(value), 6)
    except Exception:
        return None


def _vector(value: Any) -> Optional[List[float]]:
    try:
        return [
            round(float(value.X), 6),
            round(float(value.Y), 6),
            round(float(value.Z), 6),
        ]
    except Exception:
        return None


def _enum_name(value: Any) -> Optional[str]:
    try:
        return str(value.name)
    except Exception:
        try:
            return str(value)
        except Exception:
            return None


def _call_noargs(obj: Any, name: str) -> Any:
    attr = getattr(obj, name, None)
    if callable(attr):
        try:
            return attr()
        except Exception:
            return None
    return attr


def _bounding_box(shape: Any) -> Optional[Dict[str, List[float]]]:
    try:
        bb = shape.bounding_box()
        return {
            "min": _vector(bb.min) or [],
            "max": _vector(bb.max) or [],
            "size": _vector(bb.size) or [],
        }
    except Exception:
        return None


def _normal(shape: Any) -> Optional[List[float]]:
    normal_at = getattr(shape, "normal_at", None)
    if callable(normal_at):
        try:
            return _vector(normal_at(0.5, 0.5))
        except Exception:
            return None
    return None


def _radius(shape: Any) -> Optional[float]:
    try:
        return _number(shape.radius)
    except Exception:
        return None


def _fingerprint(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _entity_metadata(kind: str, index: int, shape: Any) -> Dict[str, Any]:
    bbox = _bounding_box(shape)
    metadata: Dict[str, Any] = {
        "id": f"{kind}_{index}",
        "index": index,
        "geometry_type": _enum_name(getattr(shape, "geom_type", None)),
        "topology_type": _enum_name(_call_noargs(shape, "shape_type")),
        "hash_code": _call_noargs(shape, "hash_code"),
        "bounding_box": bbox,
        "center": _vector(_call_noargs(shape, "center")),
    }

    volume = _number(getattr(shape, "volume", None))
    area = _number(getattr(shape, "area", None))
    length = _number(getattr(shape, "length", None))
    normal = _normal(shape)
    radius = _radius(shape)

    if volume is not None:
        metadata["volume_mm3"] = volume
    if area is not None:
        metadata["area_mm2"] = area
    if length is not None:
        metadata["length_mm"] = length
    if normal is not None:
        metadata["normal"] = normal
    if radius is not None:
        metadata["radius_mm"] = radius

    fingerprint_payload = {
        key: metadata[key]
        for key in (
            "geometry_type",
            "topology_type",
            "bounding_box",
            "center",
            "volume_mm3",
            "area_mm2",
            "length_mm",
            "normal",
            "radius_mm",
        )
        if key in metadata
    }
    metadata["fingerprint"] = _fingerprint(fingerprint_payload)
    return {key: value for key, value in metadata.items() if value is not None}


def extract_entities(part: Any) -> Dict[str, Any]:
    """Return topology + metric metadata from a built shape."""
    bbox = _bounding_box(part)

    volume = None
    try:
        volume = float(part.volume)
    except Exception:
        volume = None

    bodies = _items(getattr(part, "solids", None))
    faces = _items(getattr(part, "faces", None))
    edges = _items(getattr(part, "edges", None))

    return {
        "bodies": len(bodies),
        "faces": len(faces),
        "edges": len(edges),
        "vertices": _count(getattr(part, "vertices", None)),
        "volume_mm3": volume,
        "bounding_box": bbox,
        "bodies_metadata": [_entity_metadata("body", index, body) for index, body in enumerate(bodies)],
        "faces_metadata": [_entity_metadata("face", index, face) for index, face in enumerate(faces)],
        "edges_metadata": [_entity_metadata("edge", index, edge) for index, edge in enumerate(edges)],
    }
