"""
Design entity iteration helpers for Fusion 360.

Why this exists:
- Inserted components typically live under occurrences, not as native bodies on
  design.rootComponent.bRepBodies.
- The LLM needs entity_context (bodies/faces/edges) even when the timeline
  contains only a "Component Insert" feature.

This module provides a deterministic way to iterate *all* BRep bodies visible to
the design, including occurrence proxy bodies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional, Tuple

import adsk.fusion

logger = logging.getLogger(__name__)


def _safe_getattr(obj: Any, attr: str) -> Optional[str]:
    try:
        value = getattr(obj, attr, None)
        if value:
            return str(value)
    except Exception:
        return None
    return None


def iter_brep_bodies(design: adsk.fusion.Design) -> Iterator[Tuple[Dict[str, Optional[str]], adsk.fusion.BRepBody]]:
    """
    Yield (context, body) pairs for all bodies in the design.

    Context keys:
    - component: component name (best-effort)
    - occurrence_path: occurrence fullPathName/name when body is a proxy, else None
    """
    root = design.rootComponent

    # 1) Native root bodies first (stable, common case).
    try:
        for body in root.bRepBodies:
            yield ({"component": _safe_getattr(root, "name"), "occurrence_path": None}, body)
    except Exception as exc:  # pragma: no cover - defensive for Fusion runtime nuance
        logger.debug("Failed iterating rootComponent.bRepBodies: %s", exc)

    # 2) Occurrence proxy bodies (inserted components / assemblies).
    occs = []
    try:
        occs = list(root.allOccurrences)
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed reading rootComponent.allOccurrences: %s", exc)
        occs = []

    def _occ_sort_key(occ: Any) -> str:
        return (
            _safe_getattr(occ, "fullPathName")
            or _safe_getattr(occ, "name")
            or _safe_getattr(getattr(occ, "component", None), "name")
            or ""
        )

    try:
        occs.sort(key=_occ_sort_key)
    except Exception:
        pass

    for occ in occs:
        try:
            if bool(getattr(occ, "isSuppressed", False)):
                continue
        except Exception:
            pass

        occ_path = _safe_getattr(occ, "fullPathName") or _safe_getattr(occ, "name")
        comp = getattr(occ, "component", None)
        comp_name = _safe_getattr(comp, "name") or _safe_getattr(root, "name")

        bodies_iter = None
        try:
            bodies_iter = occ.bRepBodies
        except Exception:
            bodies_iter = None

        if bodies_iter is None:
            # Fallback: native bodies on the component (may be in component space).
            try:
                bodies_iter = comp.bRepBodies if comp else []
            except Exception:
                bodies_iter = []

        try:
            for body in bodies_iter:
                yield ({"component": comp_name, "occurrence_path": occ_path}, body)
        except Exception as exc:  # pragma: no cover
            logger.debug("Failed iterating bodies for occurrence %s: %s", occ_path, exc)

