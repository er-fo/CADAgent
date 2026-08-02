"""Session-scoped sketch entity and constraint reference store.

Tracks canonical refs and optional aliases for sketch geometry and constraints.
This store is separate from EntityStore (solid BRep entities) because sketch
constraint tools need curve/point-level resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


def _normalize_ref_key(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _normalize_point_key(value: str) -> str:
    normalized = _normalize_ref_key(value)
    aliases = {
        "start": "start",
        "startpoint": "start",
        "p0": "start",
        "end": "end",
        "endpoint": "end",
        "p1": "end",
        "center": "center",
        "centre": "center",
        "mid": "midpoint",
        "midpoint": "midpoint",
    }
    return aliases.get(normalized, normalized)


def _build_normalized_lookup(values: Dict[str, str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for key, val in values.items():
        norm = _normalize_ref_key(key)
        if norm and norm not in lookup:
            lookup[norm] = val
    return lookup


@dataclass
class SketchEntityRef:
    ref_id: str
    token: str
    kind: str
    sketch_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SketchEntityStore:
    """Manage sketch curve/point and constraint refs across tool calls."""

    def __init__(self) -> None:
        self._entities_by_sketch: Dict[str, Dict[str, SketchEntityRef]] = {}
        self._aliases_by_sketch: Dict[str, Dict[str, str]] = {}
        self._constraints_by_sketch: Dict[str, Dict[str, SketchEntityRef]] = {}
        self._constraint_aliases_by_sketch: Dict[str, Dict[str, str]] = {}
        self._counters: Dict[str, Dict[str, int]] = {}
        self._constraint_counters: Dict[str, Dict[str, int]] = {}
        self._origin_tokens: Dict[str, str] = {}
        self._sketch_metadata: Dict[str, Dict[str, Any]] = {}
        self._sketch_aliases: Dict[str, str] = {}

    def clear(self) -> None:
        self._entities_by_sketch.clear()
        self._aliases_by_sketch.clear()
        self._constraints_by_sketch.clear()
        self._constraint_aliases_by_sketch.clear()
        self._counters.clear()
        self._constraint_counters.clear()
        self._origin_tokens.clear()
        self._sketch_metadata.clear()
        self._sketch_aliases.clear()

    def register_entity(
        self,
        sketch_id: str,
        kind: str,
        token: str,
        metadata: Dict[str, Any],
        ref_id: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> str:
        if sketch_id not in self._entities_by_sketch:
            self._entities_by_sketch[sketch_id] = {}
            self._aliases_by_sketch[sketch_id] = {}
            self._counters[sketch_id] = {}

        if not ref_id:
            counter = self._counters[sketch_id].get(kind, 0)
            ref_id = f"{kind}_{counter}"
            self._counters[sketch_id][kind] = counter + 1

        if ref_id in self._entities_by_sketch[sketch_id]:
            raise ValueError(f"Sketch ref '{ref_id}' already exists in sketch '{sketch_id}'")

        self._entities_by_sketch[sketch_id][ref_id] = SketchEntityRef(
            ref_id=ref_id,
            token=token,
            kind=kind,
            sketch_id=sketch_id,
            metadata=metadata,
        )

        if alias:
            if alias in self._aliases_by_sketch[sketch_id]:
                raise ValueError(f"Sketch alias '{alias}' already exists in sketch '{sketch_id}'")
            self._aliases_by_sketch[sketch_id][alias] = ref_id

        return ref_id

    def register_constraint(
        self,
        sketch_id: str,
        kind: str,
        token: str,
        metadata: Dict[str, Any],
        ref_id: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> str:
        if sketch_id not in self._constraints_by_sketch:
            self._constraints_by_sketch[sketch_id] = {}
            self._constraint_aliases_by_sketch[sketch_id] = {}
            self._constraint_counters[sketch_id] = {}

        if not ref_id:
            counter = self._constraint_counters[sketch_id].get(kind, 0)
            ref_id = f"{kind}_{counter}"
            self._constraint_counters[sketch_id][kind] = counter + 1

        self._constraints_by_sketch[sketch_id][ref_id] = SketchEntityRef(
            ref_id=ref_id,
            token=token,
            kind=kind,
            sketch_id=sketch_id,
            metadata=metadata,
        )

        if alias:
            if alias in self._constraint_aliases_by_sketch[sketch_id]:
                raise ValueError(f"Constraint alias '{alias}' already exists in sketch '{sketch_id}'")
            self._constraint_aliases_by_sketch[sketch_id][alias] = ref_id

        return ref_id

    def unregister_constraint(self, sketch_id: str, ref_id: str) -> bool:
        """Remove a constraint ref (and aliases) from the store.

        Returns True when a constraint was removed, False when nothing matched.
        """
        sketch_key = str(sketch_id or "").strip()
        ref_key = str(ref_id or "").strip()
        if not sketch_key or not ref_key:
            return False

        refs = self._constraints_by_sketch.get(sketch_key)
        if not refs or ref_key not in refs:
            return False

        refs.pop(ref_key, None)

        aliases = self._constraint_aliases_by_sketch.get(sketch_key, {})
        aliases_to_remove = [alias for alias, target_ref in aliases.items() if target_ref == ref_key]
        for alias in aliases_to_remove:
            aliases.pop(alias, None)

        return True

    def register_origin(self, sketch_id: str, origin_token: str) -> None:
        """Register the sketch origin point token for a given sketch."""
        self._origin_tokens[sketch_id] = origin_token

    def register_sketch_metadata(self, sketch_id: str, metadata: Dict[str, Any]) -> None:
        """Attach sketch-level metadata (plane/orientation/bounds) to a sketch id."""
        sketch_key = str(sketch_id or "").strip()
        if not sketch_key:
            return

        metadata_copy = dict(metadata or {})
        self._sketch_metadata[sketch_key] = metadata_copy

        aliases = metadata_copy.get("aliases")
        if isinstance(aliases, (list, tuple, set)):
            for alias in aliases:
                self.register_sketch_alias(sketch_key, str(alias or ""))

        for alias_key in ("sketch_name", "name", "display_name"):
            alias = metadata_copy.get(alias_key)
            if isinstance(alias, str):
                self.register_sketch_alias(sketch_key, alias)

    def get_sketch_metadata(self, sketch_id: str) -> Dict[str, Any]:
        """Return a copy of sketch-level metadata for the given sketch id."""
        resolved_sketch_id = self.resolve_sketch_id(sketch_id) or str(sketch_id or "").strip()
        metadata = self._sketch_metadata.get(resolved_sketch_id, {})
        return dict(metadata)

    def register_sketch_alias(self, sketch_id: str, alias: str) -> None:
        """Register an alternate user-visible name for a canonical sketch id."""
        sketch_key = str(sketch_id or "").strip()
        alias_key = str(alias or "").strip()
        if not sketch_key or not alias_key or alias_key == sketch_key:
            return
        self._sketch_aliases[alias_key] = sketch_key

    def resolve_sketch_id(self, sketch_id_or_alias: str) -> Optional[str]:
        """Resolve a sketch id or display-name alias to the canonical sketch id."""
        key = str(sketch_id_or_alias or "").strip()
        if not key:
            return None
        if (
            key in self._sketch_metadata
            or key in self._entities_by_sketch
            or key in self._origin_tokens
        ):
            return key
        if key in self._sketch_aliases:
            return self._sketch_aliases[key]

        normalized_key = _normalize_ref_key(key)
        if not normalized_key:
            return None

        normalized_aliases = _build_normalized_lookup(self._sketch_aliases)
        if normalized_key in normalized_aliases:
            return normalized_aliases[normalized_key]

        known_sketches = {
            sketch_id: sketch_id
            for sketch_id in set(self._sketch_metadata) | set(self._entities_by_sketch) | set(self._origin_tokens)
        }
        normalized_known = _build_normalized_lookup(known_sketches)
        return normalized_known.get(normalized_key)

    def resolve_ref(self, sketch_id: str, ref_id_or_alias: str) -> Optional[str]:
        if ref_id_or_alias.lower() == "origin":
            return self._origin_tokens.get(sketch_id)
        refs = self._entities_by_sketch.get(sketch_id, {})
        aliases = self._aliases_by_sketch.get(sketch_id, {})
        canonical = aliases.get(ref_id_or_alias, ref_id_or_alias)
        if canonical not in refs:
            normalized_aliases = _build_normalized_lookup(aliases)
            normalized_refs = _build_normalized_lookup({k: k for k in refs.keys()})
            normalized_ref = _normalize_ref_key(ref_id_or_alias)
            canonical = normalized_aliases.get(normalized_ref) or normalized_refs.get(normalized_ref) or canonical
        entity = refs.get(canonical)
        return entity.token if entity else None

    def resolve_constraint_ref(self, sketch_id: str, ref_id_or_alias: str) -> Optional[str]:
        refs = self._constraints_by_sketch.get(sketch_id, {})
        aliases = self._constraint_aliases_by_sketch.get(sketch_id, {})
        canonical = aliases.get(ref_id_or_alias, ref_id_or_alias)
        if canonical not in refs:
            normalized_aliases = _build_normalized_lookup(aliases)
            normalized_refs = _build_normalized_lookup({k: k for k in refs.keys()})
            normalized_ref = _normalize_ref_key(ref_id_or_alias)
            canonical = normalized_aliases.get(normalized_ref) or normalized_refs.get(normalized_ref) or canonical
        entry = refs.get(canonical)
        return entry.token if entry else None

    def resolve_dimension_ref(self, sketch_id: str, ref_id_or_alias: str) -> Optional[str]:
        return self.resolve_constraint_ref(sketch_id, ref_id_or_alias)

    def resolve_point_ref(self, sketch_id: str, point_ref: str) -> Optional[str]:
        # entity.endpoint form (line_0.start, arc_0.end, my_alias.center)
        if "." not in point_ref:
            if point_ref.lower() == "origin":
                return self._origin_tokens.get(sketch_id)
            return self.resolve_ref(sketch_id, point_ref)

        entity_id, point_key = point_ref.split(".", 1)
        refs = self._entities_by_sketch.get(sketch_id, {})
        aliases = self._aliases_by_sketch.get(sketch_id, {})
        canonical = aliases.get(entity_id, entity_id)
        if canonical not in refs:
            normalized_aliases = _build_normalized_lookup(aliases)
            normalized_refs = _build_normalized_lookup({k: k for k in refs.keys()})
            normalized_entity = _normalize_ref_key(entity_id)
            canonical = normalized_aliases.get(normalized_entity) or normalized_refs.get(normalized_entity) or canonical
        entry = refs.get(canonical)
        if not entry:
            return None

        point_tokens = entry.metadata.get("point_tokens", {})
        token = point_tokens.get(point_key)
        if not token:
            normalized_lookup = {
                _normalize_point_key(key): value
                for key, value in point_tokens.items()
                if isinstance(key, str)
            }
            token = normalized_lookup.get(_normalize_point_key(point_key))
        return token if isinstance(token, str) and token else None

    def get_kind(self, sketch_id: str, ref_id_or_alias: str) -> Optional[str]:
        refs = self._entities_by_sketch.get(sketch_id, {})
        aliases = self._aliases_by_sketch.get(sketch_id, {})
        canonical = aliases.get(ref_id_or_alias, ref_id_or_alias)
        if canonical not in refs:
            normalized_aliases = _build_normalized_lookup(aliases)
            normalized_refs = _build_normalized_lookup({k: k for k in refs.keys()})
            normalized_ref = _normalize_ref_key(ref_id_or_alias)
            canonical = normalized_aliases.get(normalized_ref) or normalized_refs.get(normalized_ref) or canonical
        entry = refs.get(canonical)
        return entry.kind if entry else None

    def get_ref_and_alias_maps(self, sketch_id: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        refs = {
            k: v.token
            for k, v in self._entities_by_sketch.get(sketch_id, {}).items()
        }
        aliases = dict(self._aliases_by_sketch.get(sketch_id, {}))
        return refs, aliases

    def describe_entity_token(self, sketch_id: str, token: str) -> Dict[str, Any]:
        """Return user-facing refs/aliases that map to a sketch entity/point token."""
        token_text = str(token or "").strip()
        if not token_text:
            return {
                "entity_refs": [],
                "entity_aliases": [],
                "point_refs": [],
                "point_aliases": [],
            }

        refs = self._entities_by_sketch.get(sketch_id, {})
        aliases = self._aliases_by_sketch.get(sketch_id, {})

        aliases_by_ref: Dict[str, list[str]] = {}
        for alias, ref_id in aliases.items():
            aliases_by_ref.setdefault(ref_id, []).append(alias)

        entity_refs: list[str] = []
        entity_aliases: list[str] = []
        point_refs: list[str] = []
        point_aliases: list[str] = []

        def _append_unique(target: list[str], value: Any) -> None:
            text = str(value or "").strip()
            if text and text not in target:
                target.append(text)

        if self._origin_tokens.get(sketch_id) == token_text:
            _append_unique(point_refs, "origin")

        for ref_id, entry in refs.items():
            if entry.token == token_text:
                _append_unique(entity_refs, ref_id)
                for alias in aliases_by_ref.get(ref_id, []):
                    _append_unique(entity_aliases, alias)

            point_tokens = entry.metadata.get("point_tokens") if isinstance(entry.metadata, dict) else {}
            if not isinstance(point_tokens, dict):
                continue
            for raw_point_key, point_token in point_tokens.items():
                if str(point_token or "").strip() != token_text:
                    continue
                point_key = _normalize_point_key(str(raw_point_key))
                _append_unique(point_refs, f"{ref_id}.{point_key}")
                for alias in aliases_by_ref.get(ref_id, []):
                    _append_unique(point_aliases, f"{alias}.{point_key}")

        return {
            "entity_refs": entity_refs,
            "entity_aliases": entity_aliases,
            "point_refs": point_refs,
            "point_aliases": point_aliases,
        }
