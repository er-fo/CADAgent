"""
Per-session entity reference store.

Maps short, LLM-friendly reference IDs to the opaque Fusion 360 entity tokens.

ID scheme:
- Faces: sequential face_0, face_1, face_2... (global counter, no body prefix)
- Edges: sequential e0, e1, e2...
- Vertices: sequential v0, v1, v2...
- Bodies: sequential body_0, body_1...

The store is pre-populated with entity context extracted from Fusion
before each request. Spatial metadata (normal, centroid, surface_type)
is preserved for LLM-side filtering.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# Fingerprint validation tolerances for persistent cache
NORMAL_DOT_THRESHOLD = 0.9998      # ~1° angular tolerance for normal comparison
POSITION_EPSILON = 0.05            # 0.05mm absolute tolerance for positions
AREA_RELATIVE_TOLERANCE = 0.02    # 2% relative tolerance for area
AREA_ABSOLUTE_FLOOR = 0.1         # 0.1mm² minimum tolerance for area
LENGTH_RELATIVE_TOLERANCE = 0.02  # 2% relative tolerance for length/dimensions
LENGTH_ABSOLUTE_FLOOR = 0.05      # 0.05mm minimum tolerance for length


def _normalize_vector(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalize a 3D vector to unit length."""
    import math
    mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if mag < 1e-10:
        return (0.0, 0.0, 0.0)
    return (v[0]/mag, v[1]/mag, v[2]/mag)


def _dot_product(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Compute dot product of two 3D vectors."""
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Compute Euclidean distance between two 3D points."""
    import math
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _values_match(a: Optional[float], b: Optional[float], relative_tol: float, absolute_floor: float) -> bool:
    """Check if two values match within relative tolerance with absolute floor.

    Treats None as "unknown" - if either value is None, skip comparison (return True).
    This prevents ref churn when upstream responses inconsistently include optional fields.
    """
    if a is None or b is None:
        return True  # Unknown values don't cause mismatch
    diff = abs(a - b)
    threshold = max(absolute_floor, relative_tol * max(abs(a), abs(b)))
    return diff <= threshold


@dataclass
class EntityRef:
    """Represents a single entity reference with spatial metadata."""

    ref_id: str
    token: str
    kind: str  # "edge" | "face" | "body" | "vertex"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Spatial properties (populated based on kind)
    # For faces: normal, centroid, area, orientation, is_horizontal, is_vertical, z_level
    # For edges: length, midpoint, direction, adjacent_faces, is_linear
    # For bodies: bbox_min, bbox_max, dimensions, volume, face_count, edge_count

    @property
    def normal(self) -> Optional[Tuple[float, float, float]]:
        """Face normal vector (faces only)."""
        n = self.metadata.get("normal")
        if n and isinstance(n, dict):
            return (float(n.get("x", 0)), float(n.get("y", 0)), float(n.get("z", 0)))
        if n and isinstance(n, (list, tuple)) and len(n) >= 3:
            return (float(n[0]), float(n[1]), float(n[2]))
        return None

    @property
    def centroid(self) -> Optional[Tuple[float, float, float]]:
        """Entity centroid (faces and bodies)."""
        c = self.metadata.get("centroid")
        if c and isinstance(c, dict):
            return (float(c.get("x", 0)), float(c.get("y", 0)), float(c.get("z", 0)))
        if c and isinstance(c, (list, tuple)) and len(c) >= 3:
            return (float(c[0]), float(c[1]), float(c[2]))
        return None

    @property
    def orientation(self) -> Optional[str]:
        """Human-readable orientation (faces only)."""
        return self.metadata.get("orientation")

    @property
    def is_horizontal(self) -> bool:
        """True if face normal is primarily vertical (±Z)."""
        n = self.normal
        if not n:
            return False
        return abs(n[2]) > 0.9  # Within ~25° of vertical

    @property
    def is_vertical(self) -> bool:
        """True if face normal is primarily horizontal (XY plane)."""
        n = self.normal
        if not n:
            return False
        return abs(n[2]) < 0.1  # Within ~6° of horizontal

    @property
    def z_level(self) -> Optional[float]:
        """Z coordinate of centroid (for top/bottom classification)."""
        c = self.centroid
        return c[2] if c else None

    @property
    def dimensions(self) -> Optional[Tuple[float, float, float]]:
        """Body dimensions [x, y, z] from bounding box."""
        d = self.metadata.get("dimensions")
        if d and isinstance(d, (list, tuple)) and len(d) >= 3:
            return (float(d[0]), float(d[1]), float(d[2]))
        return None

    @property
    def midpoint(self) -> Optional[Tuple[float, float, float]]:
        """Edge midpoint (edges only)."""
        m = self.metadata.get("midpoint")
        if m and isinstance(m, (list, tuple)) and len(m) >= 3:
            return (float(m[0]), float(m[1]), float(m[2]))
        return None

    @property
    def direction(self) -> Optional[Tuple[float, float, float]]:
        """Edge direction unit vector (edges only)."""
        d = self.metadata.get("direction")
        if d and isinstance(d, (list, tuple)) and len(d) >= 3:
            return (float(d[0]), float(d[1]), float(d[2]))
        return None

    @property
    def is_linear(self) -> bool:
        """True if edge is a straight line (edges only)."""
        return bool(self.metadata.get("is_linear", False))


@dataclass
class EntityFingerprint:
    """Lightweight fingerprint for cache validation.

    Used to detect when a Fusion entity token has been recycled for a
    geometrically different entity, or when geometry has changed enough
    to warrant ref_id reassignment.
    """
    kind: str  # "face" | "edge" | "body" | "vertex"
    surface_type: Optional[str] = None  # faces only
    normal: Optional[Tuple[float, float, float]] = None  # faces only, normalized
    centroid: Optional[Tuple[float, float, float]] = None  # faces only
    area: Optional[float] = None  # faces only
    edge_type: Optional[str] = None  # edges only
    length: Optional[float] = None  # edges only
    midpoint: Optional[Tuple[float, float, float]] = None  # edges only
    dimensions: Optional[Tuple[float, float, float]] = None  # bodies only
    face_count: Optional[int] = None  # bodies only
    edge_count: Optional[int] = None  # bodies only
    position: Optional[Tuple[float, float, float]] = None  # vertices only

    def matches(self, other: "EntityFingerprint") -> bool:
        """Check if this fingerprint matches another within tolerances.

        Treats None as "unknown" - if either side is None for a field, skip that
        comparison. This prevents ref churn when upstream responses inconsistently
        include optional fields (flat vs spatial_context, partial contexts).
        """
        if self.kind != other.kind:
            return False

        if self.kind == "face":
            # Surface type: skip if either is None
            if self.surface_type and other.surface_type:
                if self.surface_type != other.surface_type:
                    return False
            # Normal: dot product threshold (both must be normalized)
            if self.normal and other.normal:
                dot = _dot_product(self.normal, other.normal)
                if dot < NORMAL_DOT_THRESHOLD:
                    return False
            # Centroid: position epsilon (skip if either is None)
            if self.centroid and other.centroid:
                if _distance(self.centroid, other.centroid) > POSITION_EPSILON:
                    return False
            # Area: relative tolerance with floor (handles None via _values_match)
            if not _values_match(self.area, other.area, AREA_RELATIVE_TOLERANCE, AREA_ABSOLUTE_FLOOR):
                return False
            return True

        elif self.kind == "edge":
            # Edge type: skip if either is None
            if self.edge_type and other.edge_type:
                if self.edge_type != other.edge_type:
                    return False
            # Length: relative tolerance (handles None via _values_match)
            if not _values_match(self.length, other.length, LENGTH_RELATIVE_TOLERANCE, LENGTH_ABSOLUTE_FLOOR):
                return False
            # Midpoint: position epsilon (skip if either is None)
            if self.midpoint and other.midpoint:
                if _distance(self.midpoint, other.midpoint) > POSITION_EPSILON:
                    return False
            return True

        elif self.kind == "body":
            # Dimensions: relative tolerance per component (skip if either is None)
            if self.dimensions and other.dimensions:
                for i in range(3):
                    if not _values_match(self.dimensions[i], other.dimensions[i],
                                        LENGTH_RELATIVE_TOLERANCE, LENGTH_ABSOLUTE_FLOOR):
                        return False
            # Counts: skip if either is None
            if self.face_count is not None and other.face_count is not None:
                if self.face_count != other.face_count:
                    return False
            if self.edge_count is not None and other.edge_count is not None:
                if self.edge_count != other.edge_count:
                    return False
            return True

        elif self.kind == "vertex":
            # Position: epsilon (skip if either is None)
            if self.position and other.position:
                if _distance(self.position, other.position) > POSITION_EPSILON:
                    return False
            return True

        return False


@dataclass
class CachedEntityMapping:
    """Cached token-to-ref mapping with fingerprint for validation."""
    ref_id: str
    fingerprint: EntityFingerprint


class EntityStore:
    """
    Stores bidirectional mappings between short ref IDs and entity tokens.

    Notes:
    - Body IDs: body_0, body_1, ...
    - Face IDs: face_0, face_1, face_2, ... (global sequential across all bodies)
    - Edge IDs: e0, e1, e2, ...
    - Vertex IDs: v0, v1, v2, ...
    - Tokens are treated as opaque handles; equality is token-string equality only.
    - The store is session-scoped. It is reset when the ConnectionManager session closes.
    """

    def __init__(self) -> None:
        # Active state (cleared on soft_clear)
        self._ref_to_entry: Dict[str, EntityRef] = {}
        self._token_to_ref: Dict[str, str] = {}
        self._counters: Dict[str, int] = {}
        self._last_signature: Optional[str] = None
        self._lock = asyncio.Lock()  # Protect counter increments from race conditions

        # Persistent cache (survives soft_clear, only cleared on full clear)
        # Maps token -> CachedEntityMapping (ref_id + fingerprint)
        self._persistent_cache: Dict[str, CachedEntityMapping] = {}
        # Persistent counters to ensure monotonic ID assignment
        self._persistent_counters: Dict[str, int] = {}  # kind -> next_id

        # Session-level cache statistics for time savings estimation
        self._session_stats: Dict[str, int] = {
            "total_entities": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_mismatches": 0,
            "active_reuse": 0,
            "refresh_count": 0,
        }
        self._session_timings_ms: List[float] = []  # Track per-registration durations

    # ------------------------------------------------------------------ #
    # Fingerprint Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_fingerprint(kind: str, entity: Mapping[str, Any]) -> EntityFingerprint:
        """Compute a fingerprint from entity data for cache validation."""
        if kind == "face":
            # Extract and normalize normal
            normal = None
            normal_raw = entity.get("normal")
            if normal_raw:
                if isinstance(normal_raw, (list, tuple)) and len(normal_raw) >= 3:
                    normal = _normalize_vector((float(normal_raw[0]), float(normal_raw[1]), float(normal_raw[2])))
                elif isinstance(normal_raw, dict):
                    normal = _normalize_vector((
                        float(normal_raw.get("x", 0)),
                        float(normal_raw.get("y", 0)),
                        float(normal_raw.get("z", 0)),
                    ))
            # Extract centroid
            centroid = None
            centroid_raw = entity.get("centroid")
            if centroid_raw:
                if isinstance(centroid_raw, (list, tuple)) and len(centroid_raw) >= 3:
                    centroid = (float(centroid_raw[0]), float(centroid_raw[1]), float(centroid_raw[2]))
                elif isinstance(centroid_raw, dict):
                    centroid = (
                        float(centroid_raw.get("x", 0)),
                        float(centroid_raw.get("y", 0)),
                        float(centroid_raw.get("z", 0)),
                    )
            return EntityFingerprint(
                kind="face",
                surface_type=entity.get("surface_type") or entity.get("geometry_type"),
                normal=normal,
                centroid=centroid,
                area=entity.get("area"),
            )

        elif kind == "edge":
            midpoint = None
            midpoint_raw = entity.get("midpoint")
            if midpoint_raw and isinstance(midpoint_raw, (list, tuple)) and len(midpoint_raw) >= 3:
                midpoint = (float(midpoint_raw[0]), float(midpoint_raw[1]), float(midpoint_raw[2]))
            return EntityFingerprint(
                kind="edge",
                edge_type=entity.get("edge_type"),
                length=entity.get("length"),
                midpoint=midpoint,
            )

        elif kind == "body":
            dimensions = None
            dims_raw = entity.get("dimensions")
            if dims_raw and isinstance(dims_raw, (list, tuple)) and len(dims_raw) >= 3:
                dimensions = (float(dims_raw[0]), float(dims_raw[1]), float(dims_raw[2]))
            return EntityFingerprint(
                kind="body",
                dimensions=dimensions,
                face_count=entity.get("face_count"),
                edge_count=entity.get("edge_count"),
            )

        elif kind == "vertex":
            position = None
            pos_raw = entity.get("p") or entity.get("position")
            if pos_raw and isinstance(pos_raw, (list, tuple)) and len(pos_raw) >= 3:
                position = (float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2]))
            return EntityFingerprint(
                kind="vertex",
                position=position,
            )

        return EntityFingerprint(kind=kind)

    # ------------------------------------------------------------------ #
    # Clear Methods
    # ------------------------------------------------------------------ #
    def soft_clear(self) -> None:
        """Clear active store but preserve persistent cache.

        Use for normal geometry refresh when unchanged entities should
        keep their ref_ids. Counters are restored from persistent state
        to ensure monotonic ID assignment.
        """
        self._ref_to_entry.clear()
        self._token_to_ref.clear()
        # Restore counters from persistent state (never decrease)
        self._counters = dict(self._persistent_counters)
        # Track refresh cycles for session stats
        self._session_stats["refresh_count"] += 1

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    async def register_entities(
        self,
        kind: str,
        entities: Iterable[Mapping[str, Any]],
    ) -> List[EntityRef]:
        """
        Register a batch of entities of the same kind.

        Returns a list of EntityRef objects corresponding to the input order.

        New ID scheme:
        - "body": body_0, body_1, ...
        - "face": face_0, face_1, face_2, ... (global sequential)
        - "edge": e0, e1, e2, ...
        - "vertex": v0, v1, v2, ...

        Supported entity kinds and their spatial metadata:
        - "face": normal, centroid, orientation, area, surface_type
        - "edge": midpoint, direction/tangent, is_linear, length, start_coords, end_coords, v0, v1, adjacent_faces
        - "body": dimensions, bbox_min, bbox_max, volume, face_count, edge_count
        - "vertex": position (p)
        """
        registered: List[EntityRef] = []
        start_time = time.perf_counter()
        total = 0
        active_reuse = 0
        cache_hits = 0
        cache_misses = 0
        cache_mismatch = 0
        new_assignments = 0

        # For faces, we need to collect, classify, sort, then assign IDs
        if kind == "face":
            return await self._register_faces(entities)

        for entity in entities:
            token = entity.get("entity_token")
            if not isinstance(token, str) or not token.strip():
                continue
            total += 1

            # Fast path: check active store first (no lock needed for read)
            ref_id = self._token_to_ref.get(token)
            if ref_id is not None:
                # Already registered this refresh cycle - skip fingerprint computation
                active_reuse += 1
            else:
                # Compute fingerprint only when needed for cache validation
                current_fingerprint = self._compute_fingerprint(kind, entity)

                # Lock protects counter increment to prevent TOCTOU races
                async with self._lock:
                    # Re-check under lock (another coroutine may have registered)
                    ref_id = self._token_to_ref.get(token)
                    if ref_id is None:
                        # Check persistent cache with fingerprint validation
                        cached = self._persistent_cache.get(token)
                        if cached and cached.fingerprint.matches(current_fingerprint):
                            # Fingerprint matches - reuse cached ref_id
                            ref_id = cached.ref_id
                            cache_hits += 1
                        else:
                            # No cache hit or fingerprint mismatch - assign new ref_id
                            if cached:
                                cache_mismatch += 1
                            else:
                                cache_misses += 1
                            # NOTE: Edge and vertex IDs are globally sequential (e0, e1, ... across all bodies)
                            # rather than body-scoped (body_0_e0, body_1_e0, ...) because:
                            # 1. IDs must be globally unique for the _token_to_ref lookup map
                            # 2. Body-scoped IDs would create collisions in the token→ref mapping
                            # 3. The nested spatial_context structure already groups entities by body,
                            #    providing organizational clarity without requiring namespaced IDs
                            counter = self._counters.get(kind, 0)
                            self._counters[kind] = counter + 1

                            if kind == "body":
                                ref_id = f"body_{counter}"
                            elif kind == "edge":
                                ref_id = f"e{counter}"
                            elif kind == "vertex":
                                ref_id = f"v{counter}"
                            else:
                                # Fallback for unknown kinds
                                ref_id = f"{kind}_{counter}"

                            # Update persistent cache and counters
                            self._persistent_cache[token] = CachedEntityMapping(
                                ref_id=ref_id,
                                fingerprint=current_fingerprint,
                            )
                            self._persistent_counters[kind] = self._counters[kind]
                            new_assignments += 1

                        self._token_to_ref[token] = ref_id

            # Build metadata with basic fields
            metadata: Dict[str, Any] = {
                "id": entity.get("id"),
                "name": entity.get("name"),
                "body": entity.get("body") or entity.get("body_name"),
                "component": entity.get("component"),
                "occurrence_path": entity.get("occurrence_path"),
                "body_index": entity.get("body_index"),
            }

            # Add spatial metadata based on entity kind
            if kind == "edge":
                # Edge spatial properties
                metadata["midpoint"] = entity.get("midpoint")
                metadata["direction"] = entity.get("direction")
                metadata["tangent"] = entity.get("tangent")  # New: curve evaluator tangent
                metadata["is_linear"] = entity.get("is_linear", False)
                metadata["length"] = entity.get("length")
                metadata["start_coords"] = entity.get("start_coords") or entity.get("p0")
                metadata["end_coords"] = entity.get("end_coords") or entity.get("p1")
                metadata["edge_type"] = entity.get("edge_type")  # New: normalized edge type
                metadata["v0"] = entity.get("v0")  # New: start vertex ID
                metadata["v1"] = entity.get("v1")  # New: end vertex ID
                metadata["adjacent_faces"] = entity.get("adjacent_faces", [])  # New: edge adjacency
            elif kind == "body":
                # Body spatial properties
                metadata["dimensions"] = entity.get("dimensions")
                metadata["volume"] = entity.get("volume")
                metadata["face_count"] = entity.get("face_count")
                metadata["edge_count"] = entity.get("edge_count")
                # Extract bbox_min/bbox_max from bounding_box
                bbox = entity.get("bounding_box") or entity.get("bbox")
                if bbox and isinstance(bbox, dict):
                    metadata["bbox_min"] = bbox.get("min")
                    metadata["bbox_max"] = bbox.get("max")
            elif kind == "vertex":
                # Vertex spatial properties
                metadata["position"] = entity.get("p") or entity.get("position")

            entry = EntityRef(
                ref_id=ref_id,
                token=token,
                kind=kind,
                metadata=metadata,
            )
            self._ref_to_entry[ref_id] = entry
            registered.append(entry)

        if total > 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            lookup_total = cache_hits + cache_misses + cache_mismatch
            hit_rate = (cache_hits / lookup_total * 100.0) if lookup_total else 0.0
            logger.info(
                "EntityStore cache %s: total=%d active_reuse=%d cache_hits=%d cache_misses=%d "
                "cache_mismatch=%d new_ids=%d hit_rate=%.1f%% duration_ms=%.2f",
                kind,
                total,
                active_reuse,
                cache_hits,
                cache_misses,
                cache_mismatch,
                new_assignments,
                hit_rate,
                elapsed_ms,
            )
            # Accumulate session-level statistics
            self._session_stats["total_entities"] += total
            self._session_stats["cache_hits"] += cache_hits
            self._session_stats["cache_misses"] += cache_misses
            self._session_stats["cache_mismatches"] += cache_mismatch
            self._session_stats["active_reuse"] += active_reuse
            self._session_timings_ms.append(elapsed_ms)

        return registered

    async def _register_faces(
        self,
        entities: Iterable[Mapping[str, Any]],
    ) -> List[EntityRef]:
        """Register faces with simple sequential IDs (face_0, face_1, ...).

        Uses a single global counter across all bodies. The persistent cache
        is consulted for fingerprint validation — if a token's geometry matches
        the cached fingerprint, its previous face_N ref is reused. Otherwise
        a new sequential ID is assigned.
        """
        start_time = time.perf_counter()

        # 1. Collect unique faces (de-dup by token, preserve order)
        seen_tokens: set = set()
        all_faces: List[Tuple[Mapping[str, Any], str]] = []

        for entity in entities:
            token = entity.get("entity_token")
            if not isinstance(token, str) or not token.strip():
                continue
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            all_faces.append((entity, token))

        if not all_faces:
            return []

        # 2. Register with sequential IDs; cache for fingerprint validation
        registered: List[EntityRef] = []
        cache_hits = 0
        cache_misses = 0
        cache_mismatch = 0
        active_reuse = 0
        new_assignments = 0
        remap_hits = 0
        remap_ambiguous = 0
        claimed_ref_ids: set = set()

        async with self._lock:
            for entity, token in all_faces:
                # Fast path: already registered this refresh cycle
                ref_id = self._token_to_ref.get(token)
                if ref_id is not None and ref_id in self._ref_to_entry:
                    registered.append(self._ref_to_entry[ref_id])
                    active_reuse += 1
                    claimed_ref_ids.add(ref_id)
                    continue

                fingerprint = self._compute_fingerprint("face", entity)

                # Check persistent cache with fingerprint validation
                cached = self._persistent_cache.get(token)
                if (
                    cached
                    and cached.fingerprint.matches(fingerprint)
                    and cached.ref_id not in claimed_ref_ids
                ):
                    ref_id = cached.ref_id
                    cache_hits += 1
                else:
                    if cached and not cached.fingerprint.matches(fingerprint):
                        cache_mismatch += 1
                    elif not cached:
                        cache_misses += 1

                    # Token churn recovery: if this token is new but geometry matches exactly,
                    # reuse the previous face_N id to keep refs stable across topology refreshes.
                    candidate_ref_ids: List[str] = []
                    for cached_token, cached_mapping in self._persistent_cache.items():
                        if cached_token == token:
                            continue
                        if cached_mapping.ref_id in claimed_ref_ids:
                            continue
                        cached_fp = cached_mapping.fingerprint
                        if cached_fp.kind != "face":
                            continue
                        if cached_fp.matches(fingerprint):
                            candidate_ref_ids.append(cached_mapping.ref_id)

                    unique_candidates = sorted(set(candidate_ref_ids))
                    if len(unique_candidates) == 1:
                        ref_id = unique_candidates[0]
                        remap_hits += 1
                    elif len(unique_candidates) > 1:
                        remap_ambiguous += 1
                        ref_id = None

                    if not ref_id:
                        # Assign next sequential ID
                        counter = self._counters.get("face", 0)
                        self._counters["face"] = counter + 1
                        ref_id = f"face_{counter}"
                        self._persistent_counters["face"] = self._counters["face"]
                        new_assignments += 1

                self._persistent_cache[token] = CachedEntityMapping(
                    ref_id=ref_id,
                    fingerprint=fingerprint,
                )

                self._token_to_ref[token] = ref_id
                claimed_ref_ids.add(ref_id)

                # Build metadata
                normal = None
                normal_raw = entity.get("normal")
                if normal_raw:
                    if isinstance(normal_raw, (list, tuple)) and len(normal_raw) >= 3:
                        normal = (float(normal_raw[0]), float(normal_raw[1]), float(normal_raw[2]))
                    elif isinstance(normal_raw, dict):
                        normal = (
                            float(normal_raw.get("x", 0)),
                            float(normal_raw.get("y", 0)),
                            float(normal_raw.get("z", 0)),
                        )

                metadata: Dict[str, Any] = {
                    "id": entity.get("id"),
                    "name": entity.get("name"),
                    "body": entity.get("body") or entity.get("body_name"),
                    "component": entity.get("component"),
                    "normal": normal or entity.get("normal"),
                    "centroid": entity.get("centroid"),
                    "area": entity.get("area"),
                    "surface_type": entity.get("surface_type") or entity.get("geometry_type"),
                    "frame": entity.get("frame"),
                    "loops": entity.get("loops"),
                }

                entry = EntityRef(
                    ref_id=ref_id,
                    token=token,
                    kind="face",
                    metadata=metadata,
                )
                self._ref_to_entry[ref_id] = entry
                registered.append(entry)

        # 3. Log
        total = len(all_faces)
        if total > 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            lookups = cache_hits + cache_misses + cache_mismatch
            hit_rate = (cache_hits / lookups * 100.0) if lookups else 0.0
            logger.info(
                "EntityStore cache face: total=%d active_reuse=%d cache_hits=%d "
                "cache_misses=%d cache_mismatch=%d remap_hits=%d remap_ambiguous=%d "
                "new_ids=%d hit_rate=%.1f%% "
                "duration_ms=%.2f",
                total, active_reuse, cache_hits, cache_misses, cache_mismatch,
                remap_hits, remap_ambiguous,
                new_assignments, hit_rate, elapsed_ms,
            )
            self._session_stats["total_entities"] += total
            self._session_stats["cache_hits"] += cache_hits
            self._session_stats["cache_misses"] += cache_misses
            self._session_stats["cache_mismatches"] += cache_mismatch
            self._session_stats["active_reuse"] += active_reuse
            self._session_timings_ms.append(elapsed_ms)

        return registered

    # ------------------------------------------------------------------ #
    # Resolution
    # ------------------------------------------------------------------ #
    def resolve_token(
        self,
        ref_or_token: str,
        expected_kind: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve a single ref ID to its token. If the input is already a token,
        it is passed through.

        Supported ref patterns:
        - body_N (bodies)
        - face_N (faces: face_0, face_1, ...)
        - eN (edges: e0, e1, ...)
        - vN (vertices: v0, v1, ...)

        Returns:
            (token, error_message) where error_message is None on success.
        """
        if not isinstance(ref_or_token, str):
            return None, "Identifier must be a string."

        cleaned = ref_or_token.strip()
        if not cleaned:
            return None, "Identifier cannot be empty."

        entry = self._ref_to_entry.get(cleaned)
        if entry:
            if expected_kind and entry.kind != expected_kind:
                return None, f"Expected a {expected_kind} ref but got {entry.kind}: {cleaned}"
            return entry.token, None

        # Check for known ref patterns that should be registered but aren't
        is_ref_pattern = (
            re.match(r"^face_\d+$", cleaned) or                # face_0, face_1, face_2
            re.match(r"^e\d+$", cleaned) or                    # e0, e1, e2
            re.match(r"^v\d+$", cleaned) or                    # v0, v1, v2
            re.match(r"^body_\d+$", cleaned) or                # body_0, body_1
            re.match(r"^[a-z]+_\d+$", cleaned)                 # Legacy: kind_N
        )
        
        if is_ref_pattern:
            return None, f"Unknown entity ref: {cleaned}"

        # Not a ref; assume it's already a token
        return cleaned, None

    def resolve_tokens(
        self,
        refs_or_tokens: Iterable[str],
        expected_kind: Optional[str] = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Resolve a list of ref IDs to tokens.

        Returns:
            (tokens, missing_refs, kind_mismatches)
        """
        tokens: List[str] = []
        missing: List[str] = []
        kind_errors: List[str] = []

        for item in refs_or_tokens:
            token, error = self.resolve_token(item, expected_kind=expected_kind)
            if token and not error:
                tokens.append(token)
            elif error:
                kind_errors.append(f"{item}: {error}")
            else:
                missing.append(str(item))

        return tokens, missing, kind_errors

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def has_ref(self, ref_id: str) -> bool:
        return ref_id in self._ref_to_entry

    def get_entry(self, ref_id: str) -> Optional[EntityRef]:
        return self._ref_to_entry.get(ref_id)

    def get_refs_by_kind(self, kind: str) -> List[str]:
        """
        Get all ref IDs for a specific entity kind.

        Args:
            kind: Entity kind to filter by (e.g., "edge", "face", "body", "vertex")

        Returns:
            List of ref IDs matching the specified kind
            - bodies: body_0, body_1, ...
            - faces: face_0, face_1, face_2, ...
            - edges: e0, e1, e2, ...
            - vertices: v0, v1, v2, ...
        """
        return [ref_id for ref_id, entry in self._ref_to_entry.items() if entry.kind == kind]

    def get_all_edges(self) -> List[str]:
        """Get all edge ref IDs (e0, e1, e2, ...)."""
        return self.get_refs_by_kind("edge")

    def get_all_faces(self) -> List[str]:
        """Get all face ref IDs (face_0, face_1, ...)."""
        return self.get_refs_by_kind("face")

    def get_all_bodies(self) -> List[str]:
        """Get all body ref IDs (body_0, body_1, ...)."""
        return self.get_refs_by_kind("body")

    def get_all_vertices(self) -> List[str]:
        """Get all vertex ref IDs (v0, v1, v2, ...)."""
        return self.get_refs_by_kind("vertex")

    # ------------------------------------------------------------------ #
    # Spatial Queries
    # ------------------------------------------------------------------ #
    def get_top_faces(self, tolerance: float = 0.01) -> List[str]:
        """
        Get faces that are likely "top" faces (horizontal, facing +Z, highest Z).
        
        Returns ref IDs of faces sorted by Z level (highest first), limited to
        faces within tolerance of the maximum Z.
        """
        # Find all horizontal faces facing up (+Z)
        candidates: List[Tuple[str, float]] = []
        for ref_id, entry in self._ref_to_entry.items():
            if entry.kind != "face":
                continue
            n = entry.normal
            if not n or n[2] <= 0.9:  # Not facing +Z
                continue
            z = entry.z_level
            if z is not None:
                candidates.append((ref_id, z))
        
        if not candidates:
            return []
        
        # Find max Z and filter to faces near the top
        max_z = max(z for _, z in candidates)
        return [ref_id for ref_id, z in candidates if abs(z - max_z) <= tolerance]

    def get_bottom_faces(self, tolerance: float = 0.01) -> List[str]:
        """Get faces that are likely "bottom" faces (horizontal, facing -Z, lowest Z)."""
        # Find all horizontal faces facing down (-Z)
        candidates: List[Tuple[str, float]] = []
        for ref_id, entry in self._ref_to_entry.items():
            if entry.kind != "face":
                continue
            n = entry.normal
            if not n or n[2] >= -0.9:  # Not facing -Z
                continue
            z = entry.z_level
            if z is not None:
                candidates.append((ref_id, z))
        
        if not candidates:
            return []
        
        # Find min Z and filter to faces near the bottom
        min_z = min(z for _, z in candidates)
        return [ref_id for ref_id, z in candidates if abs(z - min_z) <= tolerance]

    def get_horizontal_faces(self) -> List[str]:
        """Get all horizontal faces (normal primarily in ±Z)."""
        return [
            ref_id for ref_id, entry in self._ref_to_entry.items()
            if entry.kind == "face" and entry.is_horizontal
        ]

    def get_vertical_faces(self) -> List[str]:
        """Get all vertical faces (normal primarily in XY plane)."""
        return [
            ref_id for ref_id, entry in self._ref_to_entry.items()
            if entry.kind == "face" and entry.is_vertical
        ]

    def get_edges_at_z(self, z_level: float, tolerance: float = 0.01) -> List[str]:
        """Get edges whose midpoint is at the specified Z level."""
        results: List[str] = []
        for ref_id, entry in self._ref_to_entry.items():
            if entry.kind != "edge":
                continue
            m = entry.midpoint
            if m and abs(m[2] - z_level) <= tolerance:
                results.append(ref_id)
        return results

    def get_body_dimensions(self, body_ref: str) -> Optional[Dict[str, float]]:
        """Get body dimensions as {x, y, z, volume}."""
        entry = self._ref_to_entry.get(body_ref)
        if not entry or entry.kind != "body":
            return None
        
        dims = entry.dimensions
        volume = entry.metadata.get("volume")
        
        if not dims:
            return None
        
        return {
            "x": dims[0],
            "y": dims[1],
            "z": dims[2],
            "volume": float(volume) if volume is not None else None,
        }

    def compute_spatial_summary(self) -> Dict[str, Any]:
        """
        Compute a summary of spatial relationships for quick LLM lookups.
        
        Returns a dictionary with:
        - top_faces: List of ref IDs for faces at the top of the model
        - bottom_faces: List of ref IDs for faces at the bottom
        - horizontal_faces: All horizontal faces
        - vertical_faces: All vertical faces
        - z_bounds: (min_z, max_z) across all entities
        - top_edges: Edges at the highest Z level
        - bottom_edges: Edges at the lowest Z level
        """
        # Find Z bounds from all entities
        z_values: List[float] = []
        
        for entry in self._ref_to_entry.values():
            if entry.kind == "face":
                z = entry.z_level
                if z is not None:
                    z_values.append(z)
            elif entry.kind == "edge":
                m = entry.midpoint
                if m:
                    z_values.append(m[2])
            elif entry.kind == "body":
                # Use bounding box for bodies
                bbox_min = entry.metadata.get("bbox_min")
                bbox_max = entry.metadata.get("bbox_max")
                if bbox_min and isinstance(bbox_min, (list, tuple)) and len(bbox_min) >= 3:
                    z_values.append(float(bbox_min[2]))
                if bbox_max and isinstance(bbox_max, (list, tuple)) and len(bbox_max) >= 3:
                    z_values.append(float(bbox_max[2]))
        
        min_z = min(z_values) if z_values else 0.0
        max_z = max(z_values) if z_values else 0.0
        
        # Classify faces
        top_faces = self.get_top_faces()
        bottom_faces = self.get_bottom_faces()
        horizontal_faces = self.get_horizontal_faces()
        vertical_faces = self.get_vertical_faces()
        
        # Find top/bottom edges
        top_edges = self.get_edges_at_z(max_z) if z_values else []
        bottom_edges = self.get_edges_at_z(min_z) if z_values else []
        
        return {
            "top_faces": top_faces,
            "bottom_faces": bottom_faces,
            "horizontal_faces": horizontal_faces,
            "vertical_faces": vertical_faces,
            "z_bounds": (min_z, max_z),
            "top_edges": top_edges,
            "bottom_edges": bottom_edges,
        }

    def clear(self) -> None:
        """Full clear including persistent cache.

        Use for timeline rollbacks and session end when all entity refs
        are invalidated and must start fresh.
        """
        # Log session summary before clearing
        self._log_session_summary()

        self._ref_to_entry.clear()
        self._token_to_ref.clear()
        self._counters.clear()
        # Also clear persistent cache (full reset)
        self._persistent_cache.clear()
        self._persistent_counters.clear()
        # Reset session stats
        self._session_stats = {
            "total_entities": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_mismatches": 0,
            "active_reuse": 0,
            "refresh_count": 0,
        }
        self._session_timings_ms.clear()

    def _log_session_summary(self) -> None:
        """Log session-level cache statistics with time savings estimate."""
        stats = self._session_stats
        if stats["total_entities"] == 0:
            return  # Nothing to log

        total = stats["total_entities"]
        hits = stats["cache_hits"]
        misses = stats["cache_misses"]
        mismatches = stats["cache_mismatches"]
        lookups = hits + misses + mismatches
        hit_rate = (hits / lookups * 100.0) if lookups > 0 else 0.0

        total_cache_time_ms = sum(self._session_timings_ms)
        avg_time_per_entity_ms = total_cache_time_ms / total if total > 0 else 0

        # Estimate time saved: cache hits avoid full fingerprint computation
        # Conservative estimate: cache hit is ~5x faster than miss (fingerprint validation vs full assignment)
        # This is a rough estimate based on typical ratios observed in practice
        estimated_miss_time_ms = avg_time_per_entity_ms * 5 if avg_time_per_entity_ms > 0 else 0.02
        estimated_time_saved_ms = hits * (estimated_miss_time_ms - avg_time_per_entity_ms)

        logger.info(
            "EntityStore SESSION SUMMARY: refreshes=%d total_entities=%d "
            "cache_hits=%d cache_misses=%d cache_mismatches=%d hit_rate=%.1f%% "
            "total_cache_time=%.1fms estimated_time_saved=%.1fms",
            stats["refresh_count"], total,
            hits, misses, mismatches, hit_rate,
            total_cache_time_ms, max(0, estimated_time_saved_ms)
        )

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session cache statistics for external logging."""
        stats = self._session_stats.copy()
        lookups = stats["cache_hits"] + stats["cache_misses"] + stats["cache_mismatches"]
        stats["hit_rate"] = (stats["cache_hits"] / lookups * 100.0) if lookups > 0 else 0.0
        stats["total_cache_time_ms"] = sum(self._session_timings_ms)
        return stats

    def is_empty(self) -> bool:
        """Check if the store has no registered entities."""
        return len(self._ref_to_entry) == 0

    def get_entity_counts(self) -> Dict[str, int]:
        """Get counts of entities by kind."""
        counts: Dict[str, int] = {"body": 0, "face": 0, "edge": 0, "vertex": 0}
        for entry in self._ref_to_entry.values():
            if entry.kind in counts:
                counts[entry.kind] += 1
        return counts

    def has_entities_of_kind(self, kind: str) -> bool:
        """Check if the store has any entities of the specified kind."""
        return any(entry.kind == kind for entry in self._ref_to_entry.values())

    # ------------------------------------------------------------------ #
    # Signature Computation for Change Detection
    # ------------------------------------------------------------------ #
    def get_signature(self) -> str:
        """
        Compute a lightweight signature based on tokens + spatial data.

        IMPORTANT: Signature is based on tokens (not ref_ids) so that:
        - Session store and temp store compute identical signatures from identical context
        - Geometry-change detection works reliably regardless of ref_id assignment

        Signature includes:
        - Sorted entity tokens (deterministic ordering)
        - Key spatial properties (normals, centroids, bbox) to detect geometry changes

        Returns empty string if store is empty.
        """
        if not self._ref_to_entry:
            return ""

        parts: List[str] = []

        # Sort by TOKEN for deterministic ordering independent of ref_id assignment
        entries_by_token = sorted(
            self._ref_to_entry.values(),
            key=lambda e: e.token
        )

        for entry in entries_by_token:
            # Use full token - signature is only compared internally, no collision risk
            parts.append(f"{entry.kind}:{entry.token}")

            # Include key spatial properties based on kind
            # Type-safe coercion with error handling
            if entry.kind == "face":
                n = entry.normal
                c = entry.centroid
                if n:
                    try:
                        parts.append(f"n:{float(n[0]):.5f},{float(n[1]):.5f},{float(n[2]):.5f}")
                    except (TypeError, ValueError, IndexError):
                        pass
                if c:
                    try:
                        parts.append(f"c:{float(c[0]):.4f},{float(c[1]):.4f},{float(c[2]):.4f}")
                    except (TypeError, ValueError, IndexError):
                        pass
            elif entry.kind == "edge":
                length = entry.metadata.get("length")
                m = entry.midpoint
                if length is not None:
                    try:
                        parts.append(f"len:{float(length):.4f}")
                    except (TypeError, ValueError):
                        pass
                if m:
                    try:
                        parts.append(f"m:{float(m[0]):.4f},{float(m[1]):.4f},{float(m[2]):.4f}")
                    except (TypeError, ValueError, IndexError):
                        pass
            elif entry.kind == "body":
                dims = entry.dimensions
                if dims:
                    try:
                        parts.append(f"d:{float(dims[0]):.4f},{float(dims[1]):.4f},{float(dims[2]):.4f}")
                    except (TypeError, ValueError, IndexError):
                        pass

        return "|".join(parts)

    def store_signature(self) -> None:
        """Compute and store current signature for later comparison."""
        self._last_signature = self.get_signature()

    def signature_changed(self) -> bool:
        """
        Check if current signature differs from last stored signature.

        Returns True if signature has changed or no previous signature exists.
        """
        current = self.get_signature()
        if self._last_signature is None:
            return True  # First time - consider changed
        return current != self._last_signature

    def prune_stale_tokens(self, current_tokens: Iterable[str]) -> List[str]:
        """Remove entities from persistent cache that no longer exist.

        Call this after each successful context refresh to prevent unbounded
        cache growth. Tokens not in current_tokens are considered deleted
        from the model and are pruned from the persistent cache.

        Args:
            current_tokens: Set of tokens present in the fresh context

        Returns:
            List of pruned ref_ids for logging
        """
        start_time = time.perf_counter()
        current_set = set(current_tokens)
        stale_tokens = [t for t in self._persistent_cache if t not in current_set]
        pruned_refs: List[str] = []

        for token in stale_tokens:
            cached = self._persistent_cache.pop(token, None)
            if cached:
                pruned_refs.append(cached.ref_id)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if pruned_refs:
            logger.info(
                "EntityStore cache prune: removed=%d remaining=%d duration_ms=%.2f",
                len(pruned_refs),
                len(self._persistent_cache),
                elapsed_ms,
            )
        else:
            logger.debug(
                "EntityStore cache prune: removed=0 remaining=%d duration_ms=%.2f",
                len(self._persistent_cache),
                elapsed_ms,
            )

        return pruned_refs
