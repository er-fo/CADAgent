"""Shared backend intermediate representation (IR) types.

The IR is intentionally target-independent: numeric lengths are stored in
canonical millimeters, while target adapters translate to Fusion/build123d
units at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, TypedDict, Union

DatumPlane = Literal["XY", "XZ", "YZ"]
IRPlane = str
IRDirection = Literal["positive", "negative"]
IROperationMode = Literal["new", "join", "cut", "intersect"]
IREntityKind = Literal[
    "plane",
    "sketch",
    "sketch_curve",
    "sketch_point",
    "profile",
    "feature",
    "body",
    "face",
    "edge",
    "vertex",
    "occurrence",
    "selection",
]
IRValidity = Literal["valid", "stale", "invalid", "unknown"]
IRSelectionKind = Literal["edge", "face", "body"]
IROperationType = Literal[
    "create_construction_plane",
    "create_sketch",
    "add_rectangle",
    "add_circle",
    "add_line",
    "add_arc",
    "list_sketch_profiles",
    "extrude",
    "revolve",
    "loft",
    "fillet",
    "chamfer",
    "shell",
    "create_simple_hole",
    "create_counterbore_hole",
    "create_tapped_hole",
    "create_external_thread",
    "pattern_feature",
    "list_features",
    "delete_feature",
    "set_feature_suppression",
    "jump_to_timeline_position",
    "select_entities",
    "clear_selection",
]


class IRMetadata(TypedDict, total=False):
    source: Literal["fusion", "studio", "test"]
    request_id: str
    iteration: int
    model: str
    session_id: str


@dataclass(frozen=True)
class IRRef:
    """Stable IR-side reference with optional target handles."""

    kind: IREntityKind
    id: str
    source_operation_id: Optional[str] = None
    alias: Optional[str] = None
    target_handles: Dict[str, str] = field(default_factory=dict)
    fingerprint: Optional[Dict[str, Any]] = None
    validity: IRValidity = "valid"
    generation: int = 0


@dataclass(frozen=True)
class IROperationEffects:
    """Entities created, modified, or invalidated by an operation."""

    creates: Dict[str, List[str]] = field(default_factory=dict)
    modifies: Dict[str, List[str]] = field(default_factory=dict)
    invalidates: Dict[str, List[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class CreateSketchParams:
    plane: IRPlane
    sketch: str
    sketch_name: Optional[str] = None
    plane_ref: Optional[IRRef] = None


@dataclass(frozen=True)
class AddRectangleParams:
    sketch: str
    center: List[float]
    width: float
    height: float
    corner1: Optional[List[float]] = None
    corner2: Optional[List[float]] = None
    rectangle_id: Optional[str] = None


@dataclass(frozen=True)
class AddCircleParams:
    sketch: str
    center: List[float]
    radius: float
    circle_id: Optional[str] = None


@dataclass(frozen=True)
class CreateConstructionPlaneParams:
    plane: str
    mode: Literal["datum", "offset_from_datum", "angle_to_edge", "face_normal"]
    description: str = ""
    datum_axis_plane: Optional[str] = None
    base_datum_plane: Optional[str] = None
    offset: Optional[float] = None
    reference_face: Optional[str] = None
    reference_edge: Optional[str] = None
    angle_degrees: Optional[float] = None
    face: Optional[str] = None
    point: Optional[List[float]] = None


@dataclass(frozen=True)
class AddLineParams:
    sketch: str
    start: List[float]
    end: List[float]
    line_id: Optional[str] = None


@dataclass(frozen=True)
class AddArcParams:
    sketch: str
    center: List[float]
    start: List[float]
    end: List[float]
    arc_id: Optional[str] = None


@dataclass(frozen=True)
class ExtrudeParams:
    profile: str
    distance: float
    direction: IRDirection
    operation: IROperationMode
    sketch: Optional[str] = None
    profile_index: Optional[int] = None
    profile_indices: Optional[List[int]] = None
    feature_name: Optional[str] = None
    fallback_policy: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ListSketchProfilesParams:
    sketch: str


@dataclass(frozen=True)
class RevolveParams:
    profile: str
    sketch: str
    axis: Dict[str, Any]
    extent: Dict[str, Any]
    operation: IROperationMode = "new"
    profile_index: int = 0
    is_solid: bool = True
    creation_occurrence: Optional[str] = None
    feature_name: Optional[str] = None


@dataclass(frozen=True)
class LoftParams:
    profile_ids: List[str]
    operation: Literal["new", "join", "cut"] = "new"
    feature_name: Optional[str] = None
    prefer_solid: bool = True
    fallback_policy: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class FilletParams:
    edge_refs: List[str]
    radius: float
    include_tangent_edges: bool = True
    feature_name: Optional[str] = None


@dataclass(frozen=True)
class ChamferParams:
    edge_refs: List[str]
    distance: float
    include_tangent_edges: bool = True
    feature_name: Optional[str] = None


@dataclass(frozen=True)
class ShellParams:
    mode: Literal["open", "closed"]
    face_refs: List[str] = field(default_factory=list)
    body_refs: List[str] = field(default_factory=list)
    inside_thickness: float = 0.0
    outside_thickness: float = 0.0
    is_tangent_chain: bool = True
    shell_type: Literal["sharp", "rounded"] = "sharp"
    feature_name: Optional[str] = None


@dataclass(frozen=True)
class SimpleHoleParams:
    face_ref: str
    center: List[float]
    diameter: float
    extent_type: Literal["through_all", "distance"]
    depth: Optional[float] = None
    feature_name: Optional[str] = None


@dataclass(frozen=True)
class CounterboreHoleParams:
    face_ref: str
    center: List[float]
    hole_diameter: float
    hole_depth: float
    counterbore_diameter: float
    counterbore_depth: float
    feature_name: Optional[str] = None


@dataclass(frozen=True)
class TappedHoleParams:
    face_ref: str
    center: List[float]
    thread_type: Literal["metric", "unc", "unf"]
    thread_size: str
    thread_depth: float
    pilot_hole_depth: Optional[float] = None
    diameter_unit: str = "mm"
    feature_name: Optional[str] = None


@dataclass(frozen=True)
class ExternalThreadParams:
    face_ref: str
    thread_type: Literal["metric", "unc", "unf"]
    thread_size: str
    is_full_length: bool = True
    thread_length: Optional[float] = None
    thread_offset: float = 0.0
    diameter_unit: str = "mm"
    feature_name: Optional[str] = None


@dataclass(frozen=True)
class PatternFeatureParams:
    pattern_type: Literal["rectangular", "circular"]
    feature_refs: List[str]
    count_x: Optional[int] = None
    spacing_x: Optional[float] = None
    count_y: Optional[int] = None
    spacing_y: Optional[float] = None
    rotation_count: Optional[int] = None
    rotation_angle_degrees: Optional[float] = None
    orientation_hint: Optional[Any] = None
    feature_name: Optional[str] = None


@dataclass(frozen=True)
class ListFeaturesParams:
    description: str = ""


@dataclass(frozen=True)
class DeleteFeatureParams:
    feature_ref: str
    description: str = ""
    expected_name: Optional[str] = None
    expected_timeline_index: Optional[int] = None


@dataclass(frozen=True)
class FeatureSuppressionParams:
    feature_ref: str
    suppress: bool
    description: str = ""
    expected_name: Optional[str] = None
    expected_timeline_index: Optional[int] = None


@dataclass(frozen=True)
class JumpToTimelinePositionParams:
    target_index: int
    reason: str = ""
    description: str = ""


@dataclass(frozen=True)
class SelectEntitiesParams:
    kind: IRSelectionKind
    refs: List[str]
    clear_existing: bool = True


@dataclass(frozen=True)
class ClearSelectionParams:
    kind: IRSelectionKind


IRParams = Union[
    CreateConstructionPlaneParams,
    CreateSketchParams,
    AddRectangleParams,
    AddCircleParams,
    AddLineParams,
    AddArcParams,
    ListSketchProfilesParams,
    ExtrudeParams,
    RevolveParams,
    LoftParams,
    FilletParams,
    ChamferParams,
    ShellParams,
    SimpleHoleParams,
    CounterboreHoleParams,
    TappedHoleParams,
    ExternalThreadParams,
    PatternFeatureParams,
    ListFeaturesParams,
    DeleteFeatureParams,
    FeatureSuppressionParams,
    JumpToTimelinePositionParams,
    SelectEntitiesParams,
    ClearSelectionParams,
]


@dataclass(frozen=True)
class IROperation:
    id: str
    type: IROperationType
    params: IRParams
    dependencies: List[str] = field(default_factory=list)
    metadata: Optional[IRMetadata] = None
    requires: List[str] = field(default_factory=list)
    effects: IROperationEffects = field(default_factory=IROperationEffects)
    selectors: List[Mapping[str, Any]] = field(default_factory=list)
    fallback_policy: List[str] = field(default_factory=list)
    target_results: List[Mapping[str, Any]] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IRDocument:
    version: str
    units: Literal["mm"]
    operations: List[IROperation]
    entities: Dict[str, List[IRRef]] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
