"""Shared backend intermediate representation (IR) types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

DatumPlane = Literal["XY", "XZ", "YZ"]
IRPlane = str
IRDirection = Literal["positive", "negative"]
IROperationMode = Literal["new", "join", "cut", "intersect"]
IROperationType = Literal["create_sketch", "add_rectangle", "add_circle", "extrude"]


class IRMetadata(TypedDict, total=False):
    source: Literal["fusion", "studio", "test"]
    request_id: str
    iteration: int
    model: str
    session_id: str


@dataclass(frozen=True)
class CreateSketchParams:
    plane: IRPlane
    sketch: str


@dataclass(frozen=True)
class AddRectangleParams:
    sketch: str
    center: List[float]
    width: float
    height: float


@dataclass(frozen=True)
class AddCircleParams:
    sketch: str
    center: List[float]
    radius: float


@dataclass(frozen=True)
class ExtrudeParams:
    profile: str
    distance: float
    direction: IRDirection
    operation: IROperationMode
    sketch: Optional[str] = None
    profile_index: Optional[int] = None
    profile_indices: Optional[List[int]] = None


IRParams = Union[CreateSketchParams, AddRectangleParams, AddCircleParams, ExtrudeParams]


@dataclass(frozen=True)
class IROperation:
    id: str
    type: IROperationType
    params: IRParams
    dependencies: List[str] = field(default_factory=list)
    metadata: Optional[IRMetadata] = None


@dataclass(frozen=True)
class IRDocument:
    version: str
    units: Literal["mm"]
    operations: List[IROperation]
    metadata: Optional[Dict[str, Any]] = None
