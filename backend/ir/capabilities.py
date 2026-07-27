"""Target capability contract for the shared CAD IR.

This module is the Phase 0 parity contract between the Fusion-visible agent
CAD tools, the shared IR, and the build123d target.  It is deliberately static:
adding a Fusion CAD tool must update this table and the contract tests before
execution can silently diverge across targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Literal, Mapping, Tuple

Build123dStatus = Literal["supported", "unsupported", "fusion_only"]
ToolCategory = Literal[
    "sketch",
    "profile_query",
    "solid_feature",
    "finishing_feature",
    "hole_thread_feature",
    "pattern_feature",
    "selection_state",
    "feature_lifecycle",
    "timeline_revision",
]


@dataclass(frozen=True)
class ToolCapability:
    """Parity status for one Fusion-visible CAD tool."""

    tool_name: str
    ir_operation: str
    category: ToolCategory
    build123d_status: Build123dStatus
    build123d_reason: str
    fusion_status: Literal["supported"] = "supported"


# These are planning/conversation tools, not persistent CAD operations.  They are
# excluded from the CAD parity matrix intentionally.
NON_CAD_AGENT_TOOLS: FrozenSet[str] = frozenset(
    {
        "generate_question_tree",
        "propose_designs",
        "output_build_plan",
    }
)


CAD_TOOL_CAPABILITIES: Dict[str, ToolCapability] = {
    "create_construction_plane": ToolCapability(
        tool_name="create_construction_plane",
        ir_operation="create_construction_plane",
        category="sketch",
        build123d_status="supported",
        build123d_reason="Portable datum and offset_from_datum construction planes translate to build123d Plane expressions; face/edge modes still fail closed.",
    ),
    "create_sketch": ToolCapability(
        tool_name="create_sketch",
        ir_operation="create_sketch",
        category="sketch",
        build123d_status="supported",
        build123d_reason="Translated to a build123d BuildSketch plane guard.",
    ),
    "add_circle": ToolCapability(
        tool_name="add_circle",
        ir_operation="add_circle",
        category="sketch",
        build123d_status="supported",
        build123d_reason="Translated to build123d Circle on the active sketch plane.",
    ),
    "add_line": ToolCapability(
        tool_name="add_line",
        ir_operation="add_line",
        category="sketch",
        build123d_status="supported",
        build123d_reason="Translated to build123d Line inside ordered closed BuildLine profiles.",
    ),
    "add_arc": ToolCapability(
        tool_name="add_arc",
        ir_operation="add_arc",
        category="sketch",
        build123d_status="supported",
        build123d_reason="Translated to build123d CenterArc inside ordered closed BuildLine profiles.",
    ),
    "add_rectangle": ToolCapability(
        tool_name="add_rectangle",
        ir_operation="add_rectangle",
        category="sketch",
        build123d_status="supported",
        build123d_reason="Translated to build123d Rectangle on the active sketch plane.",
    ),
    "add_sketch_geometry_batch": ToolCapability(
        tool_name="add_sketch_geometry_batch",
        ir_operation="add_line/add_arc/add_circle/add_rectangle",
        category="sketch",
        build123d_status="fusion_only",
        build123d_reason="Batch tool is a Fusion execution optimization; portable IR still stores each primitive as its existing sketch operation.",
    ),
    "list_sketch_profiles": ToolCapability(
        tool_name="list_sketch_profiles",
        ir_operation="list_sketch_profiles",
        category="profile_query",
        build123d_status="supported",
        build123d_reason="Translated to a deterministic sketch-plane existence guard; build123d does not expose Fusion profile indices.",
    ),
    "extrude_profile": ToolCapability(
        tool_name="extrude_profile",
        ir_operation="extrude",
        category="solid_feature",
        build123d_status="supported",
        build123d_reason="Translated to build123d extrude with target mode mapping.",
    ),
    "revolve_profile": ToolCapability(
        tool_name="revolve_profile",
        ir_operation="revolve",
        category="solid_feature",
        build123d_status="supported",
        build123d_reason="Portable solid revolves translate to build123d revolve for construction axes and full/angle extents.",
    ),
    "create_loft": ToolCapability(
        tool_name="create_loft",
        ir_operation="loft",
        category="solid_feature",
        build123d_status="supported",
        build123d_reason="Portable solid lofts translate to build123d loft across ordered sketch/profile sections.",
    ),
    "apply_fillet": ToolCapability(
        tool_name="apply_fillet",
        ir_operation="fillet",
        category="finishing_feature",
        build123d_status="supported",
        build123d_reason="Translated to build123d fillet when edge refs resolve to committed build123d edge selectors.",
    ),
    "apply_chamfer": ToolCapability(
        tool_name="apply_chamfer",
        ir_operation="chamfer",
        category="finishing_feature",
        build123d_status="supported",
        build123d_reason="Translated to build123d chamfer when edge refs resolve to committed build123d edge selectors.",
    ),
    "create_shell": ToolCapability(
        tool_name="create_shell",
        ir_operation="shell",
        category="finishing_feature",
        build123d_status="supported",
        build123d_reason="Translated to build123d offset shell for resolved face/body selectors with one-sided thickness.",
    ),
    "create_simple_hole": ToolCapability(
        tool_name="create_simple_hole",
        ir_operation="create_simple_hole",
        category="hole_thread_feature",
        build123d_status="supported",
        build123d_reason="Translated to build123d subtractive Hole for resolved global-Z planar face selectors.",
    ),
    "create_counterbore_hole": ToolCapability(
        tool_name="create_counterbore_hole",
        ir_operation="create_counterbore_hole",
        category="hole_thread_feature",
        build123d_status="supported",
        build123d_reason="Translated to build123d CounterBoreHole for resolved global-Z planar face selectors.",
    ),
    "create_tapped_hole": ToolCapability(
        tool_name="create_tapped_hole",
        ir_operation="create_tapped_hole",
        category="hole_thread_feature",
        build123d_status="supported",
        build123d_reason="Translated to build123d pilot-hole geometry plus explicit thread metadata for resolved global-Z planar face selectors.",
    ),
    "create_external_thread": ToolCapability(
        tool_name="create_external_thread",
        ir_operation="create_external_thread",
        category="hole_thread_feature",
        build123d_status="supported",
        build123d_reason="Recorded as explicit thread metadata on resolved face selectors; modeled helical thread geometry is intentionally not generated.",
    ),
    "create_pattern_feature": ToolCapability(
        tool_name="create_pattern_feature",
        ir_operation="pattern_feature",
        category="pattern_feature",
        build123d_status="supported",
        build123d_reason="Replays explicit committed simple-hole, counterbore-hole, and tapped-hole seed refs for rectangular global-axis and circular global-origin-Z patterns; other seed or axis forms fail closed.",
    ),
    "select_edges": ToolCapability(
        tool_name="select_edges",
        ir_operation="select_entities",
        category="selection_state",
        build123d_status="fusion_only",
        build123d_reason="Selection is Fusion UI/session state, not persistent build123d geometry.",
    ),
    "clear_edge_selection": ToolCapability(
        tool_name="clear_edge_selection",
        ir_operation="clear_selection",
        category="selection_state",
        build123d_status="fusion_only",
        build123d_reason="Selection clearing is Fusion UI/session state, not persistent build123d geometry.",
    ),
    "select_faces": ToolCapability(
        tool_name="select_faces",
        ir_operation="select_entities",
        category="selection_state",
        build123d_status="fusion_only",
        build123d_reason="Selection is Fusion UI/session state, not persistent build123d geometry.",
    ),
    "clear_face_selection": ToolCapability(
        tool_name="clear_face_selection",
        ir_operation="clear_selection",
        category="selection_state",
        build123d_status="fusion_only",
        build123d_reason="Selection clearing is Fusion UI/session state, not persistent build123d geometry.",
    ),
    "select_bodies": ToolCapability(
        tool_name="select_bodies",
        ir_operation="select_entities",
        category="selection_state",
        build123d_status="fusion_only",
        build123d_reason="Selection is Fusion UI/session state, not persistent build123d geometry.",
    ),
    "clear_body_selection": ToolCapability(
        tool_name="clear_body_selection",
        ir_operation="clear_selection",
        category="selection_state",
        build123d_status="fusion_only",
        build123d_reason="Selection clearing is Fusion UI/session state, not persistent build123d geometry.",
    ),
    "list_features": ToolCapability(
        tool_name="list_features",
        ir_operation="list_features",
        category="feature_lifecycle",
        build123d_status="unsupported",
        build123d_reason="Feature registry replay is not implemented for build123d yet.",
    ),
    "delete_feature": ToolCapability(
        tool_name="delete_feature",
        ir_operation="delete_feature",
        category="feature_lifecycle",
        build123d_status="unsupported",
        build123d_reason="Feature deletion requires build123d document revision/replay semantics before translation.",
    ),
    "adjust_feature_parameters": ToolCapability(
        tool_name="adjust_feature_parameters",
        ir_operation="adjust_feature_parameters",
        category="feature_lifecycle",
        build123d_status="unsupported",
        build123d_reason="Feature parameter edits require build123d document revision/replay semantics before translation.",
    ),
    "suppress_feature": ToolCapability(
        tool_name="suppress_feature",
        ir_operation="set_feature_suppression",
        category="feature_lifecycle",
        build123d_status="unsupported",
        build123d_reason="Feature suppression requires build123d document revision/replay semantics before translation.",
    ),
    "unsuppress_feature": ToolCapability(
        tool_name="unsuppress_feature",
        ir_operation="set_feature_suppression",
        category="feature_lifecycle",
        build123d_status="unsupported",
        build123d_reason="Feature unsuppression requires build123d document revision/replay semantics before translation.",
    ),
    "jump_to_timeline_position": ToolCapability(
        tool_name="jump_to_timeline_position",
        ir_operation="jump_to_timeline_position",
        category="timeline_revision",
        build123d_status="unsupported",
        build123d_reason="Fusion timeline jumps require portable IR revision/truncation semantics before build123d replay.",
    ),
}


SUPPORTED_BUILD123D_IR_OPERATIONS: FrozenSet[str] = frozenset(
    capability.ir_operation
    for capability in CAD_TOOL_CAPABILITIES.values()
    if capability.build123d_status == "supported"
)


def get_tool_capability(tool_name: str) -> ToolCapability:
    """Return the parity contract row for a Fusion-visible CAD tool."""

    try:
        return CAD_TOOL_CAPABILITIES[tool_name]
    except KeyError as exc:
        raise KeyError(f"No CAD IR capability contract exists for tool: {tool_name}") from exc


def get_build123d_capability_for_ir_operation(ir_operation: str) -> ToolCapability:
    """Return the build123d capability for an IR operation.

    Multiple Fusion tools can map to the same IR operation. For shared operation
    types such as selection and suppression, all rows intentionally carry the
    same build123d status; returning the first is sufficient for diagnostics.
    """

    for capability in CAD_TOOL_CAPABILITIES.values():
        if capability.ir_operation == ir_operation:
            return capability
    raise KeyError(f"No build123d capability contract exists for IR operation: {ir_operation}")


def cad_tool_names() -> FrozenSet[str]:
    """Return all Fusion-visible CAD tools covered by the parity contract."""

    return frozenset(CAD_TOOL_CAPABILITIES)


def missing_capability_rows(tool_names: Iterable[str]) -> Tuple[str, ...]:
    """Return CAD tool names that are missing from the parity contract."""

    return tuple(sorted(set(tool_names) - set(NON_CAD_AGENT_TOOLS) - set(CAD_TOOL_CAPABILITIES)))


def capability_summary() -> Mapping[str, ToolCapability]:
    """Expose a read-only view shape for docs/tests without copying rows."""

    return CAD_TOOL_CAPABILITIES
