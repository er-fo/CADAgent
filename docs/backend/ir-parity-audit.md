# IR Parity Audit For Existing Fusion Tool Set

Date: 2026-05-01

## Update After IR Widening

Status: this audit was the starting-state assessment. The implementation now widens the shared IR beyond the original sketch/extrude MVP and resolves several critical mismatches called out below.

Resolved or materially improved:

- The IR operation union now includes construction planes, sketch lines/arcs, profile inspection, revolve, loft, fillet, chamfer, shell, hole/thread tools, feature patterns, feature listing, timeline edits, and selection actions.
- Fusion-facing centimeter inputs are normalized into canonical IR millimeters, and the Fusion adapter converts canonical millimeters back to centimeters at execution boundaries.
- Extrude `Intersect` is now accepted consistently by the Fusion code generator and tool schema.
- Normal workflow execution now records and validates IR for feature and selection tools, while preserving the existing Fusion reference-resolution guardrails.
- build123d/studio now translates portable construction-plane datum/offset cases, solid revolve, and solid loft, and fails explicitly for unsupported advanced operations instead of falling through into Fusion websocket execution.
- Committed IR state is now session-scoped rather than request-local, so follow-up requests can validate against prior successful operations and continue monotonic operation IDs.
- Closed line/arc loops now require `list_sketch_profiles` confirmation before extrusion; the prompt no longer claims an inferred profile index that the validator cannot prove.
- Direct Fusion IR execution preserves custom construction-plane identifiers such as `plane_0` while still failing closed on unresolved face aliases.
- IR validation rejects non-finite numeric values (`NaN`/`Infinity`) across geometry, feature, thread, and pattern parameters.
- Face-sketch UV bounds preflight now covers line and arc geometry as well as rectangles and circles.

Remaining important gaps:

- build123d directly supports the sketch/extrude subset, portable datum/offset construction planes, construction-axis solid revolve, solid loft, ordered closed line/arc profiles, profile-inspection guards, selector-resolved finishing/shell/hole features, and thread metadata.
- Topology refs are represented and invalidated at a coarse level; feature-level replay for patterns and lifecycle operations remains limited.
- Fusion and build123d result topology is captured through target results/runtime refresh and persistent IR entity registry snapshots.
- Direct Fusion IR execution for feature refs requires already-resolved tokens or an entity store; workflow execution remains the preferred path for ref-heavy feature operations.

Phase 3/4/5 review note (2026-05-05):

- Contract coverage in `apps/backend/backend/test_ir_parity_contract.py` and workflow/validator coverage in `apps/backend/backend/test_ir_mapper_validator.py` plus `apps/backend/backend/test_ir_workflow_routing.py` verifies the supported portable build123d subset while preserving explicit unsupported/fusion-only behavior for non-portable pattern seeds and lifecycle/timeline semantics.

### Current Post-Widening Matrix

| Area | Current IR Status | Fusion Adapter | build123d Adapter |
| --- | --- | --- | --- |
| Construction planes | Modeled with datum, offset, angle-to-edge, and face-normal params | Translates to existing Fusion codegen and resolves entity refs | Datum and offset_from_datum supported; face/edge-relative modes fail closed pending portable selectors |
| Sketch primitives | Sketches, rectangles, circles, lines, arcs, and profile queries are modeled | Translates to Fusion codegen with canonical mm-to-cm conversion | Rectangles/circles supported; ordered closed line/arc profiles emit `BuildLine`/`make_face`; extrudes materialize deterministic build123d sketches |
| Solid features | Extrude, revolve, loft, fillet, chamfer, shell, holes, threads, patterns modeled | Translates to Fusion codegen or feature payloads | Extrude, portable solid revolve/loft, selector-resolved fillet/chamfer/shell/holes, thread metadata, and replayable hole-like feature patterns supported; non-hole pattern seeds fail closed |
| Selection and timeline | Selection, feature listing, delete, and jump operations modeled | Routed to Fusion payload/codegen paths | Explicitly unsupported |
| Units | IR is canonical millimeters | Converts to Fusion centimeters at the boundary | Consumes millimeters |
| Results and refs | Operations can carry target execution results, effects, selectors, registry refs, and invalidation metadata | Runtime refresh remains the source of detailed topology refs | build123d extraction emits deterministic body/face/edge metadata |

## Historical Pre-Widening Result

Pre-widening finding: the shared IR was not a reliable source of truth for CADAgent design intent. Fusion execution was ahead of IR by a wide margin.

At the time of this audit, the IR represented a small sketch/extrude MVP:

- `create_sketch`
- `add_rectangle`
- `add_circle`
- `extrude`

The Fusion-facing tool surface exposes a much broader CAD feature set:

- construction planes
- sketch lines and arcs
- profile inspection
- extrude, revolve, loft
- fillet, chamfer, shell
- simple holes, counterbores, tapped holes, external threads
- rectangular and circular feature patterns
- entity selection
- feature listing
- timeline deletion and rewind

This meant CADAgent could not yet plan, validate, replay, export, compare, or retarget designs from a shared representation. At that point the Fusion execution layer was the de facto semantic source of truth for most meaningful CAD operations.

## Evidence

Primary code points audited:

- `apps/backend/backend/ir/types.py`: IR operation types and parameter dataclasses.
- `apps/backend/backend/ir/mapper.py`: planner/runtime tool call to IR mapping.
- `apps/backend/backend/ir/validator.py`: IR validation and committed dependency checks.
- `apps/backend/backend/backends/fusion/translator.py`: IR to Fusion tool call translation.
- `apps/backend/backend/backends/build123d/translator.py`: IR document to build123d code translation.
- `apps/backend/backend/agent_workflow.py`: execution routing, IR gating, feature-operation handling, entity refresh.
- `apps/backend/backend/code_generator.py`: code-generated Fusion operation templates and parameter validation.
- `apps/backend/backend/llm_client.py`: full LLM-visible Fusion tool schemas.
- `apps/backend/backend/prompt_structure.py`: routed tool clusters and documented CAD behavior.

## Layer Definitions

### Fusion tool layer

The Fusion layer is the operational surface exposed to the model. It has two execution paths:

- Code-generated Python operations handled by `apps/backend/backend/code_generator.py`.
- Feature-operation payloads sent through `apps/backend/backend/agent_workflow.py` to the Fusion add-in.

This layer knows about Fusion-specific refs, entity tokens, sketches, profiles, feature tokens, timeline indices, and add-in feature operations.

### IR layer

The IR layer is intended to be the shared, target-independent representation. It should encode design intent, dependencies, stable references, parameters, validation rules, target capability requirements, and fallback strategies.

At the time of the pre-widening audit it only covered a small subset of sketch and extrude workflows.

### Target adapter layer

The target adapters consume IR:

- Fusion adapter translates one IR operation back into Fusion tool calls.
- build123d adapter translates the whole IR document into build123d code.

Because the pre-widening IR was small, build123d had only MVP coverage and Fusion was still needed for most real CAD work.

## Pre-Widening IR Surface

Pre-widening operation type union:

```python
IROperationType = Literal["create_sketch", "add_rectangle", "add_circle", "extrude"]
```

Pre-widening IR parameter objects:

- `CreateSketchParams`
- `AddRectangleParams`
- `AddCircleParams`
- `ExtrudeParams`

Pre-widening document model:

- `version`
- `units`, fixed to `"mm"`
- ordered list of operations
- optional metadata

Pre-widening dependency model:

- each operation has an explicit `dependencies: List[str]`
- validation enforces dependencies must point to committed operations
- sketch geometry depends on `create_sketch`
- extrude depends on latest sketch profile operation plus sketch creation

This was a useful skeleton, but it did not yet model CAD topology, durable entity identity, feature ownership, timeline edits, query operations, or manufacturing feature semantics.

## Historical Fusion Tool Inventory And IR Gap Matrix

The following matrix is the original pre-widening gap inventory. It is preserved as implementation history, not as the current status table.

Status key:

- `Covered`: IR has a direct operation and target translators.
- `Partial`: IR has a related operation but loses important semantics.
- `Missing`: no IR operation exists.
- `Procedural`: tool affects UI/session state or inspection more than persistent design geometry, but still needs an IR/query model if planning, replay, and validation should be deterministic.

| Fusion tool | Pre-widening IR status | Mechanical addition needed | Semantic addition needed |
|---|---:|---|---|
| `create_construction_plane` | Missing | Add `create_plane` / `create_construction_plane` params for datum, offset, angle-to-edge, face-normal modes. | Plane identity, coordinate frame, origin, normal, u/v axes, parent face/edge refs, offset/angle units, plane reuse/no-op semantics. |
| `create_sketch` | Partial | Extend params with sketch name, face-backed plane refs, resolved plane token, plane frame. | Sketch must reference a stable `PlaneRef`, not raw string only. It should store orientation feedback and face UV bounds when sketching on model faces. |
| `add_circle` | Partial | Preserve optional circle id/alias, created curve token, center point token. | Need sketch entity graph: curves, points, aliases, constraints, closed-loop contribution, units. |
| `add_line` | Missing | Add line params: sketch, start uv, end uv, optional alias. | Needed for arbitrary profiles. Must create sketch point refs and curve refs, support loop closure and profile inference. |
| `add_arc` | Missing | Add arc params: center, start, end, optional alias. | Must validate nondegenerate geometry, radius consistency, orientation, profile contribution. |
| `add_rectangle` | Partial | Preserve rectangle id/alias and created line/point refs. | Rectangle is currently reduced to center/width/height, losing original corner intent, line entities, and profile topology. |
| `list_sketch_profiles` | Missing / Procedural | Add query op or derived `profiles` section on sketches. | Profiles must become first-class refs with areas, loop counts, centroids, inner/outer loops, and stability rules. |
| `extrude_profile` | Partial | Extend extrude params with feature name, extent variants, target bodies, start offsets, taper, created entities, actual distance. | Existing IR has unit/sign issues, no created-feature ref, no body/face/edge outputs, no retry/fallback record, and inconsistent `intersect` capability. |
| `revolve_profile` | Missing | Add revolve params: profile ref/index, axis union, extent union, operation, solid/surface, occurrence, feature name. | Axis must be a semantic ref or construction axis. Need profile-axis clearance validation, extent-to-entity refs, created topology, and target capability flags. |
| `create_loft` | Missing | Add loft params: ordered section profile refs, operation, solid/surface preference, tangent merge, feature name. | Preserve ordered cross-section intent, fallback from solid to surface, profile compatibility validation, guide/rail extensibility. |
| `apply_fillet` | Missing | Add fillet params: edge refs/selectors, radius, unit, tangent-chain flag, feature name. | Need durable edge selection semantics. Edge refs are topology-unstable, so IR must support semantic selectors and invalidation/reselection rules. |
| `apply_chamfer` | Missing | Add chamfer params: edge refs/selectors, distance, unit, tangent-chain flag, feature name. | Need chamfer type extensibility: equal distance now, distance-angle/two-distance later. Same topology stability issue as fillets. |
| `create_shell` | Missing | Add shell params: mode open/closed, face refs/body refs, inside/outside thickness, unit, tangent chain, shell type. | Must distinguish removed faces from hollowed bodies. Validate solids, thickness feasibility, all-face removal, face/body exclusivity, shell-before-fillet ordering. |
| `create_simple_hole` | Missing | Add hole params: face ref/selector, center point, diameter, extent type, depth, unit, feature name. | Hole must be a manufacturing feature, not sketch-cut fallback. Need face-plane lock, bounds checks, normal direction, material clearance, through-all semantics. |
| `create_counterbore_hole` | Missing | Add counterbore params: face, center, hole diameter/depth, counterbore diameter/depth, unit, feature name. | Need stepped-hole semantics, diameter ordering validation, counterbore depth validation, fastener intent. |
| `create_tapped_hole` | Missing | Add tapped-hole params: face, center, thread standard, thread size, thread depth, pilot depth, unit, feature name. | Thread catalog validation belongs in IR. Need tap drill data, pilot-depth relationship, manufacturing metadata, supported-size fallback. |
| `create_external_thread` | Missing | Add external-thread params: cylindrical face ref, thread type/size, length, offset, full-length flag, unit, feature name. | Need cylindrical-face validation, nominal diameter compatibility, length/offset validation, cosmetic vs modeled thread policy. |
| `create_pattern_feature` | Supported for explicit hole seeds | Add pattern params: pattern type, seed feature refs, counts, spacing, axes, rotation count/angle, feature name. | build123d now replays committed simple-hole, counterbore-hole, and tapped-hole seed refs for global-axis/global-origin patterns. Raw Fusion feature tokens, `auto_last`, non-hole seeds, and oriented axes remain non-portable and fail closed. |
| `select_edges` | Missing / Procedural | Add selection query/action or semantic selector model. | Selection should not be the design truth. IR needs selectors like "top perimeter edges of body_0" that can resolve per target/kernel. |
| `clear_edge_selection` | Missing / Procedural | Add session action only if UI replay matters. | Should usually stay outside persistent design IR unless transcript replay requires it. |
| `select_faces` | Missing / Procedural | Add selection query/action or semantic selector model. | Same as edge selection. Face refs are topology-unstable and must be regenerated after mutations. |
| `clear_face_selection` | Missing / Procedural | Add session action only if UI replay matters. | Same as clear edge selection. |
| `select_bodies` | Missing / Procedural | Add selection query/action or semantic selector model. | Body refs are more stable than faces/edges but still need semantic identity and target resolution. |
| `clear_body_selection` | Missing / Procedural | Add session action only if UI replay matters. | Same as other clear-selection tools. |
| `list_features` | Missing / Procedural | Add feature-query op or document state snapshot. | IR needs feature registry with names, types, tokens, timeline indices, created bodies, and validity status. |
| `jump_to_timeline_position` | Missing | Add destructive timeline operation or revision operation. | This cannot be a normal geometry feature. It invalidates later ops and entity refs. IR needs document revisions/branches or explicit truncation semantics. |
| `delete_feature` | Missing | Add delete/suppress feature op by stable feature ref plus safety checks. | Must model feature lifecycle, expected name/index safety checks, downstream dependency invalidation, and target timeline capability. |

Excluded from CAD IR parity:

- `respond_to_user`
- `generate_question_tree`
- `propose_designs`
- `output_build_plan`

These are conversation/design-planning tools, not persistent CAD operations. Their outputs can reference IR but should not become geometry operations.

## Mechanical Additions Required

### 1. Expand operation type system

Add operation families rather than a flat sketch-only enum:

```text
document
  create_plane
  create_sketch
  add_sketch_curve
  inspect_profiles
  create_feature
  modify_feature
  query_entities
  select_entities
  edit_timeline
```

Concrete MVP-plus operation types:

- `create_construction_plane`
- `add_line`
- `add_arc`
- `list_sketch_profiles` or `derive_sketch_profiles`
- `revolve`
- `loft`
- `fillet`
- `chamfer`
- `shell`
- `create_simple_hole`
- `create_counterbore_hole`
- `create_tapped_hole`
- `create_external_thread`
- `pattern_feature`
- `list_features`
- `delete_feature`
- `jump_to_timeline_position`

### 2. Add stable reference types

IR needs typed refs instead of free-form strings:

```text
PlaneRef
SketchRef
SketchCurveRef
SketchPointRef
ProfileRef
BodyRef
FaceRef
EdgeRef
VertexRef
FeatureRef
OccurrenceRef
SelectionRef
```

Each ref should store:

- stable IR id
- target-specific handles, if known
- source operation id
- semantic label/alias
- geometric fingerprint where relevant
- validity state
- generation number after topology changes

### 3. Add output contracts per operation

Operations must declare what they create or mutate:

```text
creates:
  features: [...]
  bodies: [...]
  faces: [...]
  edges: [...]
  profiles: [...]
modifies:
  bodies: [...]
invalidates:
  refs: [...]
```

At the time of the audit, Fusion execution collected created bodies/faces/edges in runtime results for extrude/revolve/loft, but IR did not retain those outputs.

### 4. Normalize units

The pre-widening IR said documents use `mm`, but Fusion schemas mixed units:

- sketch coordinates: cm
- extrude distance: cm
- hole center coordinates: mm
- hole depths: mm
- fillet/chamfer radius/distance: default mm
- thread length/offset for external threads: cm

IR must either:

- store all lengths in canonical millimeters, plus explicit source-unit metadata, or
- store typed length values with `{value, unit}` everywhere and normalize only at target translation.

The first option is better for validation, comparison, and benchmark scoring.

### 5. Add target capability metadata

Each operation should declare required target capabilities:

```text
requires:
  - b_rep_kernel
  - parametric_timeline
  - hole_feature
  - thread_catalog
  - shell_feature
  - feature_pattern
```

Adapters can then report:

- supported directly
- supported through fallback
- unsupported
- lossy translation

### 6. Add IR serialization and migration rules

The current dataclass model is fine for MVP tests, but parity needs a versioned schema with migrations.

Required:

- schema version per document
- operation version per operation type
- backward-compatible readers
- stable JSON shape
- unknown-operation handling
- lossiness annotations when imported from Fusion

## Semantic Additions Required

### 1. Design intent must be separate from kernel handles

Fusion entity tokens are opaque and topology-dependent. They are useful execution handles, not design intent.

IR should represent intent like:

- "fillet all top perimeter edges of base body"
- "create M6 tapped hole on the right mounting face at local coordinates"
- "shell this enclosure by removing the top face with 2 mm wall thickness"

Then target adapters resolve intent into current topology.

### 2. Topology refs must be invalidated after mutations

The current workflow already knows topology mutations require refresh:

- extrude
- revolve
- loft
- fillet
- chamfer
- shell
- holes
- threads
- patterns
- timeline edits

IR must encode that these operations invalidate face/edge/body refs and require re-resolution before downstream operations.

### 3. Features need lifecycle semantics

IR needs first-class features:

- feature id
- feature type
- timeline position
- feature name
- source operation id
- created/modified bodies
- status: active, suppressed, deleted, failed, replaced
- target handles

Without this, delete, pattern, benchmark replay, comparison, and fallback cannot be reliable.

### 4. Profiles need first-class identity

Extrude, revolve, and loft relied on sketch profile indices. Profile indices are target-generated and can shift when sketch geometry changes.

IR should store:

- profile ref
- owning sketch
- loops
- loop orientation
- constituent sketch curves
- centroid
- area
- inner/outer classification
- target profile index after resolution

### 5. Selection should become semantic selectors

Selection tools are procedural, but the reason they exist is semantic: choose entities by spatial and topological properties.

IR selectors should support:

- by explicit ref
- by owning feature
- by owning body
- by face normal/orientation
- by centroid/bounds
- by edge adjacency
- by tangent chain
- by loop/perimeter
- by created-by operation

Example:

```text
selector:
  kind: edge
  body: base_body
  adjacent_face:
    normal: +Z
  chain: tangent
```

This is the only scalable way to target multiple kernels.

### 6. Fallbacks must be explicit

Fusion code had implicit fallback behavior, such as:

- extrude cut/intersect can retry with flipped distance
- loft can fall back from solid to surface
- pattern axes and spacing can be inferred
- circular pattern is limited to global-origin axes

IR should record fallback policy and actual outcome:

```text
requested:
  distance: -10 mm
fallbacks:
  - flip_extrude_direction_on_no_target
actual:
  distance: 10 mm
  fallback_used: true
```

This makes validation and replay honest.

## Pre-Widening Critical Mismatches

### Unit mismatch

IR documents declared `mm`, but Fusion sketch/extrude tools used centimeters. The mapper transferred raw numeric values into IR without conversion. That meant build123d could reinterpret Fusion centimeters as millimeters.

This must be fixed before IR becomes a canonical geometric source of truth.

### Extrude operation mismatch

IR allows `intersect` in `IROperationMode`. Fusion translator maps it to `"Intersect"`. But code-generated Fusion extrude validation only accepts:

- `NewBody`
- `Join`
- `Cut`

So IR can represent an extrude operation that the current Fusion codegen path rejects.

### Lost feature identity

Fusion tools accept `feature_name` for many operations. IR drops it for current mapped operations.

Without feature identity:

- user-facing timeline names are not reproducible
- non-hole feature patterns cannot robustly reference seed features
- deletion/suppression cannot be modeled safely
- benchmark comparison loses semantic landmarks

### Lost created topology

Fusion results can include created bodies/faces/edges. IR does not store created topology.

This prevents downstream IR operations from depending on "the face created by op_3" or "the cylindrical face of the shaft."

### Query and inspection are outside IR

`list_sketch_profiles` and `list_features` are runtime tools only. IR needs either query operations or maintained document state snapshots so profile and feature references can be replayed and verified.

## Proposed Target IR Shape

This is not implementation code. It is the shape IR should grow toward.

```text
IRDocument
  version
  canonical_units
  operations[]
  entities
    planes
    sketches
    sketch_curves
    profiles
    features
    bodies
    faces
    edges
  metadata
```

Each operation:

```text
IROperation
  id
  type
  params
  dependencies
  selectors
  creates
  modifies
  invalidates
  validation
  fallback_policy
  target_results
  metadata
```

Each target result:

```text
TargetResult
  target
  success
  operation_handle
  created_refs
  modified_refs
  invalidated_refs
  warnings
  fallback_used
  lossiness
  raw_handles
```

## Priority Roadmap

### Phase 1: Make current MVP honest

Goal: prevent the existing IR subset from lying.

Add:

- canonical unit conversion
- sketch name and feature name preservation
- created topology outputs for extrude
- explicit operation capability checks
- fixed extrude `intersect` handling
- profile refs as first-class objects
- regression tests proving Fusion cm inputs become canonical IR mm values

### Phase 2: Complete sketch and profile parity

Goal: represent all current sketch/profile workflows.

Add:

- construction planes
- line and arc sketch curves
- sketch entity refs
- sketch point refs
- profile inspection/derivation
- loop/profile identity
- face-sketch orientation and bounds metadata

### Phase 3: Add core solid feature parity

Goal: represent all current solid creation tools.

Add:

- enhanced extrude
- revolve
- loft
- feature refs
- target-created body/face/edge refs
- adapter capability reports

### Phase 4: Add BRep modification parity

Goal: represent modifications that depend on existing topology.

Add:

- semantic selectors
- fillet
- chamfer
- shell
- topology invalidation and re-resolution

### Phase 5: Add manufacturing feature parity

Goal: preserve design/manufacturing intent.

Add:

- simple holes
- counterbore holes
- tapped holes
- external threads
- thread catalog validation in IR
- hole placement validation
- manufacturing metadata

### Phase 6: Add feature pattern and timeline parity

Goal: support replay, deletion, revision, and patterning.

Add:

- rectangular feature pattern
- circular feature pattern with explicit global-origin limitation
- explicit repeated-feature fallback
- list/query feature snapshots
- delete feature
- jump timeline / truncate document
- operation branch/revision semantics

## Recommended Definition Of Done For Parity

IR is "up to par" with the existing Fusion tool set when all of the following are true:

1. Every Fusion CAD tool has either a direct IR operation or an explicit reason it is treated as non-persistent UI/session state.
2. Every IR operation translates to Fusion without semantic loss, or records lossiness explicitly.
3. Every IR operation can report whether build123d supports it, falls back, or rejects it.
4. Units are canonical and target translators perform all target-specific conversion.
5. Feature outputs are captured as IR refs.
6. Topology-mutating operations invalidate dependent refs.
7. Edge/face/body-dependent operations use semantic selectors, not stale raw refs only.
8. Queries such as profiles and features are reproducible as document state or query operations.
9. Timeline-destructive operations are modeled as revisions/truncations, not normal features.
10. Tests cover each Fusion tool's map-to-IR, validate-IR, IR-to-Fusion, unsupported-target behavior, and at least one replay scenario.

## Historical Bottom Line

The pre-widening IR was a useful MVP gate for sketch/extrude execution, but it was not yet a design representation. To get parity with Fusion, the project needed to promote IR from "operation list" to "parametric feature graph with typed refs, selectors, topology invalidation, unit normalization, and target capability metadata."

That is the right architectural move. Without it, every new Fusion tool will deepen the gap and make multi-kernel targeting, replay, validation, and benchmark scoring less trustworthy.
