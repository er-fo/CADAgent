# Agent IR build123d Parity Contract

Date: 2026-05-05

## What Changed

- Added a Phase 0 parity contract for the Fusion-visible CAD tool surface in `backend/ir/capabilities.py`.
- Added Phase 1 golden tests in `backend/test_ir_parity_contract.py`.
- Changed the build123d translator to raise a typed capability error for unsupported IR operations.

## Why

The agent input → IR path already models the broad Fusion CAD tool surface, but build123d translation still supports only the sketch/extrude subset. Without a contract, new Fusion tools can silently widen the gap and make build123d replay unreliable.

## Architectural Decisions

- The parity matrix is static and explicit. Every Fusion-visible CAD tool must be classified as build123d `supported`, `unsupported`, or `fusion_only`.
- Planning/conversation tools are excluded from CAD parity because they do not represent persistent CAD operations.
- Unsupported build123d operations now fail through `Build123dCapabilityError` with the operation type and reason.
- Phase 1 tests verify tool-call mapping, IR validation, Fusion translation preservation, and build123d capability behavior.

## Current build123d Supported Surface

- `create_sketch`
- `add_rectangle`
- `add_circle`
- `add_line`
- `add_arc`
- `list_sketch_profiles`
- `extrude_profile`
- portable `create_construction_plane` datum/offset modes
- portable solid `revolve_profile` using construction axes and full/angle extents
- portable solid `create_loft` across ordered sketch/profile sections

## Phase 3/4/5 Review Status

### Phase 3: build123d solid feature parity

- The parity contract and golden tests now cover portable construction planes, solid revolve, and solid loft in the build123d translator.
- `revolve_profile` supports construction-axis full/angle solid revolves. Non-portable edge/face axes and to-entity extents still fail closed with typed capability errors.
- `create_loft` supports ordered solid loft sections that can be rebuilt from sketch/profile refs. Surface loft policy and guide/rail semantics remain outside the current portable contract.
- `apply_fillet`, `apply_chamfer`, `create_shell`, `create_simple_hole`, `create_counterbore_hole`, `create_tapped_hole`, and `create_external_thread` remain intentionally unsupported for build123d because they still depend on portable selector/replay semantics.

### Phase 4: entity registry and selectors

- Edge/face/body selection tools are classified as `fusion_only` in the static capability matrix because they manipulate Fusion UI/session state rather than persistent build123d geometry.
- The backend now persists IR selector/effect/target-result metadata through checkpoint restore, but that is not yet the same as a portable build123d entity registry.
- The current build123d gap is durable semantic re-selection and feature/body/face registry replay across topology-changing edits.

### Phase 5: feature lifecycle and timeline semantics

- `list_features`, `delete_feature`, `adjust_feature_parameters`, `suppress_feature`, `unsuppress_feature`, and `jump_to_timeline_position` are all modeled in shared IR and validated for Fusion execution.
- Those lifecycle/timeline operations remain intentionally unsupported for build123d replay until document revision/truncation semantics and feature-registry replay exist on the build123d side.
- The review confirmed that current workflow tests preserve the existing Fusion behavior while build123d fails closed with explicit capability reasons.

## Verification

- `python3 -m pytest -q backend/test_ir_parity_contract.py backend/test_ir_mapper_validator.py backend/test_ir_workflow_routing.py`
- `python3 -m compileall backend`

## Remaining Work

- Implement build123d translation for remaining selector-dependent solid operations beyond construction planes, extrude, revolve, and loft.
- Add durable topology selectors for edge/face/body-dependent operations.
- Define replay/revision semantics for Fusion timeline and feature lifecycle operations.

## Breaking Changes

None intended. Existing `ValueError` expectations still work because `Build123dCapabilityError` subclasses `ValueError`.
