# Timeline Feature Control Implementation Plan

## Goal

Give CADAgent safe, agent-facing control over existing Fusion 360 timeline features:
inspect, rename/edit supported parameters, suppress, unsuppress, and delete features.

## Scope

- Extend existing `list_features`, `delete_feature`, and `adjust_feature_parameters`.
- Add `suppress_feature` and `unsuppress_feature`.
- Keep edits feature-token based with optional expected name/index safety checks.
- Refresh feature context after successful timeline mutations.

## Non-Goals

- No arbitrary timeline object mutation.
- No sketch-geometry editing in the first pass.
- No timeline reorder/move support.
- No direct EC2 or production deployment in this task.

## Backend Tasks

1. Add tool schemas for `suppress_feature` and `unsuppress_feature` in `backend/llm_client.py`.
2. Add backend validation and payload construction in `backend/agent_workflow.py`.
3. Update prompt routing/docs in `backend/prompt_structure.py` so edit/delete/suppress requests select timeline tools.
4. Ensure successful timeline mutations clear or refresh cached feature snapshots.
5. Add focused tests for validation, routing, stale-token messaging, and result formatting.

## Add-In Tasks

1. Add `suppress_feature` and `unsuppress_feature` handling in `CADAgent.py` feature operation dispatch.
2. Add helpers in `feature_tools.py` to resolve feature tokens, validate identity, toggle suppression, recompute, and serialize updated state.
3. Keep `capture_feature_snapshot` suppression and editable-parameter metadata consistent.
4. Return clear errors for direct modeling mode, stale tokens, ambiguous tokens, unsupported feature types, and recompute failures.

## Safety Contract

- Timeline tools require Parametric Design Mode.
- `feature_token` is the primary identity.
- `expected_name` and `expected_timeline_index` are guardrails, not lookup keys.
- Ambiguous token resolution aborts.
- Expected-name or expected-index mismatch aborts.
- Mutation attempts call `design.computeAll()` and report recompute status.
- Agent should prefer suppression before deletion when user intent is reversible or ambiguous.

## Acceptance Criteria

- Agent can inspect timeline features and see editable/suppression metadata.
- Agent can suppress and unsuppress a feature safely.
- Agent can still delete a feature with existing identity safeguards.
- Agent can edit supported ExtrudeFeature and HoleFeature parameters.
- Stale/ambiguous/unsupported targets fail with clear guidance.
- Backend tests cover new schemas, validation, routing, and result formatting.
- Add-in logic is documented and smoke-tested locally where Fusion availability allows.

## Implementation Order

1. Backend schemas, prompt guidance, and validation tests.
2. Add-in suppress/unsuppress helpers and dispatch.
3. Backend/add-in result formatting and snapshot refresh.
4. Regression tests for existing list/edit/delete behavior.
5. Manual smoke test in Fusion if available.
