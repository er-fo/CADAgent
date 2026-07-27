# Timeline Feature Control Implementation

## What Changed

- Added backend schemas and execution routing for `suppress_feature` and `unsuppress_feature`.
- Added capability-aware `delete_feature` routing so newer add-ins receive the structured `feature_operation` payload path while older add-ins fall back to the existing codegen execution path instead of failing with "Unsupported feature operation."
- Added shared IR mapping and validation for suppression operations so the normal workflow path reaches Fusion execution.
- Expanded IR checkpoint/runtime serialization so `requires`, `effects`, `selectors`, `fallback_policy`, `target_results`, and `validation` survive persistence and restore.
- Added IR deserialization support for current mapped operation types, including `list_sketch_profiles`, `list_features`, `shell`, `create_simple_hole`, `delete_feature`, and suppression operations.
- Added Fusion target-adapter support for suppression IR so direct shared-IR execution can send suppress/unsuppress feature payloads.
- Added shared backend validation for feature-token timeline mutations.
- Updated timeline prompt guidance and router examples for suppress/restore requests.
- Added focused backend regression tests for suppression validation, capability gating, IR mapping, workflow execution, payload construction, and result formatting.
- Added checkpoint regression coverage so restored `list_sketch_profiles` results still satisfy later profile validation after resume/revert.
- Changed no-timeline revert results to chat-history rollback success instead of warning/error output.
- Kept live runtime geometry/entity state intact during chat-only rollback because Fusion geometry is not changed.
- Rejected operation-resume rollback when Fusion reports `geometry_reverted=False`, since operation resume requires matching model geometry.
- Replaced truthy string coercion with strict boolean parsing for capability flags, geometry rollback flags, and serialized suppression IR.
- Added a controlled unauthenticated execute rejection path so the backend returns a websocket auth error instead of surfacing a background task failure.
- When timeline refresh fails after delete/suppress/unsuppress/jump, the backend now asks Fusion for feature-snapshot diagnostics and reports whether the design is empty, in Direct Modeling mode, or simply stale.

## Why

The agent already had timeline inspection, deletion, and narrow parameter editing. This change closes the remaining gap between backend intent and add-in execution, keeps delete support compatible with older clients, and makes suppression a safer reversible option before destructive deletion.

## Design Notes

- `feature_token` remains the primary feature identity.
- `expected_name` and `expected_timeline_index` are optional guardrails.
- `expected_timeline_index` rejects boolean and negative values before execution.
- Suppress and unsuppress are treated as timeline/topology mutations so stale entity refs are cleared and refreshed.
- The tools and prompt text are exposed to the LLM only when the add-in request declares `client_capabilities.timeline_feature_suppression`.
- `delete_feature` uses the existing `feature_operation` transport only when the add-in declares `client_capabilities.timeline_feature_delete`; otherwise the backend reuses the codegen template path.
- `list_sketch_profiles` checkpoint results are now persisted in `target_results`, which lets restored IR state continue to validate later extrude/revolve operations correctly.
- Revert results with `geometry_reverted=False` now emit an info log and still trim conversation history.

## Setup And Migration

- No database, environment, or deployment configuration changes.
- The matching Fusion add-in worktree must include the `feature_tools.delete_feature` and `feature_tools.set_feature_suppression` implementations plus the `timeline_feature_delete` and `timeline_feature_suppression` capability declarations for runtime execution.
