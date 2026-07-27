# CAD Tool Failure Hardening - 2026-05-02

## What Changed

- Hardened MiniMax prompt-router parsing for wrapped, labeled, and conservatively truncated router outputs.
- Added runtime `extrude_profile` preflight for cached profile counts, selected profile indices, and target-body availability for `Join`, `Cut`, and `Intersect`.
- Tightened `create_pattern_feature` preparation so `auto_last` resolves only to a patternable timeline feature with body, bounds, or hole evidence.
- Enriched CAD failure summaries with selected body/face identity, profile counts/indices, resolved object types, and feature token diagnostics.
- Persisted successful `list_sketch_profiles` results into sketch metadata so later extrude preflight can validate profile selections.

## Why

AWS logs showed successful MiniMax usage with avoidable failures around invalid Fusion inputs:

- `extrude_profile` attempted invalid profile/operation combinations.
- `create_pattern_feature` retried `auto_last` against an invalid feature target.
- MiniMax router responses sometimes arrived as wrapped text, labeled text, or truncated JSON.
- Failure logs lacked enough resolved-object detail to diagnose issues quickly.

## Architectural Notes

- The changes are scoped to the legacy backend agent loop and router.
- Fusion execution remains the source of truth for CAD operations; the new checks prevent clearly invalid requests before Fusion receives them.
- Pattern preparation now treats `auto_last` as a convenience alias only after validating the resolved feature snapshot entry.
- Router fallback remains intact for genuinely unparseable output.

## Setup And Migration

- No database, deployment, or environment changes are required.
- Existing clients continue using the same tool names and request shapes.

## Validation

- `python3 -m compileall backend/agent_workflow.py backend/prompt_router.py backend/test_agent_workflow_hardening.py backend/test_prompt_router_emulation.py`
- `python3 -m pytest backend/test_prompt_routing.py backend/test_prompt_router_emulation.py backend/test_agent_workflow_hardening.py backend/test_extrude_profile_normalization.py backend/test_intent_retry_guards.py backend/test_ir_workflow_routing.py::test_execute_workflow_commits_resolved_pattern_feature_refs backend/test_target_adapters_ir.py::test_fusion_executor_fails_closed_for_pattern_auto_last_without_workflow_preparation`
