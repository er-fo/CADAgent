# API Call Batching Patch Notes - 2026-05-08

## What Changed

- Added `add_sketch_geometry_batch`, a narrow Fusion sketch batching tool for independent sketch primitives.
- The batch tool currently supports:
  - `add_line`
  - `add_arc`
  - `add_circle`
  - `add_rectangle`
- Exposed the batch tool through the sketch tool cluster so the model can create several same-sketch primitives in one assistant turn.
- Kept sketch creation, face-sketch orientation feedback, profile creation, and downstream 3D features outside the first batching scope.
- Added checkpoint support for successful sketch geometry batches so resume and recovery behavior matches existing single-operation execution.
- Hardened follow-up ordering so downstream tools such as `extrude_profile`, `revolve_profile`, and `create_loft` are deferred when they reference a sketch modified by a batch in the same model turn.
- Canonicalized sketch IDs for both the batch parent and child operations so aliases such as sketch names resolve to the committed sketch identity before execution.

## Why

Production-shaped logs showed that CADAgent was usually getting one tool call per model response, even though the backend could already process multiple tool calls in one response.

That made sketch-heavy requests expensive because every independent primitive forced another LLM iteration:

```text
Before:
create sketch -> observe
add line      -> observe
add circle    -> observe
add arc       -> observe
extrude       -> observe

After:
create sketch                  -> observe
add line + add circle + add arc -> observe
extrude                        -> observe
```

The change targets the safest high-confidence case first: independent sketch geometry on an already-created sketch. Topology-changing model operations remain guarded because they can invalidate face, edge, body, profile, and feature references.

## How It Works

- The model calls `add_sketch_geometry_batch` with one `sketch_id` and an ordered `operations` array.
- The backend validates that every child operation is one of the supported sketch primitives.
- The backend translates the child operations into one Fusion `execute_code` payload.
- Fusion executes the batch and returns compact per-operation results.
- The backend registers successful child entities in the sketch entity store and appends the corresponding IR operations.
- The LLM receives one compact summary instead of several separate tool observations.

The implementation is intentionally conservative:

- It does not batch geometry onto a face-backed sketch created in the same turn, because Fusion orientation feedback must be observed first.
- It does not batch sketch geometry with profile-consuming 3D operations in the same turn.
- It rejects unsupported child operations rather than partially broadening the tool contract.
- It creates operation checkpoints only after Fusion reports committed batch work.

## Token Savings

The token savings come from avoiding model iterations, not from reducing the Fusion websocket payload itself.

The current planning baseline in `docs/free-tier-economics-2026-05.md` measured an average CADAgent iteration at about:

- `3,761` input tokens
- `452` output tokens
- `4,213` total tokens per iteration

For a batch with `N` independent sketch primitive operations, the geometry section changes from `N` model iterations to `1` model iteration.

Estimated savings for the sketch-geometry section:

| Batched sketch primitives | Avoided model iterations | Approx tokens saved | Geometry-section reduction |
| ---: | ---: | ---: | ---: |
| 2 | 1 | `~4.2k` | `~50%` |
| 3 | 2 | `~8.4k` | `~67%` |
| 4 | 3 | `~12.6k` | `~75%` |
| 6 | 5 | `~21.1k` | `~83%` |
| 8 | 7 | `~29.5k` | `~88%` |

Estimated savings for a full simple sketch-to-feature flow, including one required `create_sketch` turn and one downstream feature turn:

| Batched sketch primitives | Before | After | Avoided model iterations | Approx full-flow reduction |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 4 turns | 3 turns | 1 | `~25%` |
| 3 | 5 turns | 3 turns | 2 | `~40%` |
| 4 | 6 turns | 3 turns | 3 | `~50%` |
| 6 | 8 turns | 3 turns | 5 | `~63%` |
| 8 | 10 turns | 3 turns | 7 | `~70%` |

These are estimates, not a production billing measurement. Actual savings vary with prompt size, selected model, tool-result verbosity, and whether the model naturally emits all eligible primitives in one batch. The strongest win is on sketch-heavy designs where several independent primitives belong to the same already-created sketch.

## Validation

- Local full test suite on the feature branch: `445 passed, 2 xfailed`.
- Local full test suite on the exact integration merge: `445 passed, 2 xfailed`.
- `python3 -m compileall backend` passed.
- `git diff --check` passed.
- Fusion add-in emulator smoke passed through the live backend websocket path:
  - authenticated the emulator session
  - created a sketch
  - executed one `add_sketch_geometry_batch` payload
  - observed batch checkpoint creation
  - verified same-turn `extrude_profile` was deferred
  - completed the request successfully
- GitHub Actions deployment run `25581193263` passed.
- CodeDeploy deployment `d-OKNMJOZAH` succeeded.
- Live health check passed at `https://ws.cadagentpro.com/health`.

## Deployment Notes

- No database migration is required.
- No add-in update is required for clients that already consume backend-driven `execute_code` payloads.
- The change is backward compatible with existing single-tool sketch operations.
- The backend remains the source of truth for dependency gating and operation checkpointing.
