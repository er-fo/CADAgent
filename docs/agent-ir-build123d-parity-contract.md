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
- `list_sketch_profiles`
- `extrude_profile`

## Remaining Work

- Implement build123d translation for portable geometry operations.
- Add durable topology selectors for edge/face/body-dependent operations.
- Define replay/revision semantics for Fusion timeline and feature lifecycle operations.

## Breaking Changes

None intended. Existing `ValueError` expectations still work because `Build123dCapabilityError` subclasses `ValueError`.
