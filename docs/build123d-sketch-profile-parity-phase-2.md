# build123d Sketch/Profile Parity Phase 2

Date: 2026-05-05

## What Changed

- Added build123d translation for `add_line` and `add_arc` IR sketch operations.
- Updated the build123d capability matrix so line and arc sketch primitives are `supported`.
- Changed sketch emission to materialize deterministic sketch geometry immediately before extrusion.
- Added tests for generated line/arc code, open-loop rejection, and real build123d extrusion of a closed line/arc profile.

## Why

Phase 2 requires build123d parity for non-rectangle/non-circle sketches. The previous adapter rejected line and arc IR even though Fusion could execute them, which prevented portable replay for common closed profiles.

## Architectural Decisions

- Line/arc geometry is emitted through build123d `BuildLine`, `Line`, `CenterArc`, and `make_face`.
- Ordered line/arc loops must be closed before extrusion. Open profiles fail during translation instead of failing later in generated code.
- Arc endpoints must be equidistant from the center so the center/start/end IR maps deterministically to `CenterArc`.
- Rectangles and circles remain supported as direct build123d sketch primitives.

## Breaking Changes

None intended. Invalid open line/arc extrusion now fails earlier with a deterministic `ValueError`.

## Remaining Gaps

- Multiple build123d profile-index selection remains limited by build123d/Fusion profile model differences.
- Later parity work now supports portable construction-plane datum/offset cases, solid revolve/loft, selector-resolved fillets/chamfers/shell/holes, and thread metadata; feature patterns and lifecycle replay remain unsupported.
