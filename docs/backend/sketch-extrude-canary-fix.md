# Sketch Extrude Canary Fix

## What Changed

- Added sketch display-name alias resolution so user-visible sketch names map back to canonical IR `sketch_id` values.
- Canonicalized sketch-bearing tool inputs before IR mapping and validation.
- Added a control-only `stop` fast path so stop/cancel commands do not enter the modeling LLM loop.
- Grouped repeated extrusion sketch-reference failures under one retry intent so retries stop cleanly instead of creating new sketches indefinitely.

## Why

The Pi Zero enclosure canary showed Fusion snapshots exposing sketch display names such as `Base Footprint`, while IR state tracked the original tool id such as `base_footprint`. The agent alternated between those identifiers, causing repeated `extrude_profile` preflight and IR validation errors.

## Impact

- Existing sketch geometry tools keep using canonical IR identifiers internally.
- Fusion-visible names remain usable as aliases.
- Stop requests remain chat/control actions and do not produce a fresh design response.
- No migration or setup changes are required.
