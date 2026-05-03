# STEP Reference Attachment UI

## What Changed

The add-in palette accepts `.step` and `.stp` files as attachments and labels
them as `STEP` in the attachment preview.

## Why

The backend can parse STEP references into neutral geometry context, but users
need the palette to allow those files through the existing attachment bridge.

## Architecture

The UI sends STEP files as generic attachments with `kind: cad_reference`. The
backend owns parsing and prompt-context generation.

## Setup Notes

No add-in dependency changes are required.
