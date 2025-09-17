#!/bin/bash

FUSION_ADDINS_DIR="$HOME/Library/Application Support/Autodesk/Autodesk Fusion 360/API/AddIns"
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="$FUSION_ADDINS_DIR/CADAgent"

# Get the current folder name (might be CADAgent-main, CADAgent-1.0.0, etc.)
CURRENT_FOLDER_NAME="$(basename "$SOURCE_DIR")"

mkdir -p "$FUSION_ADDINS_DIR"

if [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"
fi

# Copy the contents and ensure the target is named exactly "CADAgent"
cp -R "$SOURCE_DIR" "$TARGET_DIR"

# If we're not already in a folder named "CADAgent", rename the parent folder too
if [ "$CURRENT_FOLDER_NAME" != "CADAgent" ]; then
    PARENT_DIR="$(dirname "$SOURCE_DIR")"
    NEW_SOURCE_DIR="$PARENT_DIR/CADAgent"
    if [ "$SOURCE_DIR" != "$NEW_SOURCE_DIR" ] && [ ! -d "$NEW_SOURCE_DIR" ]; then
        mv "$SOURCE_DIR" "$NEW_SOURCE_DIR"
    fi
fi
