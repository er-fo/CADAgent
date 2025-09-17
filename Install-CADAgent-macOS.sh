#!/bin/bash

FUSION_ADDINS_DIR="$HOME/Library/Application Support/Autodesk/Autodesk Fusion 360/API/AddIns"
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="$FUSION_ADDINS_DIR/CADAgent"

mkdir -p "$FUSION_ADDINS_DIR"

if [ -d "$TARGET_DIR" ]; then
    rm -rf "$TARGET_DIR"
fi

cp -R "$SOURCE_DIR" "$TARGET_DIR"
