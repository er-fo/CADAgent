"""Export helpers for build123d shapes."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_step_file(part: Any, output_path: str) -> str:
    """Export part to STEP and return absolute path."""
    from build123d import export_step  # Imported lazily to avoid hard failure at module import.

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    export_step(part, str(path))
    return str(path)
