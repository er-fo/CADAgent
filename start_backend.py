#!/usr/bin/env python3
"""Local entry point for the CADAgent legacy backend."""

import os
import sys


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    backend_path = os.path.join(script_dir, "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    os.environ.setdefault("CADAGENT_DEV_MODE", "true")
    os.environ.setdefault("CADAGENT_AUTH_BYPASS", "true")
    os.environ.setdefault("BYPASS_SUPABASE_GATEWAY", "true")

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
