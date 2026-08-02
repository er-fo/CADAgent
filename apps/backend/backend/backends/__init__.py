"""Shared target adapters behind backend IR."""

from .base import TargetExecutionResult
from .build123d import Build123dTargetExecutor
from .fusion import FusionTargetExecutor

__all__ = [
    "Build123dTargetExecutor",
    "FusionTargetExecutor",
    "TargetExecutionResult",
]
