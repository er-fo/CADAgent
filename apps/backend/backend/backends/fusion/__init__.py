"""Fusion target adapter package."""

from .executor import FusionTargetExecutor
from .translator import translate_ir_to_fusion_tool_call

__all__ = ["FusionTargetExecutor", "translate_ir_to_fusion_tool_call"]
