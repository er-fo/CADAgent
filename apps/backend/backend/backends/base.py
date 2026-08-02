"""Shared backend target adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol

from ..ir.types import IRDocument, IROperation


@dataclass
class TargetExecutionResult:
    success: bool
    target: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    raw_result: Optional[Mapping[str, Any]] = None


class TargetAdapter(Protocol):
    """Protocol for target adapters behind shared IR."""

    target_name: str

    async def execute_operation(
        self,
        session_id: str,
        operation: IROperation,
        *,
        tool_use_id: str,
        description: str = "",
    ) -> TargetExecutionResult:
        ...

    async def execute_document(
        self,
        session_id: str,
        document: IRDocument,
        *,
        request_id: Optional[str] = None,
    ) -> TargetExecutionResult:
        ...
