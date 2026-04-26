"""Fusion target adapter execution behind shared IR."""

from __future__ import annotations

import asyncio
from typing import Optional

from ...code_generator import CodeGenerationError, format_error_for_llm, translate_tool_call
from ...ir.types import IRDocument, IROperation
from ...websocket_manager import ConnectionManager
from ..base import TargetExecutionResult
from .translator import translate_ir_to_fusion_tool_call


class FusionTargetExecutor:
    """Execute IR operations by reusing the current Fusion execution semantics."""

    target_name = "fusion"

    def __init__(self, manager: ConnectionManager, *, timeout_seconds: int = 30):
        self._manager = manager
        self._timeout_seconds = timeout_seconds

    async def execute_operation(
        self,
        session_id: str,
        operation: IROperation,
        *,
        tool_use_id: str,
        description: str = "",
    ) -> TargetExecutionResult:
        try:
            tool_name, tool_input = translate_ir_to_fusion_tool_call(operation)
            if description:
                tool_input["description"] = description
            code = translate_tool_call(tool_name, tool_input)
        except CodeGenerationError as exc:
            return TargetExecutionResult(
                success=False,
                target=self.target_name,
                message=f"Fusion code generation failed: {format_error_for_llm(exc)}",
            )
        except Exception as exc:
            return TargetExecutionResult(
                success=False,
                target=self.target_name,
                message=f"Fusion translation failed: {exc}",
            )

        payload = {
            "type": "execute_code",
            "operation": tool_name,
            "tool_use_id": tool_use_id,
            "description": description or tool_input.get("description", ""),
            "code": code,
        }

        await self._manager.send_message(session_id, payload)

        try:
            raw_result = await self._manager.wait_for_fusion_result(session_id, timeout=self._timeout_seconds)
        except asyncio.TimeoutError:
            return TargetExecutionResult(
                success=False,
                target=self.target_name,
                message=f"Timed out waiting for Fusion result for '{tool_name}'.",
            )

        success = bool(raw_result.get("success", raw_result.get("type") != "error"))
        message = str(raw_result.get("message") or f"{tool_name} executed in Fusion.")
        return TargetExecutionResult(
            success=success,
            target=self.target_name,
            message=message,
            data={"tool_name": tool_name, "tool_input": tool_input},
            raw_result=raw_result,
        )

    async def execute_document(
        self,
        session_id: str,
        document: IRDocument,
        *,
        request_id: Optional[str] = None,
    ) -> TargetExecutionResult:
        last_result: Optional[TargetExecutionResult] = None
        for index, operation in enumerate(document.operations):
            tool_use_id = f"{request_id or 'ir'}_{index+1}"
            last_result = await self.execute_operation(
                session_id,
                operation,
                tool_use_id=tool_use_id,
                description=f"IR operation {operation.id}",
            )
            if not last_result.success:
                return last_result

        return last_result or TargetExecutionResult(
            success=True,
            target=self.target_name,
            message="No IR operations to execute for Fusion.",
        )
