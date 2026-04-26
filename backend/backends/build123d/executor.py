"""build123d target adapter executor behind shared IR."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from ...ir.types import IRDocument, IROperation
from ..base import TargetExecutionResult
from .entity_extractor import extract_entities
from .exporter import export_step_file
from .translator import translate_ir_document_to_build123d


class Build123dTargetExecutor:
    """Execute IR operations server-side with real build123d geometry."""

    target_name = "build123d"

    def __init__(self) -> None:
        self._session_part_cache: Dict[str, Any] = {}

    async def execute_operation(
        self,
        session_id: str,
        operation: IROperation,
        *,
        tool_use_id: str,
        description: str = "",
    ) -> TargetExecutionResult:
        document = IRDocument(version="1.0", units="mm", operations=[operation], metadata=None)
        return await self.execute_document(session_id, document, request_id=tool_use_id)

    async def execute_document(
        self,
        session_id: str,
        document: IRDocument,
        *,
        request_id: Optional[str] = None,
    ) -> TargetExecutionResult:
        try:
            program = translate_ir_document_to_build123d(document)
            part = await asyncio.to_thread(self._run_program, program.code)
            entities: Dict[str, Any] = {}
            if part is not None:
                self._session_part_cache[session_id] = part
                entities = extract_entities(part)
                message = f"Executed {len(document.operations)} IR operation(s) in build123d."
            else:
                message = (
                    f"Executed {len(document.operations)} IR operation(s) in build123d "
                    "without producing a solid body yet."
                )

            return TargetExecutionResult(
                success=True,
                target=self.target_name,
                message=message,
                data={
                    "entities": entities,
                    "ir_operation_count": len(document.operations),
                },
            )
        except Exception as exc:
            return TargetExecutionResult(
                success=False,
                target=self.target_name,
                message=f"build123d execution failed: {exc}",
            )

    def export_step(
        self,
        session_id: str,
        output_path: str,
    ) -> str:
        part = self._session_part_cache.get(session_id)
        if part is None:
            raise ValueError(f"No build123d part is cached for session '{session_id}'.")
        return export_step_file(part, output_path)

    @staticmethod
    def _run_program(code: str) -> Any:
        namespace: Dict[str, Any] = {}
        exec(code, namespace, namespace)
        return namespace.get("part_result")
