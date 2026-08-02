"""Fusion target adapter execution behind shared IR."""

from __future__ import annotations

import asyncio
import re
from uuid import uuid4
from typing import Any, Dict, Iterable, Mapping, Optional

from ...code_generator import CodeGenerationError, OPERATION_TEMPLATES, format_error_for_llm, translate_tool_call
from ...ir.types import IRDocument, IROperation
from ...websocket_manager import ConnectionManager
from ..base import TargetExecutionResult
from .translator import translate_ir_to_fusion_tool_call


_FEATURE_PAYLOAD_TOOLS = {
    "apply_fillet",
    "apply_chamfer",
    "create_shell",
    "create_simple_hole",
    "create_counterbore_hole",
    "create_tapped_hole",
    "create_external_thread",
    "create_pattern_feature",
    "suppress_feature",
    "unsuppress_feature",
}
_DELETE_FEATURE_TOOL = "delete_feature"
_PARAMETER_EDIT_TOOL = "adjust_feature_parameters"
_TIMELINE_DELETE_CAPABILITY = "timeline_feature_delete"
_TIMELINE_PARAMETER_EDIT_CAPABILITY = "timeline_feature_parameter_edit"
_SELECTION_PAYLOAD_TYPES = {
    "select_edges": "edge_operation",
    "clear_edge_selection": "edge_operation",
    "select_faces": "face_operation",
    "clear_face_selection": "face_operation",
    "select_bodies": "body_operation",
    "clear_body_selection": "body_operation",
}
_FACE_REF_PATTERN = re.compile(r"^face_\d+$")


def _looks_like_short_ref(value: str) -> bool:
    return bool(
        _FACE_REF_PATTERN.match(value)
        or re.match(r"^e\d+$", value)
        or re.match(r"^body_\d+$", value)
        or re.match(r"^v\d+$", value)
    )


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _request_declares_capability(
    request: Optional[Mapping[str, Any]],
    capability_name: str,
    *legacy_aliases: str,
) -> bool:
    capabilities = (request or {}).get("client_capabilities")
    if capabilities is None:
        capabilities = (request or {}).get("capabilities")

    aliases = tuple(alias for alias in legacy_aliases if alias)
    if isinstance(capabilities, Mapping):
        value = capabilities.get(capability_name)
        if value is None:
            for alias in aliases:
                if alias in capabilities:
                    value = capabilities.get(alias)
                    break
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value in (1,)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        return False

    if isinstance(capabilities, (list, tuple, set, frozenset)):
        normalized = {str(item).strip().lower() for item in capabilities}
        return capability_name in normalized or any(alias in normalized for alias in aliases)

    return False


class FusionTargetExecutor:
    """Execute IR operations by reusing the current Fusion execution semantics."""

    target_name = "fusion"

    def __init__(
        self,
        manager: ConnectionManager,
        *,
        timeout_seconds: int = 30,
        request: Optional[Mapping[str, Any]] = None,
    ):
        self._manager = manager
        self._timeout_seconds = timeout_seconds
        self._request = dict(request) if isinstance(request, Mapping) else None

    def _should_send_feature_payload(self, tool_name: str) -> bool:
        if tool_name == _DELETE_FEATURE_TOOL:
            return _request_declares_capability(
                self._request,
                _TIMELINE_DELETE_CAPABILITY,
                "feature_delete",
            )
        if tool_name == _PARAMETER_EDIT_TOOL:
            return _request_declares_capability(
                self._request,
                _TIMELINE_PARAMETER_EDIT_CAPABILITY,
            )
        return tool_name in _FEATURE_PAYLOAD_TOOLS

    def _resolve_token(self, session_id: str, ref_or_token: str, *, expected_kind: Optional[str] = None) -> str:
        cleaned = str(ref_or_token or "").strip()
        if not cleaned:
            kind_text = f"{expected_kind} " if expected_kind else ""
            raise ValueError(f"Missing {kind_text}reference.")

        get_store = getattr(self._manager, "get_entity_store", None)
        if callable(get_store):
            store = get_store(session_id)
            token, error = store.resolve_token(cleaned, expected_kind=expected_kind)
            if token and not error:
                return token
            raise ValueError(error or f"Could not resolve {expected_kind} reference '{cleaned}'.")

        if _looks_like_short_ref(cleaned):
            kind_text = f"{expected_kind} " if expected_kind else ""
            raise ValueError(
                f"Cannot resolve {kind_text}reference '{cleaned}' without an entity store. "
                "Pass a target token or execute through the workflow ref-resolution path."
            )
        return cleaned

    def _resolve_codegen_token_or_ref(
        self,
        session_id: str,
        params: Dict[str, Any],
        token_key: str,
        ref_key: str,
        *,
        expected_kind: Optional[str] = None,
    ) -> None:
        if ref_key in params and not str(params.get(token_key) or "").strip():
            params[token_key] = params.pop(ref_key)
        else:
            params.pop(ref_key, None)
        if str(params.get(token_key) or "").strip():
            params[token_key] = self._resolve_token(
                session_id,
                str(params[token_key]),
                expected_kind=expected_kind,
            )

    def _should_resolve_sketch_plane_as_face(self, session_id: str, plane_id: str) -> bool:
        if _FACE_REF_PATTERN.match(plane_id):
            return True

        get_store = getattr(self._manager, "get_entity_store", None)
        if not callable(get_store):
            return False

        store = get_store(session_id)
        has_ref = getattr(store, "has_ref", None)
        if callable(has_ref):
            return bool(has_ref(plane_id))
        return False

    def _prepare_codegen_tool_input(
        self,
        session_id: str,
        tool_name: str,
        tool_input: Mapping[str, Any],
    ) -> Dict[str, Any]:
        params = dict(tool_input)

        if tool_name == "create_sketch":
            plane_id = str(params.get("plane_id") or "").strip()
            if plane_id and self._should_resolve_sketch_plane_as_face(session_id, plane_id):
                params["plane_id"] = self._resolve_token(session_id, plane_id, expected_kind="face")
            return params

        if tool_name == "create_construction_plane":
            self._resolve_codegen_token_or_ref(
                session_id,
                params,
                "reference_edge_token",
                "reference_edge_ref",
                expected_kind="edge",
            )
            self._resolve_codegen_token_or_ref(
                session_id,
                params,
                "reference_face_token",
                "reference_face_ref",
                expected_kind="face",
            )
            self._resolve_codegen_token_or_ref(
                session_id,
                params,
                "face_token",
                "face_ref",
                expected_kind="face",
            )
            return params

        if tool_name == "revolve_profile":
            axis = params.get("axis")
            if isinstance(axis, Mapping):
                axis_params = dict(axis)
                axis_type = str(axis_params.get("type") or "").strip().lower()
                if axis_type == "edge":
                    self._resolve_codegen_token_or_ref(
                        session_id,
                        axis_params,
                        "edge_token",
                        "edge_ref",
                        expected_kind="edge",
                    )
                elif axis_type == "face":
                    self._resolve_codegen_token_or_ref(
                        session_id,
                        axis_params,
                        "face_token",
                        "face_ref",
                        expected_kind="face",
                    )
                params["axis"] = axis_params

            extent = params.get("extent")
            if isinstance(extent, Mapping):
                extent_params = dict(extent)
                for token_key, ref_key in (
                    ("to_entity_token", "to_entity_ref"),
                    ("to_entity1_token", "to_entity1_ref"),
                    ("to_entity2_token", "to_entity2_ref"),
                ):
                    self._resolve_codegen_token_or_ref(
                        session_id,
                        extent_params,
                        token_key,
                        ref_key,
                    )
                params["extent"] = extent_params

            self._resolve_codegen_token_or_ref(
                session_id,
                params,
                "creation_occurrence_token",
                "creation_occurrence_ref",
            )
            return params

        return params

    def _resolve_tokens(
        self,
        session_id: str,
        refs_or_tokens: Iterable[str],
        *,
        expected_kind: str,
    ) -> list[str]:
        resolved = [
            self._resolve_token(session_id, item, expected_kind=expected_kind)
            for item in refs_or_tokens
        ]
        if not resolved:
            raise ValueError(f"Expected at least one {expected_kind} reference.")
        return resolved

    def _prepare_payload_parameters(
        self,
        session_id: str,
        tool_name: str,
        tool_input: Mapping[str, Any],
    ) -> Dict[str, Any]:
        params = dict(tool_input)

        if tool_name in {"apply_fillet", "apply_chamfer"}:
            refs = _as_string_list(params.pop("edge_refs", params.get("entity_tokens")))
            params["entity_tokens"] = self._resolve_tokens(session_id, refs, expected_kind="edge")
            return params

        if tool_name == "create_shell":
            mode = str(params.get("mode") or "open").strip().lower()
            if mode == "closed":
                refs = _as_string_list(params.pop("body_refs", params.get("entity_tokens")))
                params.pop("face_refs", None)
                params["entity_tokens"] = self._resolve_tokens(session_id, refs, expected_kind="body")
            else:
                refs = _as_string_list(params.pop("face_refs", params.get("entity_tokens")))
                params.pop("body_refs", None)
                params["entity_tokens"] = self._resolve_tokens(session_id, refs, expected_kind="face")
            return params

        if tool_name in {
            "create_simple_hole",
            "create_counterbore_hole",
            "create_tapped_hole",
            "create_external_thread",
        }:
            face_ref = params.pop("face_ref", params.get("face_token"))
            params["face_token"] = self._resolve_token(session_id, str(face_ref or ""), expected_kind="face")
            return params

        if tool_name == "create_pattern_feature":
            refs = _as_string_list(params.pop("feature_refs", params.get("feature_tokens")))
            if any(ref.lower() == "auto_last" for ref in refs):
                raise ValueError(
                    "create_pattern_feature cannot resolve 'auto_last' in direct Fusion IR execution. "
                    "Provide concrete feature tokens or execute through the workflow feature-preparation path."
                )
            if not refs:
                raise ValueError("create_pattern_feature requires at least one feature token.")
            if any(_looks_like_short_ref(ref) for ref in refs):
                raise ValueError("create_pattern_feature requires feature tokens, not face/edge/body refs.")
            params["feature_tokens"] = refs
            return params

        if tool_name in {"select_edges", "select_faces", "select_bodies"}:
            kind = {"select_edges": "edge", "select_faces": "face", "select_bodies": "body"}[tool_name]
            ref_key = {"edge": "edge_refs", "face": "face_refs", "body": "body_refs"}[kind]
            refs = _as_string_list(params.pop(ref_key, params.get("entity_tokens")))
            params["entity_tokens"] = self._resolve_tokens(session_id, refs, expected_kind=kind)
            return params

        return params

    async def _wait_for_matching_tool_result(self, session_id: str, tool_use_id: str) -> dict:
        """Wait for a Fusion result with the expected tool_use_id, re-queuing mismatches."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0, int(self._timeout_seconds))
        deferred_results = []

        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                raw_result = await self._manager.wait_for_fusion_result(session_id, timeout=remaining)
                result_tool_use_id = raw_result.get("tool_use_id") if isinstance(raw_result, dict) else None
                result_tool_use_id_str = str(result_tool_use_id).strip() if result_tool_use_id is not None else ""
                if result_tool_use_id_str != tool_use_id:
                    deferred_results.append(raw_result)
                    continue
                return raw_result
        finally:
            for deferred in deferred_results:
                try:
                    await self._manager.store_fusion_result(session_id, dict(deferred))
                except Exception:
                    pass

    async def _wait_for_matching_message_id(self, session_id: str, message_id: str) -> dict:
        """Wait for a Fusion result with a specific message_id, re-queuing mismatches."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0, int(self._timeout_seconds))
        deferred_results = []

        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                raw_result = await self._manager.wait_for_fusion_result(session_id, timeout=remaining)
                result_message_id = raw_result.get("message_id") if isinstance(raw_result, dict) else None
                result_message_id_str = str(result_message_id).strip() if result_message_id is not None else ""
                if result_message_id_str != message_id:
                    deferred_results.append(raw_result)
                    continue
                return raw_result
        finally:
            for deferred in deferred_results:
                try:
                    await self._manager.store_fusion_result(session_id, dict(deferred))
                except Exception:
                    pass

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
        except Exception as exc:
            return TargetExecutionResult(
                success=False,
                target=self.target_name,
                message=f"Fusion translation failed: {exc}",
            )

        if tool_name == "list_features":
            message_id = f"feature_snapshot_{uuid4().hex}"
            payload = {
                "type": "feature_snapshot_request",
                "reason": tool_input.get("description") or description or "ir_list_features",
                "max_features": 25,
                "message_id": message_id,
            }
            await self._manager.send_message(session_id, payload)
            try:
                raw_result = await self._wait_for_matching_message_id(session_id, message_id)
            except asyncio.TimeoutError:
                return TargetExecutionResult(
                    success=False,
                    target=self.target_name,
                    message="Timed out waiting for Fusion feature snapshot.",
                )
            success = bool(raw_result.get("success", raw_result.get("type") != "error"))
            return TargetExecutionResult(
                success=success,
                target=self.target_name,
                message=str(raw_result.get("message") or "Feature snapshot received."),
                data={"tool_name": tool_name, "tool_input": tool_input},
                raw_result=raw_result,
            )

        if tool_name == _PARAMETER_EDIT_TOOL and not self._should_send_feature_payload(tool_name):
            return TargetExecutionResult(
                success=False,
                target=self.target_name,
                message=(
                    f"Fusion target requires client capability "
                    f"'{_TIMELINE_PARAMETER_EDIT_CAPABILITY}' for '{tool_name}'."
                ),
                data={"tool_name": tool_name, "tool_input": tool_input},
            )

        if self._should_send_feature_payload(tool_name) or tool_name in _SELECTION_PAYLOAD_TYPES:
            try:
                payload_parameters = self._prepare_payload_parameters(session_id, tool_name, tool_input)
            except ValueError as exc:
                return TargetExecutionResult(
                    success=False,
                    target=self.target_name,
                    message=f"Fusion IR reference resolution failed: {exc}",
                    data={"tool_name": tool_name, "tool_input": tool_input},
                )

            payload_type = "feature_operation" if self._should_send_feature_payload(tool_name) else _SELECTION_PAYLOAD_TYPES[tool_name]
            if tool_name == _PARAMETER_EDIT_TOOL:
                editable_parameters = payload_parameters.get("parameters")
                if not isinstance(editable_parameters, Mapping) or not editable_parameters:
                    return TargetExecutionResult(
                        success=False,
                        target=self.target_name,
                        message="Fusion IR parameter edit requires a non-empty nested 'parameters' object.",
                        data={"tool_name": tool_name, "tool_input": tool_input},
                    )
                payload = {
                    "type": payload_type,
                    "operation": tool_name,
                    "tool_use_id": tool_use_id,
                    "description": description or tool_input.get("description", ""),
                    "feature_token": payload_parameters.get("feature_token"),
                    "parameters": dict(editable_parameters),
                }
                if str(payload_parameters.get("expected_name") or "").strip():
                    payload["expected_name"] = payload_parameters["expected_name"]
                if payload_parameters.get("expected_timeline_index") is not None:
                    payload["expected_timeline_index"] = payload_parameters["expected_timeline_index"]
            else:
                payload = {
                    "type": payload_type,
                    "operation": tool_name,
                    "tool_use_id": tool_use_id,
                    "description": description or tool_input.get("description", ""),
                    "parameters": payload_parameters,
                }
            await self._manager.send_message(session_id, payload)
            try:
                raw_result = await self._wait_for_matching_tool_result(session_id, tool_use_id)
            except asyncio.TimeoutError:
                return TargetExecutionResult(
                    success=False,
                    target=self.target_name,
                    message=f"Timed out waiting for Fusion result for '{tool_name}'.",
                )
            success = bool(raw_result.get("success", raw_result.get("type") != "error"))
            message = str(
                raw_result.get("message")
                or raw_result.get("error")
                or f"{tool_name} {'executed' if success else 'failed'} in Fusion."
            )
            return TargetExecutionResult(
                success=success,
                target=self.target_name,
                message=message,
                data={"tool_name": tool_name, "tool_input": tool_input},
                raw_result=raw_result,
            )

        if tool_name not in OPERATION_TEMPLATES:
            return TargetExecutionResult(
                success=False,
                target=self.target_name,
                message=f"Fusion target does not support IR operation '{operation.type}' via tool '{tool_name}'.",
                data={"tool_name": tool_name, "tool_input": tool_input},
            )

        try:
            tool_input = self._prepare_codegen_tool_input(session_id, tool_name, tool_input)
        except ValueError as exc:
            return TargetExecutionResult(
                success=False,
                target=self.target_name,
                message=f"Fusion IR reference resolution failed: {exc}",
                data={"tool_name": tool_name, "tool_input": tool_input},
            )

        try:
            code = translate_tool_call(tool_name, tool_input)
        except CodeGenerationError as exc:
            return TargetExecutionResult(
                success=False,
                target=self.target_name,
                message=f"Fusion code generation failed: {format_error_for_llm(exc)}",
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
            raw_result = await self._wait_for_matching_tool_result(session_id, tool_use_id)
        except asyncio.TimeoutError:
            return TargetExecutionResult(
                success=False,
                target=self.target_name,
                message=f"Timed out waiting for Fusion result for '{tool_name}'.",
            )

        success = bool(raw_result.get("success", raw_result.get("type") != "error"))
        if success:
            message = str(raw_result.get("message") or f"{tool_name} executed in Fusion.")
        else:
            message = str(
                raw_result.get("error")
                or raw_result.get("details")
                or raw_result.get("message")
                or f"{tool_name} failed in Fusion."
            )
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
