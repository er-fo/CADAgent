"""
Intelligent prompt routing system for CADAgent.

Routes user requests to appropriate tool clusters using a configurable LLM to
minimize context usage while maintaining full functionality.

The router analyzes the user's request and returns a hierarchical selection
of required and optional tool clusters. The prompt builder then assembles
the final prompt with only the relevant tools and documentation.

Performance:
- Router latency: low (using the configured router model)
- Token savings: 60-80% on input tokens
- Cost: <$0.0001 per routing decision

Usage:
    from backend.prompt_router import route_request
    from backend.prompt_builder import build_prompt

    # Route user request
    routing = await route_request(user_request)

    # Build optimized prompt
    system_prompt, tools = build_prompt(routing)

    # Use with LLM
    response = await llm_client.call_llm_agentic(
        messages=messages,
        tools=tools,
        system_prompt=system_prompt
    )
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from openai import AsyncOpenAI

from .prompt_structure import get_tool_catalog_text, CLUSTER_TOOL_MAPPING

logger = logging.getLogger(__name__)

_BEDROCK_API_KEY_ALIASES = (
    "aws_bearer_token_bedrock",
    "AWS_BEARER_TOKEN_BEDROCK",
    "bedrock_api_key",
    "bedrock_bearer_token",
)
_OPENAI_API_KEY_ALIASES = ("openai_api_key", "OPENAI_API_KEY")


def _resolve_bedrock_router_api_key(api_keys: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], str]:
    """Resolve Bedrock router key, preferring session BYOK over environment."""
    for key_name in _BEDROCK_API_KEY_ALIASES:
        candidate = (api_keys or {}).get(key_name)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip(), "session_byok"

    env_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if env_token and env_token.strip():
        return env_token.strip(), "environment"
    return None, "missing"


def _resolve_openai_router_api_key(api_keys: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], str]:
    """Resolve OpenAI-compatible router key, preferring session BYOK over environment."""
    for key_name in _OPENAI_API_KEY_ALIASES:
        candidate = (api_keys or {}).get(key_name)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip(), "session_byok"

    env_token = os.environ.get("OPENAI_API_KEY")
    if env_token and env_token.strip():
        return env_token.strip(), "environment"
    return None, "missing"


def _resolve_router_api_key_source(api_keys: Optional[Dict[str, str]] = None) -> str:
    """Return which source is currently backing the routing API key."""
    if _router_uses_bedrock():
        _, source = _resolve_bedrock_router_api_key(api_keys)
        return source
    _, source = _resolve_openai_router_api_key(api_keys)
    return source

def _build_routing_client(api_keys: Optional[Dict[str, str]] = None) -> Optional[AsyncOpenAI]:
    """Create routing client from per-session BYOK key with env fallback."""
    if _router_uses_bedrock():
        api_key, _ = _resolve_bedrock_router_api_key(api_keys)
        if not api_key:
            return None
        return AsyncOpenAI(api_key=api_key, base_url=ROUTER_BEDROCK_BASE_URL)

    api_key, _ = _resolve_openai_router_api_key(api_keys)
    if not api_key:
        return None

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    openai_base_url = os.environ.get("OPENAI_BASE_URL")
    if openai_base_url and openai_base_url.strip():
        client_kwargs["base_url"] = openai_base_url.strip()
    return AsyncOpenAI(**client_kwargs)


# Router model configuration
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "gpt-4.1-nano")
ROUTER_BEDROCK_BASE_URL = os.environ.get(
    "ROUTER_BEDROCK_BASE_URL",
    "https://bedrock-mantle.us-east-1.api.aws/v1",
)
ROUTER_MAX_TOKENS = 500
ROUTER_TEMPERATURE = 0.0  # Deterministic routing
_BEDROCK_ROUTER_PREFIXES = ("minimax.",)
ROUTER_PARSE_RETRY_LIMIT = 1


def _router_uses_bedrock() -> bool:
    """Return True when the configured router model should use Bedrock."""
    model_name = (ROUTER_MODEL or "").strip().lower()
    return any(model_name.startswith(prefix) for prefix in _BEDROCK_ROUTER_PREFIXES)


def _router_provider_path() -> str:
    """Return a short provider label for logs and fallback reasons."""
    return "bedrock" if _router_uses_bedrock() else "openai-compatible"


def _coerce_content_to_text(content: Any) -> str:
    """Best-effort conversion of OpenAI-compatible content payloads to text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = ""
                for key in ("text", "content"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        text = value.strip()
                        break
            else:
                text_value = getattr(item, "text", None)
                if not isinstance(text_value, str):
                    text_value = getattr(item, "content", None)
                text = text_value.strip() if isinstance(text_value, str) else ""

            if text:
                parts.append(text)

        return "\n".join(parts).strip()

    text_value = getattr(content, "text", None)
    if not isinstance(text_value, str):
        text_value = getattr(content, "content", None)
    return text_value.strip() if isinstance(text_value, str) else ""


def _extract_text_from_payload(payload: Any) -> str:
    """Recursively extract text from OpenAI-compatible payload variants."""
    if payload is None:
        return ""

    if isinstance(payload, str):
        return payload.strip()

    if isinstance(payload, list):
        parts = [_extract_text_from_payload(item) for item in payload]
        return "\n".join(part for part in parts if part).strip()

    if isinstance(payload, dict):
        if payload.get("type") == "text" and isinstance(payload.get("text"), str):
            text = payload.get("text", "").strip()
            if text:
                return text

        for key in ("text", "content", "output_text", "reasoning_content", "reasoning", "arguments"):
            text = _extract_text_from_payload(payload.get(key))
            if text:
                return text

        for key in ("parts", "items", "segments", "fragments", "value"):
            text = _extract_text_from_payload(payload.get(key))
            if text:
                return text

        for value in payload.values():
            text = _extract_text_from_payload(value)
            if text:
                return text

        return ""

    for attr in ("text", "content", "output_text", "reasoning_content", "reasoning", "arguments"):
        if hasattr(payload, attr):
            text = _extract_text_from_payload(getattr(payload, attr))
            if text:
                return text

    if hasattr(payload, "__dict__") and getattr(payload, "__dict__", None):
        return _extract_text_from_payload(vars(payload))

    return ""


def _extract_routing_response_text(response: Any) -> str:
    """
    Extract router response text from OpenAI-compatible response objects.

    Some providers return non-string message content (e.g. list parts), and
    some can return `None` in `message.content` while placing text elsewhere.
    """
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        raise ValueError("Routing response has no choices")

    candidates: List[str] = []

    for choice in choices:
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")

        if isinstance(message, dict):
            candidates.append(_extract_text_from_payload(message.get("content")))
            candidates.append(_extract_text_from_payload(message.get("reasoning_content")))
            candidates.append(_extract_text_from_payload(message.get("reasoning")))
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        function_payload = tool_call.get("function") or {}
                        if isinstance(function_payload, dict):
                            candidates.append(_extract_text_from_payload(function_payload.get("arguments")))
        else:
            candidates.append(_extract_text_from_payload(getattr(message, "content", None)))
            candidates.append(_extract_text_from_payload(getattr(message, "reasoning_content", None)))
            candidates.append(_extract_text_from_payload(getattr(message, "reasoning", None)))
            tool_calls = getattr(message, "tool_calls", None)
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        function_payload = tool_call.get("function") or {}
                        if isinstance(function_payload, dict):
                            candidates.append(_extract_text_from_payload(function_payload.get("arguments")))
                    else:
                        function_payload = getattr(tool_call, "function", None)
                        candidates.append(_extract_text_from_payload(getattr(function_payload, "arguments", None)))

        if isinstance(choice, dict):
            candidates.append(_extract_text_from_payload(choice.get("text")))
        else:
            candidates.append(_extract_text_from_payload(getattr(choice, "text", None)))

    candidates.append(_extract_text_from_payload(getattr(response, "output_text", None)))
    if isinstance(response, dict):
        candidates.append(_extract_text_from_payload(response.get("output_text")))

    for candidate in candidates:
        if candidate:
            return candidate

    raise ValueError("Routing response has no textual content")


def _strip_json_fences(text: str) -> str:
    """Remove a single outer markdown code fence when present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) == 1:
        return stripped

    body = "\n".join(lines[1:]).strip()
    if body.endswith("```"):
        body = body[:-3].strip()
    return body


def _extract_json_candidate(text: str) -> str:
    """Extract the most likely JSON object or array from a response."""
    stripped = text.strip()
    if not stripped:
        return ""

    if stripped[0] in "{[" and stripped[-1] in "}]":
        return stripped

    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end != -1 and start < end:
            candidate = stripped[start : end + 1].strip()
            if candidate.startswith(opener) and candidate.endswith(closer):
                return candidate

    return stripped


def _bounded_json_repair_suffix(text: str) -> str:
    """
    Return a short suffix that safely closes an obviously truncated JSON value.

    The repair is intentionally conservative: it only appends missing closing
    braces/brackets when the text appears to be a complete prefix and is not
    currently inside a string literal.
    """
    stripped = text.rstrip()
    if not stripped or stripped[0] not in "{[":
        return ""

    stack: List[str] = []
    in_string = False
    escape = False

    for char in stripped:
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in "{[":
            stack.append(char)
            continue

        if char in "}]":
            if not stack:
                return ""
            opener = stack.pop()
            if (opener == "{" and char != "}") or (opener == "[" and char != "]"):
                return ""

    if in_string:
        return ""

    if not stack:
        return ""

    last_non_ws = ""
    for char in reversed(stripped):
        if not char.isspace():
            last_non_ws = char
            break

    if last_non_ws in {":", ",", "{", "["}:
        return ""

    if len(stack) > 4:
        return ""

    return "".join("}" if opener == "{" else "]" for opener in reversed(stack))


def _classify_router_parse_failure(text: str, error: Optional[Exception]) -> str:
    """Return a short parse-failure category for logs and fallback reasons."""
    stripped = (text or "").strip()
    if not stripped:
        return "empty_content"

    if stripped[0] in "{[":
        last_non_ws = ""
        for char in reversed(stripped):
            if not char.isspace():
                last_non_ws = char
                break
        if last_non_ws in {":", ",", "{", "["}:
            return "truncated_json"
        if isinstance(error, json.JSONDecodeError) and "Unterminated string" in str(error):
            return "truncated_json"
        return "malformed_json"

    if "{" in stripped or "[" in stripped:
        return "wrapped_json_parse_error"

    return "non_json_text"


def _normalize_cluster_list(value: Any) -> List[str]:
    """Normalize cluster collections into a clean string list."""
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []

    if isinstance(value, list):
        output: List[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text:
                output.append(text)
        return output

    return []


def _coerce_router_payload_shape(payload: Any) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Coerce provider payload variants into canonical routing dict shape.

    Returns (routing_dict_or_none, was_coerced).
    """
    if isinstance(payload, list):
        # Common variant: a single-object list around the JSON payload.
        for item in payload:
            if isinstance(item, dict):
                coerced, was_coerced = _coerce_router_payload_shape(item)
                if coerced is not None:
                    return coerced, True
        return None, False

    if not isinstance(payload, dict):
        return None, False

    alias_required = payload.get("required_clusters")
    alias_optional = payload.get("optional_clusters")
    required = payload.get("required")
    optional = payload.get("optional")
    reasoning = payload.get("reasoning")

    if required is None and alias_required is not None:
        required = alias_required
    if optional is None and alias_optional is not None:
        optional = alias_optional

    if required is not None or optional is not None:
        normalized_required = _normalize_cluster_list(required)
        normalized_optional = _normalize_cluster_list(optional)
        normalized_reasoning = reasoning if isinstance(reasoning, str) else "No reasoning provided"
        coerced = dict(payload)
        coerced["required"] = normalized_required
        coerced["optional"] = normalized_optional
        coerced["reasoning"] = normalized_reasoning
        was_coerced = alias_required is not None or alias_optional is not None
        return coerced, was_coerced

    for nested_key in ("routing", "result", "output", "data", "response", "decision"):
        nested = payload.get(nested_key)
        nested_coerced, nested_was_coerced = _coerce_router_payload_shape(nested)
        if nested_coerced is not None:
            if isinstance(reasoning, str) and not nested_coerced.get("reasoning"):
                nested_coerced["reasoning"] = reasoning
            return nested_coerced, True or nested_was_coerced

    return None, False


def _safe_parse_router_response(response_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Parse a router response without raising on malformed provider output.

    Returns a tuple of (parsed_result, status). Status is "ok" or a short
    failure category such as "truncated_json" or "empty_content".
    """
    normalized_text = _strip_json_fences(response_text)
    if not normalized_text:
        return None, "empty_content"

    candidates = [normalized_text]
    extracted_candidate = _extract_json_candidate(normalized_text)
    if extracted_candidate and extracted_candidate != normalized_text:
        candidates.append(extracted_candidate)

    last_error: Optional[Exception] = None

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            coerced, was_coerced = _coerce_router_payload_shape(parsed)
            if coerced is not None:
                return coerced, "coerced_shape" if was_coerced else "ok"
            return None, "unexpected_json_shape"
        except json.JSONDecodeError as exc:
            last_error = exc

        repair_suffix = _bounded_json_repair_suffix(candidate)
        if repair_suffix:
            try:
                parsed = json.loads(candidate + repair_suffix)
                coerced, was_coerced = _coerce_router_payload_shape(parsed)
                if coerced is not None:
                    return coerced, "repaired_coerced_shape" if was_coerced else "repaired"
                return None, "unexpected_json_shape"
            except json.JSONDecodeError as exc:
                last_error = exc

    return None, _classify_router_parse_failure(normalized_text, last_error)


def _fallback_with_reason(
    user_request: str,
    fallback_reason: str,
    *,
    provider_path: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key_source: Optional[str] = None,
    parse_status: Optional[str] = None,
    retry_attempted: Optional[bool] = None,
    retry_count: Optional[int] = None,
    retry_parse_status: Optional[str] = None,
    initial_parse_status: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call the fallback router while keeping compatibility with older stubs.

    Some tests monkeypatch _fallback_routing with a single-argument stub. We
    prefer to pass the explicit reason when the real function is present, but
    gracefully degrade to the older call shape when a stub rejects it.
    """
    try:
        fallback_result = _fallback_routing(user_request, fallback_reason=fallback_reason)
    except TypeError as exc:
        if "fallback_reason" not in str(exc):
            raise
        fallback_result = _fallback_routing(user_request)

    fallback_result["fallback_reason"] = fallback_reason
    fallback_result["reasoning"] = f"Fallback routing using keyword matching ({fallback_reason})"
    fallback_result.setdefault("fallback", True)
    fallback_result.setdefault("routing_source", "keyword_fallback")
    if provider_path:
        fallback_result["router_provider"] = provider_path
    if model_name:
        fallback_result["router_model"] = model_name
    if api_key_source:
        fallback_result["router_api_key_source"] = api_key_source
    if parse_status:
        fallback_result["router_parse_status"] = parse_status
    if retry_attempted is not None:
        fallback_result["router_retry_attempted"] = retry_attempted
    if retry_count is not None:
        fallback_result["router_retry_count"] = retry_count
    if retry_parse_status is not None:
        fallback_result["router_retry_parse_status"] = retry_parse_status
    if initial_parse_status is not None:
        fallback_result["router_initial_parse_status"] = initial_parse_status
    return fallback_result


def _build_router_json_repair_user_message(
    user_request: str,
    raw_router_text: str,
    parse_status: str,
) -> str:
    """Build strict JSON-only repair instruction for a single retry."""
    return (
        "Repair this router output into strict JSON only.\n\n"
        "Constraints:\n"
        '- Output must be ONLY one JSON object with keys: "required", "optional", "reasoning".\n'
        '- "required" and "optional" must be arrays of strings.\n'
        '- Include "core" in "required".\n'
        "- No markdown, no prose, no comments, no code fences.\n"
        f"- Previous parse status: {parse_status}\n\n"
        f"User request:\n{user_request}\n\n"
        "Router output to repair:\n"
        f"{raw_router_text}"
    )

# Router system prompt (system role)
ROUTER_PROMPT = """You are a tool routing classifier for a CAD modeling agent.

Given a user request, identify which tool clusters are needed to fulfill it.

{tool_catalog}

**Your Task:**
Analyze the user request and return a JSON object specifying which clusters are required and which might be optionally useful.

**Rules:**
1. Your decision locks the tool set for the ENTIRE execution loop. When uncertain, INCLUDE the cluster; missing tools halt progress.
2. "core" is ALWAYS required (communication).
3. Include sketch_tools + 3d_modeling when new sketches/profiles/extrudes/revolves/lofts are needed. For face/edge-only edits (holes, fillets, threads, shell) you may skip sketch_tools unless a new sketch is clearly required.
4. For modifications, typically need: inspection + selection + modification.
5. **HOLES - AGGRESSIVE INCLUSION (CRITICAL):** Any mention of holes/bosses/standoffs/mounting/fasteners → ALWAYS include `holes` cluster as REQUIRED, even if uncertain:
   - "holes", "hole", "mounting", "boss", "standoff", "fastener", "screw", "bolt" → holes cluster REQUIRED
   - This includes: through holes, blind holes, counterbores, countersinks, tapped holes
   - If ANY possibility exists that holes will be needed, include the cluster
   - Better to have hole tools available and not use them than to lack them when needed
   - For threads: also include `threading` cluster
6. For patterns: include patterns + inspection (to refresh feature tokens) and modeling/selection if the source feature needs picking.
7. **MULTI-FEATURE PARTS WITH HOLES:** When the request includes multi-feature patterns (enclosures, brackets, machined parts) AND mentions holes/bosses/standoffs/mounting hardware, ALWAYS include the `holes` cluster as REQUIRED (not optional):
   - "mounting holes" → holes cluster REQUIRED
   - "bosses for standoffs" → holes cluster REQUIRED (bosses typically need tapped holes)
   - "corner holes" → holes cluster REQUIRED
   - "four holes and a pocket" → holes cluster REQUIRED
8. **NEW - Build Plan Routing:** If an "Active Build Plan" is provided with specific operations, ensure those operations' tool clusters are REQUIRED:
   - apply_fillet, apply_chamfer, create_shell → modification
   - create_simple_hole, create_counterbore_hole, create_tapped_hole → holes
   - create_external_thread → threading
   - create_pattern_feature → patterns
   - select_edges, select_faces, select_bodies → selection
   - list_features → inspection
   - jump_to_timeline_position, delete_feature, adjust_feature_parameters → timeline
9. Consider full multi-step workflows, not just the first action. Include optional clusters that might reasonably be helpful.
10. **WHEN IN DOUBT, BE GENEROUS:** If a cluster might be useful, include it. The cost of loading extra tools is minimal compared to the cost of missing required tools.
11. Output strict JSON only—no markdown, no comments.

Examples:
- "drill four holes in the top face" → required: [core, inspection, holes]; optional: [selection]
- "shell the body then fillet the outer edges" → required: [core, inspection, selection, modification]
- "add M6 thread to existing cylinder" → required: [core, inspection, threading, holes]; optional: [selection]
- "delete the faulty fillet" → required: [core, inspection, timeline]; optional: []
- "remove the last extrude" → required: [core, inspection, timeline]; optional: []
- "machined aluminum enclosure with mounting holes and pocket" → required: [core, design_exploration, sketch_tools, 3d_modeling, holes]; optional: [modification]
- "bracket with four corner holes" → required: [core, design_exploration, sketch_tools, 3d_modeling, holes]; optional: []
- "part with bosses for standoffs" → required: [core, design_exploration, sketch_tools, 3d_modeling, holes]; optional: [threading, modification]
- "base plate with mounting features" → required: [core, design_exploration, sketch_tools, 3d_modeling, holes]; optional: [modification] (mounting implies holes)

**Design Exploration Routing:**

Include `design_exploration` cluster when:
- User describes a problem, need, or functional intent (not specific geometry)
- Request is ambiguous or could be interpreted multiple ways
- User explicitly asks for options, alternatives, or suggestions
- Request mentions "something to...", "I need a...", "help me with..."
- Functional language: "hold", "catch", "block", "organize", "mount", "attach"

**MULTI-FEATURE PART PATTERNS (ALWAYS include design_exploration + sketch_tools + 3d_modeling):**
- Multiple features listed: "with pocket, holes, and bosses", "mounting holes and standoffs"
- Manufacturing-specified: "machined", "milled", "CNC", "3D printed", "cast", "injection molded"
- Enclosure/housing language: "enclosure base", "electronics housing", "case", "chassis"
- Material + function: "aluminum bracket", "plastic housing", "steel base plate"
- Feature enumeration: "four holes", "two bosses", "internal pocket", "corner mounts"
- Electronics/mechanical integration: "PCB mount", "standoffs", "board mounting"
- **CRITICAL**: Multi-feature parts ALWAYS need: design_exploration + sketch_tools + 3d_modeling as REQUIRED
- **HOLES DETECTION**: If the request mentions holes/bosses/standoffs/mounting, also include `holes` as REQUIRED

Exclude `design_exploration` cluster when:
- User specifies exact geometry ("make a 50mm cube")
- User provides COMPLETE dimensions for ALL features (not just overall size)
- Request is for modification of existing geometry (fillets, holes at specific coords)
- User says "just build", "exactly like", or provides detailed specs
- Simple primitive shapes with full dimensions

Design exploration examples:
- "I need a drain cover for my shower" → required: [core, design_exploration, sketch_tools, 3d_modeling]; optional: [modification]
- "Make a cylinder 50mm diameter, 20mm tall" → required: [core, sketch_tools, 3d_modeling]; optional: [] (NO design_exploration)
- "Something to hold my phone on my desk" → required: [core, design_exploration, sketch_tools, 3d_modeling]; optional: []
- "Add a 5mm fillet to the top edge" → required: [core, inspection, selection, modification]; optional: [] (NO design_exploration)
- "I need a bracket but not sure what kind" → required: [core, design_exploration, sketch_tools, 3d_modeling]; optional: []
- "Machined aluminum enclosure with mounting holes" → required: [core, design_exploration, sketch_tools, 3d_modeling, holes]; optional: [modification] (manufacturing + features + holes)
- "Electronics housing base with internal pocket and standoffs" → required: [core, design_exploration, sketch_tools, 3d_modeling, holes]; optional: [modification] (enclosure + multi-feature + bosses/standoffs need holes)
- "CNC part with four corner holes and a slot" → required: [core, design_exploration, sketch_tools, 3d_modeling, holes]; optional: [] (manufacturing + enumerated holes)
- "Design a compact base with pocket, holes, and bosses" → required: [core, design_exploration, sketch_tools, 3d_modeling, holes]; optional: [modification] (multi-feature with holes)

**Output Format (JSON only):**
{{
  "required": ["cluster_id1", "cluster_id2"],
  "optional": ["cluster_id3"],
  "reasoning": "Brief explanation of why these clusters were selected"
}}"""

# Common routing patterns (used as hints in conversation history)
ROUTING_PATTERNS = {
    "basic_sketch": {
        "keywords": ["sketch", "draw", "circle", "rectangle", "line", "arc", "arcs"],
        "typical_clusters": ["core", "sketch_tools", "3d_modeling"]
    },
    "extrude": {
        "keywords": ["extrude", "box", "cylinder", "base", "plate"],
        "typical_clusters": ["core", "sketch_tools", "3d_modeling"]
    },
    "modifications": {
        "keywords": ["fillet", "chamfer", "round", "edge", "smooth", "shell", "hollow", "thin wall"],
        "typical_clusters": ["core", "inspection", "selection", "modification"]
    },
    "holes": {
        "keywords": ["hole", "drill", "bore", "tap", "thread"],
        "typical_clusters": ["core", "inspection", "holes"]
    },
    "complex": {
        "keywords": ["bracket", "housing", "mount", "assembly"],
        "typical_clusters": ["core", "construction", "sketch_tools", "3d_modeling", "inspection", "modification"]
    }
}


def _extract_conversation_context(conversation_history: Optional[List[Dict[str, Any]]]) -> str:
    """
    Extract recent conversation context for routing decisions.

    Takes the last 2-3 conversation turns (user + assistant pairs) to help
    the router understand follow-up requests like "make it bigger" or
    "add fillets to those edges".

    Args:
        conversation_history: Full conversation history from agent_workflow

    Returns:
        Formatted string with recent context, or empty string if no history
    """
    if not conversation_history or len(conversation_history) == 0:
        return ""

    # Extract last 2-3 turns (up to 6 messages: user, assistant, user, assistant, user, assistant)
    recent_messages = conversation_history[-6:]

    # Build compact context summary
    context_lines = []
    for msg in recent_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", [])

        # Extract text from content blocks
        text_parts = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        # Include tool names to show what operations were attempted
                        tool_name = block.get("name", "unknown_tool")
                        text_parts.append(f"[Used tool: {tool_name}]")
        elif isinstance(content, str):
            text_parts.append(content)

        # Join and truncate if needed
        text = " ".join(text_parts).strip()
        if len(text) > 200:  # Keep context compact
            text = text[:200] + "..."

        if text:
            context_lines.append(f"{role.capitalize()}: {text}")

    if not context_lines:
        return ""

    return "\n".join(context_lines)


def _extract_operations_from_build_plan(build_plan: Optional[Dict[str, Any]]) -> str:
    """
    Extract CAD operations from build plan steps for router context.

    Args:
        build_plan: Build plan with design_name, steps, completed_steps

    Returns:
        Formatted string listing operations, or empty string if no plan
    """
    if not build_plan or not build_plan.get("steps"):
        return ""

    steps = build_plan.get("steps", [])
    design_name = build_plan.get("design_name", "Design")

    # Extract operation names from steps
    operations = []
    for step in steps:
        operation = step.get("operation", "")
        description = step.get("description", "")
        if operation:
            operations.append(f"- {operation}: {description}")
        elif description:
            operations.append(f"- {description}")

    if not operations:
        return ""

    return f"""
**Active Build Plan: {design_name}**
Planned operations ({len(operations)} steps):
{chr(10).join(operations)}
"""


async def route_request(
    user_request: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    build_plan: Optional[Dict[str, Any]] = None,
    api_keys: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Route user request to appropriate tool clusters.

    Uses the configured router model to quickly classify the request and
    determine which tool clusters are needed. Returns a hierarchical selection
    with required and optional clusters.

    Args:
        user_request: The user's natural language CAD request
        conversation_history: Optional conversation context for better routing
        build_plan: Optional active build plan with planned operations

    Returns:
        Dictionary with structure:
        {
            "required": ["cluster_id1", "cluster_id2", ...],
            "optional": ["cluster_id3", ...],
            "reasoning": "Explanation of routing decision",
            "confidence": "high" | "medium" | "low"
        }

    Raises:
        Exception: If routing fails (fallback: return all clusters)
    """
    try:
        provider_path = _router_provider_path()
        router_key_source = _resolve_router_api_key_source(api_keys)
        retry_attempted = False
        retry_count = 0
        retry_parse_status = "not_attempted"
        initial_parse_status = "not_attempted"
        logger.info(f"Routing request: {user_request[:100]}...")

        routing_client = _build_routing_client(api_keys)
        if routing_client is None:
            logger.warning(
                "Prompt router missing credentials | provider=%s model=%s api_key_source=%s",
                provider_path,
                ROUTER_MODEL,
                router_key_source,
            )
            return _fallback_with_reason(
                user_request,
                "missing_router_credentials",
                provider_path=provider_path,
                model_name=ROUTER_MODEL,
                api_key_source=router_key_source,
                parse_status="missing_router_credentials",
                retry_attempted=retry_attempted,
                retry_count=retry_count,
                retry_parse_status=retry_parse_status,
                initial_parse_status=initial_parse_status,
            )
        logger.info(
            "Prompt router initialized | provider=%s model=%s api_key_source=%s",
            provider_path,
            ROUTER_MODEL,
            router_key_source,
        )

        tool_catalog = get_tool_catalog_text()
        system_prompt = ROUTER_PROMPT.format(tool_catalog=tool_catalog)
        context_summary = _extract_conversation_context(conversation_history)
        build_plan_context = _extract_operations_from_build_plan(build_plan)

        message_parts = []
        if context_summary:
            message_parts.append(f"**Recent conversation context:**\n{context_summary}")
        if build_plan_context:
            message_parts.append(build_plan_context)
        message_parts.append(f"**Current request:**\n{user_request}")
        user_message = "\n\n".join(message_parts)

        response = await routing_client.chat.completions.create(
            model=ROUTER_MODEL,
            max_completion_tokens=ROUTER_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        try:
            response_text = _extract_routing_response_text(response)
        except Exception as exc:
            logger.warning(
                "Prompt router response extraction failed | provider=%s model=%s category=missing_text error=%s",
                provider_path,
                ROUTER_MODEL,
                type(exc).__name__,
            )
            return _fallback_with_reason(
                user_request,
                "missing_text",
                provider_path=provider_path,
                model_name=ROUTER_MODEL,
                api_key_source=router_key_source,
                parse_status="missing_text",
                retry_attempted=retry_attempted,
                retry_count=retry_count,
                retry_parse_status=retry_parse_status,
                initial_parse_status=initial_parse_status,
            )

        routing_result, parse_status = _safe_parse_router_response(response_text)
        initial_parse_status = parse_status
        if routing_result is None:
            logger.warning(
                "Prompt router parse failure | provider=%s model=%s category=%s",
                provider_path,
                ROUTER_MODEL,
                parse_status,
            )

            if ROUTER_PARSE_RETRY_LIMIT > 0:
                retry_attempted = True
                retry_count = 1
                try:
                    repair_response = await routing_client.chat.completions.create(
                        model=ROUTER_MODEL,
                        max_completion_tokens=ROUTER_MAX_TOKENS,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a strict JSON repair assistant for CAD router output. "
                                    "Return JSON only."
                                ),
                            },
                            {
                                "role": "user",
                                "content": _build_router_json_repair_user_message(
                                    user_request=user_request,
                                    raw_router_text=response_text,
                                    parse_status=parse_status,
                                ),
                            },
                        ],
                    )
                    repair_text = _extract_routing_response_text(repair_response)
                    repaired_result, repair_status = _safe_parse_router_response(repair_text)
                    retry_parse_status = repair_status
                    if repaired_result is not None:
                        routing_result = repaired_result
                        parse_status = f"retry_{repair_status}"
                    else:
                        parse_status = f"retry_failed:{repair_status}"
                except Exception as retry_exc:
                    logger.warning(
                        "Prompt router repair retry failed | provider=%s model=%s error=%s",
                        provider_path,
                        ROUTER_MODEL,
                        type(retry_exc).__name__,
                    )
                    retry_parse_status = f"retry_error:{type(retry_exc).__name__}"
                    parse_status = retry_parse_status

            if routing_result is None:
                return _fallback_with_reason(
                    user_request,
                    f"parse_failure:{initial_parse_status}",
                    provider_path=provider_path,
                    model_name=ROUTER_MODEL,
                    api_key_source=router_key_source,
                    parse_status=parse_status,
                    retry_attempted=retry_attempted,
                    retry_count=retry_count,
                    retry_parse_status=retry_parse_status,
                    initial_parse_status=initial_parse_status,
                )

        if parse_status == "repaired":
            logger.info(
                "Prompt router applied bounded JSON repair | provider=%s model=%s",
                provider_path,
                ROUTER_MODEL,
            )
        if parse_status in {"coerced_shape", "repaired_coerced_shape"} or parse_status.startswith("retry_"):
            logger.info(
                "Prompt router coerced/retried response shape | provider=%s model=%s status=%s",
                provider_path,
                ROUTER_MODEL,
                parse_status,
            )

        if "required" not in routing_result:
            routing_result["required"] = []
        if "optional" not in routing_result:
            routing_result["optional"] = []
        if "reasoning" not in routing_result:
            routing_result["reasoning"] = "No reasoning provided"

        if "core" not in routing_result["required"]:
            routing_result["required"].insert(0, "core")

        routing_result["required"] = list(dict.fromkeys(routing_result["required"]))
        routing_result["optional"] = list(dict.fromkeys(routing_result["optional"]))

        total_clusters = len(routing_result["required"]) + len(routing_result["optional"])
        if total_clusters <= 4:
            routing_result["confidence"] = "high"
        elif total_clusters <= 7:
            routing_result["confidence"] = "medium"
        else:
            routing_result["confidence"] = "low"

        routing_result.setdefault("routing_source", "llm_router")
        routing_result["router_provider"] = provider_path
        routing_result["router_model"] = ROUTER_MODEL
        routing_result["router_api_key_source"] = router_key_source
        routing_result["router_parse_status"] = parse_status
        routing_result["router_retry_attempted"] = retry_attempted
        routing_result["router_retry_count"] = retry_count
        routing_result["router_retry_parse_status"] = retry_parse_status
        routing_result["router_initial_parse_status"] = initial_parse_status

        logger.info(
            f"Routing complete: {len(routing_result['required'])} required, "
            f"{len(routing_result['optional'])} optional clusters "
            f"(confidence: {routing_result['confidence']})"
        )
        logger.debug(f"Required: {routing_result['required']}")
        logger.debug(f"Optional: {routing_result['optional']}")
        logger.debug(f"Reasoning: {routing_result['reasoning']}")
        return routing_result

    except Exception as e:
        logger.error(
            "Routing failed | provider=%s model=%s error=%s",
            _router_provider_path(),
            ROUTER_MODEL,
            type(e).__name__,
            exc_info=True,
        )
        return _fallback_with_reason(
            user_request,
            f"routing_error:{type(e).__name__}",
            provider_path=_router_provider_path(),
            model_name=ROUTER_MODEL,
            api_key_source=_resolve_router_api_key_source(api_keys),
            parse_status=f"routing_error:{type(e).__name__}",
            retry_attempted=False,
            retry_count=0,
            retry_parse_status="not_attempted",
            initial_parse_status="not_attempted",
        )


def _fallback_routing(user_request: str, fallback_reason: str = "ai_router_failed") -> Dict[str, Any]:
    """
    Fallback routing when AI router fails.

    Uses simple keyword matching to make a best-effort routing decision.
    Better than loading ALL clusters but not as smart as AI routing.

    Args:
        user_request: The user's request

    Returns:
        Conservative routing result that should cover most use cases
    """
    logger.warning("Using fallback routing based on keywords | reason=%s", fallback_reason)

    request_lower = user_request.lower()

    # Start with core (always needed)
    required = ["core"]
    optional = []

    # Check for common patterns
    if any(kw in request_lower for kw in ["sketch", "draw", "circle", "rectangle", "line"]):
        required.extend(["sketch_tools", "3d_modeling"])

    if any(kw in request_lower for kw in ["extrude", "cylinder", "box", "plate"]):
        required.extend(["sketch_tools", "3d_modeling"])

    if any(kw in request_lower for kw in ["fillet", "chamfer", "round", "edge", "shell", "hollow", "thin wall", "wall thickness"]):
        required.extend(["inspection", "selection", "modification"])

    # AGGRESSIVE hole detection - any indication of holes/fasteners/mounting → include holes cluster
    # This is intentionally broad to prevent missing hole tools
    hole_keywords = [
        "hole", "drill", "bore", "tap", "tapped",
        "mounting hole", "corner hole", "clearance hole", "pilot hole",
        "boss", "standoff", "mounting", "mount",
        "counterbore", "countersink", "cbore", "csink",
        "fastener", "screw", "bolt", "nut",
        "m3", "m4", "m5", "m6", "m8", "m10",  # Thread sizes strongly imply holes
        "#6-", "#8-", "#10-", "1/4-", "5/16-", "3/8-",  # Imperial thread sizes
    ]
    if any(kw in request_lower for kw in hole_keywords):
        if "holes" not in required:
            required.append("holes")
        # For modifications, also need inspection
        if "inspection" not in required and any(kw in request_lower for kw in ["hole", "drill", "bore"]):
            required.append("inspection")

    if any(kw in request_lower for kw in ["housing", "enclosure", "container", "tray", "case", "box enclosure", "shell"]):
        required.extend(["sketch_tools", "3d_modeling", "modification", "inspection"])

    if any(kw in request_lower for kw in ["thread", "screw", "bolt"]):
        required.extend(["inspection", "threading"])
        # If creating threaded shaft/part, need modeling too
        if any(kw in request_lower for kw in ["create", "shaft", "cylinder", "bolt"]):
            required.extend(["sketch_tools", "3d_modeling"])

    if any(kw in request_lower for kw in ["pattern", "array", "repeat"]):
        required.append("patterns")

    if any(kw in request_lower for kw in ["angle", "tilt", "rotate", "offset"]):
        required.append("construction")

    # For complex multi-step requests mentioning multiple operations
    operation_count = sum([
        any(kw in request_lower for kw in ["sketch", "draw"]),
        any(kw in request_lower for kw in ["extrude", "revolve", "loft"]),
        any(kw in request_lower for kw in ["hole", "drill"]),
        any(kw in request_lower for kw in ["fillet", "chamfer"]),
        any(kw in request_lower for kw in ["angle", "construction"]),
    ])

    if operation_count >= 3:
        # Complex request, include more clusters
        required.extend(["sketch_tools", "3d_modeling", "inspection"])

    # Design exploration for problem-first requests
    design_exploration_include = [
        "i need", "something to", "help me", "not sure",
        "options", "alternatives", "what would you suggest",
        "problem", "issue", "trying to", "want to",
        "hold", "catch", "block", "organize", "mount", "attach"
    ]

    # Multi-feature part patterns (ALWAYS trigger design exploration)
    multi_feature_patterns = [
        # Manufacturing methods
        "machined", "milled", "cnc", "3d printed", "cast", "injection molded",
        # Enclosure/housing language
        "enclosure", "housing", "chassis", "case base",
        # Feature keywords (when combined)
        "pocket", "standoff", "boss", "rib", "slot", "cavity",
        # Electronics integration
        "pcb mount", "board mount", "electronics",
        # Material + function patterns
        "aluminum part", "plastic part", "steel part",
        "aluminum bracket", "plastic housing", "steel base"
    ]

    # Feature enumeration patterns (indicates multiple features)
    feature_enumeration = [
        "mounting hole", "corner hole", "internal pocket",
        "four hole", "two boss", "corner mount",
        "with pocket", "with boss", "with standoff",
        "holes and", "pocket and", "bosses and", "slots and"
    ]

    design_exploration_exclude = [
        "exactly", "precisely", "just make", "just build",
        "add fillet", "add chamfer", "drill hole at"
    ]

    has_include_signal = any(kw in request_lower for kw in design_exploration_include)
    has_multi_feature = any(kw in request_lower for kw in multi_feature_patterns)
    has_feature_enumeration = any(kw in request_lower for kw in feature_enumeration)
    has_exclude_signal = any(kw in request_lower for kw in design_exploration_exclude)

    # Check for dimension indicators (suggests specific geometry, not problem description)
    # But multi-feature parts still need exploration even with overall dimensions
    has_dimensions = any(unit in request_lower for unit in ["mm", "cm", "inch", "inches"])

    # Trigger design exploration for:
    # 1. Problem-first requests without dimensions
    # 2. Multi-feature parts (even with some dimensions - need feature details)
    # 3. Feature enumeration patterns
    if has_multi_feature or has_feature_enumeration:
        if not has_exclude_signal:
            required.append("design_exploration")
            # Multi-feature parts ALWAYS need sketch_tools and 3d_modeling
            if "sketch_tools" not in required:
                required.append("sketch_tools")
            if "3d_modeling" not in required:
                required.append("3d_modeling")
    elif has_include_signal and not has_exclude_signal and not has_dimensions:
        required.append("design_exploration")
        # Design exploration typically needs modeling tools
        if "sketch_tools" not in required:
            required.append("sketch_tools")
        if "3d_modeling" not in required:
            required.append("3d_modeling")

    # Remove duplicates
    required = list(dict.fromkeys(required))

    # If nothing matched, include common clusters as safety
    if len(required) == 1:  # Only 'core'
        required.extend(["sketch_tools", "3d_modeling"])
        optional.extend(["inspection", "selection"])

    return {
        "required": required,
        "optional": optional,
        "reasoning": f"Fallback routing using keyword matching ({fallback_reason})",
        "confidence": "low",
        "fallback": True,
        "fallback_reason": fallback_reason,
        "routing_source": "keyword_fallback",
    }


def get_routing_summary(routing_result: Dict[str, Any]) -> str:
    """
    Generate human-readable summary of routing decision.

    Args:
        routing_result: Output from route_request()

    Returns:
        Formatted string summarizing the routing
    """
    required = routing_result.get("required", [])
    optional = routing_result.get("optional", [])
    reasoning = routing_result.get("reasoning", "No reasoning")
    confidence = routing_result.get("confidence", "unknown")

    summary = [
        f"Routing Decision (confidence: {confidence}):",
        f"  Required clusters: {', '.join(required)}",
    ]

    if optional:
        summary.append(f"  Optional clusters: {', '.join(optional)}")

    summary.append(f"  Reasoning: {reasoning}")

    return "\n".join(summary)


__all__ = [
    "route_request",
    "get_routing_summary",
    "ROUTING_PATTERNS",
]
