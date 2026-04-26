"""
Session-based logging system for tracking LLM API calls with full context.

This module provides structured logging for CAD agent sessions, organizing logs
hierarchically: session → iteration → API call. Each API call logs both the raw
LLM response and the session-unique context (excluding hardcoded system prompts).

Folder Structure:
    runs/
      └── session_<session_id>_<timestamp>/
          └── iteration_<iteration_number>/
              └── api_call_<call_number>/
                  ├── llm_response.json          (raw LLM output)
                  └── session_context.json        (dynamic context only)

Usage:
    from backend.session_logger import initialize_session, log_api_call

    # At start of execution
    session_path = initialize_session(session_id)

    # For each API call
    log_api_call(
        session_path=session_path,
        iteration=1,
        api_call_num=1,
        llm_response=response_dict,
        context=context_dict
    )
"""

import json
import logging
import os
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Root directory for all session logs
RUNS_DIR = Path(__file__).parent.parent / "runs"

# Minimum free disk space required (100MB)
MIN_DISK_SPACE_MB = 100


def _check_disk_space() -> bool:
    """
    Check if sufficient disk space is available for logging.

    Returns:
        True if sufficient space available, False otherwise
    """
    try:
        stat = shutil.disk_usage(RUNS_DIR.parent)
        free_mb = stat.free / (1024 * 1024)
        return free_mb >= MIN_DISK_SPACE_MB
    except Exception as e:
        logger.warning(f"Failed to check disk space: {e}")
        return True  # Assume OK if check fails


def _sanitize_for_json(obj: Any, max_depth: int = 10, current_depth: int = 0) -> Any:
    """
    Recursively sanitize objects for JSON serialization.

    Handles non-serializable objects like sets, bytes, custom classes, etc.

    Args:
        obj: Object to sanitize
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth

    Returns:
        JSON-serializable version of obj
    """
    if current_depth >= max_depth:
        return "<max_depth_reached>"

    # Handle None, bool, int, float, str - already JSON serializable
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Handle sequences (list, tuple, set, etc.)
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item, max_depth, current_depth + 1) for item in obj]

    if isinstance(obj, set):
        return [_sanitize_for_json(item, max_depth, current_depth + 1) for item in sorted(obj, key=str)]

    # Handle dictionaries
    if isinstance(obj, dict):
        return {
            str(key): _sanitize_for_json(value, max_depth, current_depth + 1)
            for key, value in obj.items()
        }

    # Handle bytes
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes: {len(obj)} bytes>"

    # Handle objects with model_dump (Pydantic models)
    if hasattr(obj, "model_dump"):
        try:
            return _sanitize_for_json(obj.model_dump(), max_depth, current_depth + 1)
        except Exception:
            pass

    # Handle objects with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return _sanitize_for_json(obj.__dict__, max_depth, current_depth + 1)
        except Exception:
            pass

    # Fallback: convert to string
    try:
        return str(obj)
    except Exception:
        return "<non-serializable>"


def _prepare_llm_response_for_logging(llm_response: Any) -> Any:
    """
    Strip request-side fields (system prompt, tool schemas, etc.) before
    writing llm_response.json. We only want the model's output and basic
    metadata so we don't leak prompts or large tool definitions.

    Args:
        llm_response: Raw response object or dictionary from the LLM SDK

    Returns:
        JSON-serializable object focused on output/usage fields
    """
    try:
        if hasattr(llm_response, "model_dump"):
            data = llm_response.model_dump()
        elif isinstance(llm_response, dict):
            data = dict(llm_response)
        else:
            return _sanitize_for_json(llm_response)

        # Remove request echo fields that include the full system prompt/tools
        drop_keys = {
            "instructions",
            "input",
            "messages",
            "tools",
            "parallel_tool_calls",
            "tool_choice",
            "temperature",
            "top_logprobs",
            "truncation",
            "metadata",
            "text",
            "prompt_cache_retention",
            "store",
        }
        for key in drop_keys:
            data.pop(key, None)

        # Keep a minimal, output-centric view
        allowed_keys = {
            "id",
            "created_at",
            "model",
            "object",
            "output",
            "content",
            "choices",
            "usage",
            "stop_reason",
            "finish_reason",
            "status",
            "error",
            "incomplete_details",
            "system_fingerprint",
            "service_tier",
            "billing",
            "warnings",
        }

        slim_data = {k: v for k, v in data.items() if k in allowed_keys and v is not None}

        return _sanitize_for_json(slim_data if slim_data else data)

    except Exception as e:
        logger.warning(f"Failed to prepare LLM response for logging: {e}")
        return _sanitize_for_json(llm_response)


def _find_latest_session_dir(session_id: str) -> Optional[Path]:
    """Return the newest runs/ directory for a given session_id."""
    if not session_id:
        return None

    candidates = sorted(
        RUNS_DIR.glob(f"*_session_{session_id}"),
        key=lambda p: p.name,
        reverse=True,
    )
    for path in candidates:
        if path.is_dir():
            return path
    return None


def update_iteration_feedback(session_id: str, iteration: int, verdict: str) -> bool:
    """
    Persist a thumbs-up/down verdict at the session level into session_metadata.json.
    Each new feedback overwrites the previous session feedback.

    Args:
        session_id: Session identifier from Fusion/backend
        iteration: 1-based iteration index (used for logging/display only)
        verdict: "positive" or "negative" (case-insensitive)

    Returns:
        True if metadata was updated, False otherwise.
    """
    try:
        session_dir = _find_latest_session_dir(session_id)
        if not session_dir:
            logger.warning("No session directory found for session_id=%s", session_id)
            return False

        meta_path = session_dir / "session_metadata.json"
        if not meta_path.exists():
            logger.warning("session_metadata.json missing at %s", meta_path)
            return False

        verdict_norm = (verdict or "").strip().lower()
        if verdict_norm in {"up", "thumbs_up", "positive", "good"}:
            verdict_norm = "positive"
        elif verdict_norm in {"down", "thumbs_down", "negative", "bad"}:
            verdict_norm = "negative"
        else:
            logger.warning("Unsupported verdict '%s'", verdict)
            return False

        if iteration is None or iteration <= 0:
            logger.warning("Invalid iteration '%s' for feedback", iteration)
            return False

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Store feedback at session level (overwrites previous feedback)
        meta["session_feedback"] = verdict_norm

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(
            "Recorded session feedback: session=%s verdict=%s (from iteration %s)",
            session_id,
            verdict_norm,
            iteration,
        )
        return True

    except Exception:
        logger.exception("Failed to update session feedback for session %s", session_id)
        return False


def _extract_session_context(
    user_request: Optional[str],
    timeline_state: Optional[Dict[str, Any]],
    messages: Optional[List[Dict[str, Any]]],
    feature_snapshot: Optional[Dict[str, Any]],
    entity_store_data: Optional[Dict[str, Any]],
    routing_result: Optional[Dict[str, Any]],
    loaded_tools: Optional[List[str]],
    model_name: Optional[str],
    reasoning_effort: Optional[str],
    iteration: int,
    max_iterations: int,
    system_prompt: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    spatial_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract session-unique context for logging.

    This function extracts both the context we build AND the actual data
    sent to the LLM API (system prompt, tools, formatted messages).

    Args:
        user_request: User's natural language request
        timeline_state: Fusion 360 timeline state (operations, marker position)
        messages: Conversation history
        feature_snapshot: Current feature state from Fusion
        entity_store_data: Selected entities (edges, faces, bodies)
        routing_result: Tool routing decision (if routing enabled)
        loaded_tools: List of tool names loaded for this iteration
        model_name: LLM model identifier
        reasoning_effort: Reasoning effort level
        iteration: Current iteration number
        max_iterations: Maximum iterations allowed
        system_prompt: Actual system prompt sent to LLM (optional)
        tools: Actual tools array sent to LLM (optional)
        spatial_context: New spatial context structure (optional)

    Returns:
        Dictionary containing session-unique context
    """
    context: Dict[str, Any] = {
        "metadata": {
            "iteration": iteration,
            "max_iterations": max_iterations,
            "model_name": model_name,
            "reasoning_effort": reasoning_effort,
            "timestamp": datetime.now().isoformat(),
        }
    }

    # User request
    if user_request:
        context["user_request"] = user_request

    # Timeline state
    if timeline_state:
        context["timeline_state"] = {
            "marker_position": timeline_state.get("marker_position"),
            "count": timeline_state.get("count"),
            "operations": timeline_state.get("operations", []),
        }

    # Conversation history - extract actual content from blocks
    if messages:
        filtered_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", [])

            # Extract readable content from content blocks
            content_items = []
            if isinstance(content, str):
                content_items.append({"type": "text", "text": content[:1000]})
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "text":
                            text = block.get("text", "")
                            # Truncate very long text but keep it readable
                            content_items.append({
                                "type": "text",
                                "text": text[:1000] + "..." if len(text) > 1000 else text
                            })
                        elif block_type == "tool_use":
                            # Show tool name and key inputs
                            content_items.append({
                                "type": "tool_use",
                                "id": block.get("id"),
                                "name": block.get("name"),
                                "input_preview": str(block.get("input", {}))
                            })
                        elif block_type == "tool_result":
                            # Show tool result summary (full content for debugging spatial summary)
                            tool_content = block.get("content", "")
                            if isinstance(tool_content, list):
                                tool_content = str(tool_content)
                            content_items.append({
                                "type": "tool_result",
                                "tool_use_id": block.get("tool_use_id"),
                                "is_error": block.get("is_error", False),
                                "content": str(tool_content)
                            })
                        elif block_type == "image":
                            content_items.append({
                                "type": "image",
                                "source": "base64_data_omitted"
                            })

            filtered_messages.append({
                "role": role,
                "content": content_items
            })

        context["conversation_history"] = {
            "total_messages": len(messages),
            "recent_messages": filtered_messages[-5:],  # Last 5 messages with full content
        }

    # Feature snapshot
    if feature_snapshot:
        features = feature_snapshot.get("features", [])
        context["feature_snapshot"] = {
            "success": feature_snapshot.get("success"),
            "feature_count": len(features),
            "features": [
                {
                    "entity_token": f.get("entity_token"),
                    "name": f.get("name"),
                    "type": f.get("type"),
                }
                for f in features[:10]  # First 10 features only
            ],
        }

    # Entity store (selected geometry)
    if entity_store_data:
        context["entity_store"] = entity_store_data

    # Spatial context (new nested per-body structure from SPATIAL_CONTEXT_OVERHAUL.md)
    if spatial_context:
        # Log a summary of the spatial context to avoid huge log files
        bodies = spatial_context.get("bodies", [])
        context["spatial_context"] = {
            "units": spatial_context.get("units", "mm"),
            "body_count": len(bodies),
            "bodies_summary": [
                {
                    "id": b.get("id"),
                    "name": b.get("name"),
                    "vertex_count": len(b.get("vertices", [])),
                    "face_count": len(b.get("faces", [])),
                    "edge_count": len(b.get("edges", [])),
                    "bbox": b.get("bbox"),
                }
                for b in bodies[:5]  # First 5 bodies only
            ],
        }

    # Routing result
    if routing_result:
        context["routing"] = {
            "required_clusters": routing_result.get("required", []),
            "optional_clusters": routing_result.get("optional", []),
            "reasoning": routing_result.get("reasoning"),
            "confidence": routing_result.get("confidence"),
        }

    # Loaded tools (names only, not full schemas)
    if loaded_tools:
        context["loaded_tools"] = loaded_tools

    # System prompt metadata only (no full text to avoid logging static prompts)
    if system_prompt:
        context["system_prompt"] = {
            "length": len(system_prompt),
            "sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "redacted": True,
        }

    # Tools array sent to LLM (actual tool schemas)
    if tools:
        context["tools_sent_to_llm"] = {
            "count": len(tools),
            "tool_names": [t.get("name") for t in tools if isinstance(t, dict)],
            "tool_schemas": tools  # Include full schemas
        }

    return context


def initialize_session(session_id: str) -> Optional[str]:
    """
    Initialize a new session logging directory.

    Creates the session folder with timestamp in the runs/ directory.

    Args:
        session_id: Unique session identifier

    Returns:
        Path to session directory as string, or None if initialization failed
    """
    try:
        # Check disk space
        if not _check_disk_space():
            logger.warning(
                f"Insufficient disk space for session logging (< {MIN_DISK_SPACE_MB}MB free). "
                "Session logging disabled."
            )
            return None

        # Create runs directory if it doesn't exist
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

        # Create session directory with timestamp first (for chronological sorting)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir_name = f"{timestamp}_session_{session_id}"
        session_path = RUNS_DIR / session_dir_name

        session_path.mkdir(parents=True, exist_ok=True)

        # Create metadata file
        metadata = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "session_path": str(session_path),
            "model_costs": {},  # per-model cumulative costs/tokens
        }

        metadata_file = session_path / "session_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Session logging initialized: {session_path}")
        return str(session_path)

    except Exception as e:
        logger.error(f"Failed to initialize session logging: {e}", exc_info=True)
        return None


def _extract_reasoning_from_response(llm_response: Any, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Extract reasoning summaries from LLM responses.

    Supports:
    - OpenAI GPT-5/o-series extended thinking (reasoning tokens + summaries)
    - Claude Extended Thinking (thinking blocks with signatures)

    Args:
        llm_response: Raw response from LLM API
        model_name: Model name for context

    Returns:
        Dictionary with reasoning data, or None if no reasoning found
    """
    try:
        # Convert to dict if needed
        if hasattr(llm_response, "model_dump"):
            data = llm_response.model_dump()
        elif isinstance(llm_response, dict):
            data = llm_response
        else:
            return None

        reasoning_data = {
            "model": model_name,
            "response_id": data.get("id", "unknown"),
            "reasoning_tokens": 0,
            "summaries": [],
            "thinking_blocks": [],  # For Claude extended thinking
            "has_reasoning": False,
        }

        # Extract reasoning tokens from usage (OpenAI)
        usage = data.get("usage", {})
        if usage:
            output_tokens_details = usage.get("output_tokens_details", {})
            reasoning_tokens = output_tokens_details.get("reasoning_tokens", 0)
            reasoning_data["reasoning_tokens"] = reasoning_tokens

            if reasoning_tokens > 0:
                reasoning_data["has_reasoning"] = True

        # Extract reasoning summaries from output items (OpenAI Responses API)
        output_items = data.get("output", [])
        if output_items:
            for item in output_items:
                if isinstance(item, dict) and item.get("type") == "reasoning":
                    summary_array = item.get("summary", [])
                    for summary_obj in summary_array:
                        if isinstance(summary_obj, dict):
                            text = summary_obj.get("text", "")
                            if text and text.strip():
                                reasoning_data["summaries"].append(text.strip())

        # Extract thinking blocks from content (Claude Extended Thinking)
        content_blocks = data.get("content", [])
        if content_blocks:
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_text = block.get("thinking", "")
                    signature = block.get("signature", "")

                    if thinking_text and thinking_text.strip():
                        reasoning_data["thinking_blocks"].append({
                            "thinking": thinking_text.strip(),
                            "signature": signature
                        })
                        reasoning_data["has_reasoning"] = True

        # If we found reasoning tokens but no summaries, note this (OpenAI limitation)
        if reasoning_data["reasoning_tokens"] > 0 and not reasoning_data["summaries"]:
            reasoning_data["note"] = "Model used reasoning tokens but API did not provide summary text (known OpenAI limitation)"

        # Return None if no reasoning at all
        if not reasoning_data["has_reasoning"]:
            return None

        return reasoning_data

    except Exception as e:
        logger.warning(f"Failed to extract reasoning from response: {e}")
        return None


def _get_model_pricing(model_name: Optional[str]) -> Optional[Dict[str, float]]:
    """Return pricing config for a given model."""
    if not model_name:
        return None
    return MODEL_PRICING.get(model_name)


def _extract_usage_stats(llm_response: Any) -> Dict[str, int]:
    """
    Normalize token usage across providers into a flat dict.

    Returns:
        {
            "input_tokens": int,
            "output_tokens": int,
            "cache_creation_tokens": int,
            "cache_read_tokens": int,
            "output_reasoning_tokens": int,
        }
    """
    stats = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "output_reasoning_tokens": 0,
        "_input_tokens_from_cacheable_prompt": False,  # OpenAI: input_tokens include cacheable portion
    }

    try:
        usage_obj = getattr(llm_response, "usage", None)
        if usage_obj is None and isinstance(llm_response, dict):
            usage_obj = llm_response.get("usage")

        def _get_val(obj: Any, key: str, default: Any = 0) -> Any:
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # Base token counts (provider-specific fields added below)
        input_tokens_total = int(_get_val(usage_obj, "input_tokens", 0) or 0)
        stats["input_tokens"] = input_tokens_total
        stats["output_tokens"] = int(_get_val(usage_obj, "output_tokens", 0) or 0)
        stats["cache_creation_tokens"] = int(_get_val(usage_obj, "cache_creation_input_tokens", 0) or 0)
        stats["cache_read_tokens"] = int(_get_val(usage_obj, "cache_read_input_tokens", 0) or 0)

        # OpenAI Responses API: cache hits are reported in input_tokens_details.cached_tokens
        input_details = _get_val(usage_obj, "input_tokens_details")
        if input_details is None:
            input_details = _get_val(usage_obj, "prompt_tokens_details")  # Chat Completions naming

        cached_tokens = 0
        if input_details:
            stats["_input_tokens_from_cacheable_prompt"] = True
            cached_tokens = int(_get_val(input_details, "cached_tokens", 0) or 0)
            # Treat cached tokens as cache reads; uncached portion becomes both input + cache write
            stats["cache_read_tokens"] = max(stats["cache_read_tokens"], cached_tokens)
            uncached_tokens = max(input_tokens_total - cached_tokens, 0)
            stats["input_tokens"] = uncached_tokens
            # If the provider didn't report cache_creation_input_tokens (OpenAI), infer it
            if stats["cache_creation_tokens"] == 0 and (cached_tokens > 0 or input_details is not None):
                stats["cache_creation_tokens"] = uncached_tokens

        output_details = _get_val(usage_obj, "output_tokens_details")
        if output_details:
            stats["output_reasoning_tokens"] = int(_get_val(output_details, "reasoning_tokens", 0) or 0)

    except Exception:
        logger.debug("Failed to extract usage stats from response", exc_info=True)

    return stats


def _calculate_call_cost(model_name: Optional[str], usage_stats: Dict[str, int]) -> Optional[Dict[str, float]]:
    """
    Compute per-call cost using static pricing table.

    Returns None if pricing is unavailable for the model.
    """
    pricing = _get_model_pricing(model_name)
    if not pricing:
        return None

    base_in = pricing["input_per_mtok"]
    base_out = pricing["output_per_mtok"]
    cache_write_mult = pricing.get("cache_write_multiplier", 1.0)
    cache_read_mult = pricing.get("cache_read_multiplier", 1.0)

    cache_write_tokens = usage_stats.get("cache_creation_tokens", 0)
    cache_read_tokens = usage_stats.get("cache_read_tokens", 0)
    uncached_input_tokens = max(usage_stats.get("input_tokens", 0), 0)
    output_tokens = max(usage_stats.get("output_tokens", 0), 0)

    # OpenAI Reports cacheable prompts in input_tokens_details; those uncached tokens are already
    # represented as cache_creation_tokens, so exclude them from the separate uncached bucket
    if usage_stats.get("_input_tokens_from_cacheable_prompt"):
        uncached_for_cost = 0
    else:
        uncached_for_cost = uncached_input_tokens

    total_input_tokens = cache_write_tokens + cache_read_tokens + uncached_for_cost

    input_cost = (uncached_for_cost / 1_000_000) * base_in
    cache_write_cost = (cache_write_tokens / 1_000_000) * base_in * cache_write_mult
    cache_read_cost = (cache_read_tokens / 1_000_000) * base_in * cache_read_mult
    output_cost = (output_tokens / 1_000_000) * base_out

    actual_total = input_cost + cache_write_cost + cache_read_cost + output_cost
    baseline_no_cache = (total_input_tokens / 1_000_000) * base_in + output_cost
    savings = baseline_no_cache - actual_total

    return {
        "actual_total_usd": round(actual_total, 6),
        "baseline_without_cache_usd": round(baseline_no_cache, 6),
        "savings_usd": round(savings, 6),
    }


def _update_model_costs(
    session_dir: Path,
    model_name: Optional[str],
    usage_stats: Dict[str, int],
    cost_summary: Optional[Dict[str, float]],
) -> None:
    """Aggregate per-model cost + token usage into session_metadata.json."""
    if not model_name:
        return

    try:
        meta_path = session_dir / "session_metadata.json"
        if not meta_path.exists():
            return

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_costs = meta.get("model_costs") or {}
        entry = model_costs.get(model_name) or {
            "call_count": 0,
            "tokens": {
                "input": 0,
                "output": 0,
                "cache_write": 0,
                "cache_read": 0,
                "output_reasoning": 0,
            },
            "cost_usd": {
                "actual_total": 0.0,
                "baseline_without_cache": 0.0,
                "savings": 0.0,
            },
        }

        entry["call_count"] += 1
        entry["tokens"]["input"] += usage_stats.get("input_tokens", 0)
        entry["tokens"]["output"] += usage_stats.get("output_tokens", 0)
        entry["tokens"]["cache_write"] += usage_stats.get("cache_creation_tokens", 0)
        entry["tokens"]["cache_read"] += usage_stats.get("cache_read_tokens", 0)
        entry["tokens"]["output_reasoning"] += usage_stats.get("output_reasoning_tokens", 0)

        if cost_summary:
            entry["cost_usd"]["actual_total"] = round(
                entry["cost_usd"]["actual_total"] + cost_summary.get("actual_total_usd", 0.0), 6
            )
            entry["cost_usd"]["baseline_without_cache"] = round(
                entry["cost_usd"]["baseline_without_cache"] + cost_summary.get("baseline_without_cache_usd", 0.0), 6
            )
            entry["cost_usd"]["savings"] = round(
                entry["cost_usd"]["savings"] + cost_summary.get("savings_usd", 0.0), 6
            )

        model_costs[model_name] = entry
        meta["model_costs"] = model_costs
        meta["model_costs_last_updated"] = datetime.now().isoformat()

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    except Exception:
        logger.debug("Failed to update model_costs in session_metadata", exc_info=True)


def log_api_call(
    session_path: str,
    iteration: int,
    api_call_num: int,
    llm_response: Any,
    context: Dict[str, Any],
) -> bool:
    """
    Log a single API call with response and context.

    Creates the iteration and API call directories, then writes:
    1. llm_response.json - Raw LLM output
    2. session_context.json - Session-unique context

    Args:
        session_path: Path to session directory
        iteration: Current iteration number
        api_call_num: API call number within this iteration
        llm_response: Raw response from LLM (dict or object with model_dump)
        context: Session context dictionary

    Returns:
        True if logging succeeded, False otherwise
    """
    try:
        # Check disk space before writing
        if not _check_disk_space():
            logger.warning("Insufficient disk space, skipping API call logging")
            return False

        # Create iteration directory
        session_dir = Path(session_path)
        iteration_dir = session_dir / f"iteration_{iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        # Create API call directory
        api_call_dir = iteration_dir / f"api_call_{api_call_num:03d}"
        api_call_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize and write LLM response (output only; strip request echo fields)
        response_data = _prepare_llm_response_for_logging(llm_response)
        response_file = api_call_dir / "llm_response.json"
        with open(response_file, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)

        # Sanitize and write session context
        context_data = _sanitize_for_json(context)
        context_file = api_call_dir / "session_context.json"
        with open(context_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False)

        # Extract and write reasoning summary (if present)
        model_name = context.get("metadata", {}).get("model_name") if isinstance(context, dict) else None
        reasoning_data = _extract_reasoning_from_response(llm_response, model_name)
        if reasoning_data:
            reasoning_file = api_call_dir / "reasoning.json"
            with open(reasoning_file, "w", encoding="utf-8") as f:
                json.dump(reasoning_data, f, indent=2, ensure_ascii=False)
            logger.debug(
                f"Extracted reasoning: {reasoning_data['reasoning_tokens']} tokens, "
                f"{len(reasoning_data['summaries'])} summaries"
            )

        # Update per-model cost totals in session_metadata.json
        usage_stats = _extract_usage_stats(llm_response)
        try:
            cost_summary = _calculate_call_cost(model_name, usage_stats)
            _update_model_costs(session_dir, model_name, usage_stats, cost_summary)
        except Exception:
            logger.debug("Failed to update per-model costs", exc_info=True)

        logger.debug(
            f"Logged API call: session={session_dir.name}, "
            f"iteration={iteration}, call={api_call_num}"
        )
        return True

    except Exception as e:
        logger.error(
            f"Failed to log API call (iteration={iteration}, call={api_call_num}): {e}",
            exc_info=True
        )
        return False


__all__ = [
    "initialize_session",
    "log_api_call",
    "_extract_session_context",
    "update_iteration_feedback",
    "log_cache_metrics",
    "finalize_session_cache_summary",
]


# ============================================================================
# CACHE METRICS TRACKING
# ============================================================================

# Claude Sonnet 4.5 pricing (per million tokens)
CACHE_PRICING = {
    "base_input": 3.00,      # $/MTok for regular input
    "cache_write": 3.75,     # $/MTok for 5m cache write (1.25x)
    "cache_write_1h": 6.00,  # $/MTok for 1h cache write (2x)
    "cache_read": 0.30,      # $/MTok for cache read (0.1x)
}

# Estimated per-model pricing (USD per million tokens)
# These values are static and meant for cost estimation in logs only.
MODEL_PRICING = {
    # Anthropic (latest as of 2025-12)
    "claude-sonnet-4-5-20250929": {
        "input_per_mtok": 3.0,
        "output_per_mtok": 15.0,
        "cache_write_multiplier": 1.25,  # aligns with CACHE_PRICING
        "cache_read_multiplier": 0.10,
    },
    "claude-opus-4-5-20251101": {
        "input_per_mtok": 15.0,
        "output_per_mtok": 75.0,
        "cache_write_multiplier": 1.25,
        "cache_read_multiplier": 0.10,
    },
    "claude-opus-4-1-20250514": {
        "input_per_mtok": 15.0,
        "output_per_mtok": 75.0,
        "cache_write_multiplier": 1.25,
        "cache_read_multiplier": 0.10,
    },
    # OpenAI (approximate, for logging only)
    "gpt-5": {
        "input_per_mtok": 5.0,
        "output_per_mtok": 15.0,
        "cache_write_multiplier": 1.0,  # Prompt cache creation billed at standard input rate
        "cache_read_multiplier": 0.25,  # Cached reads ~25% of input price (OpenAI prompt cache)
    },
    "gpt-5.2": {
        "input_per_mtok": 5.0,
        "output_per_mtok": 15.0,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.25,
    },
    "gpt-5-mini": {
        "input_per_mtok": 1.0,
        "output_per_mtok": 3.0,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.25,
    },
    # Google Gemini (estimate for log-only costing)
    "gemini-3-pro-preview": {
        "input_per_mtok": 3.5,
        "output_per_mtok": 10.0,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 1.0,
    },
}


def log_cache_metrics(
    session_path: str,
    iteration: int,
    api_call_num: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
    input_tokens: int,
    model_name: Optional[str] = None,
) -> bool:
    """
    Log cache metrics for a single API call and update session cumulative stats.

    Args:
        session_path: Path to session directory
        iteration: Current iteration number
        api_call_num: API call number within this iteration
        cache_creation_tokens: Tokens written to cache
        cache_read_tokens: Tokens read from cache
        input_tokens: Tokens not from cache (after last breakpoint)
        model_name: Model name for pricing context

    Returns:
        True if logging succeeded, False otherwise
    """
    try:
        session_dir = Path(session_path)
        if not session_dir.exists():
            return False

        # Calculate costs (prefer per-model pricing when available)
        pricing = MODEL_PRICING.get(model_name) if model_name else None
        base_in = pricing["input_per_mtok"] if pricing else CACHE_PRICING["base_input"]
        cache_write_mult = pricing.get("cache_write_multiplier", CACHE_PRICING["cache_write"] / CACHE_PRICING["base_input"]) if pricing else CACHE_PRICING["cache_write"] / CACHE_PRICING["base_input"]
        cache_read_mult = pricing.get("cache_read_multiplier", CACHE_PRICING["cache_read"] / CACHE_PRICING["base_input"]) if pricing else CACHE_PRICING["cache_read"] / CACHE_PRICING["base_input"]

        cache_write_cost = (cache_creation_tokens / 1_000_000) * base_in * cache_write_mult
        cache_read_cost = (cache_read_tokens / 1_000_000) * base_in * cache_read_mult
        uncached_cost = (input_tokens / 1_000_000) * base_in
        actual_cost = cache_write_cost + cache_read_cost + uncached_cost

        # What would it have cost without caching?
        total_tokens = cache_creation_tokens + cache_read_tokens + input_tokens
        baseline_cost = (total_tokens / 1_000_000) * base_in
        savings = baseline_cost - actual_cost

        # Calculate hit rate
        cacheable_tokens = cache_creation_tokens + cache_read_tokens
        hit_rate = (cache_read_tokens / cacheable_tokens * 100) if cacheable_tokens > 0 else 0.0

        call_metrics = {
            "iteration": iteration,
            "api_call": api_call_num,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "tokens": {
                "cache_write": cache_creation_tokens,
                "cache_read": cache_read_tokens,
                "uncached_input": input_tokens,
                "total": total_tokens,
            },
            "cost_usd": {
                "cache_write": round(cache_write_cost, 6),
                "cache_read": round(cache_read_cost, 6),
                "uncached": round(uncached_cost, 6),
                "actual_total": round(actual_cost, 6),
                "baseline_without_cache": round(baseline_cost, 6),
                "savings": round(savings, 6),
            },
            "hit_rate_pct": round(hit_rate, 1),
        }

        # Write per-call cache metrics
        api_call_dir = session_dir / f"iteration_{iteration:03d}" / f"api_call_{api_call_num:03d}"
        if api_call_dir.exists():
            cache_file = api_call_dir / "cache_metrics.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(call_metrics, f, indent=2)

        # Update session-level cumulative metrics
        _update_session_cache_totals(session_dir, call_metrics)

        logger.debug(
            f"Cache metrics: write={cache_creation_tokens}, read={cache_read_tokens}, "
            f"hit_rate={hit_rate:.1f}%, savings=${savings:.4f}"
        )
        return True

    except Exception as e:
        logger.warning(f"Failed to log cache metrics: {e}")
        return False


def _update_session_cache_totals(session_dir: Path, call_metrics: Dict[str, Any]) -> None:
    """Update cumulative cache metrics in session_metadata.json."""
    try:
        meta_path = session_dir / "session_metadata.json"
        if not meta_path.exists():
            return

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Initialize cache_summary if not present
        if "cache_summary" not in meta:
            meta["cache_summary"] = {
                "total_api_calls": 0,
                "total_cache_write_tokens": 0,
                "total_cache_read_tokens": 0,
                "total_uncached_tokens": 0,
                "total_cost_usd": 0.0,
                "total_baseline_cost_usd": 0.0,
                "total_savings_usd": 0.0,
            }

        summary = meta["cache_summary"]
        tokens = call_metrics["tokens"]
        costs = call_metrics["cost_usd"]

        summary["total_api_calls"] += 1
        summary["total_cache_write_tokens"] += tokens["cache_write"]
        summary["total_cache_read_tokens"] += tokens["cache_read"]
        summary["total_uncached_tokens"] += tokens["uncached_input"]
        summary["total_cost_usd"] = round(summary["total_cost_usd"] + costs["actual_total"], 6)
        summary["total_baseline_cost_usd"] = round(
            summary["total_baseline_cost_usd"] + costs["baseline_without_cache"], 6
        )
        summary["total_savings_usd"] = round(summary["total_savings_usd"] + costs["savings"], 6)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.warning(f"Failed to update session cache totals: {e}")


def finalize_session_cache_summary(session_path: str) -> Optional[Dict[str, Any]]:
    """
    Finalize and return the cache metrics summary for a session.

    Call this at session end to get the final cache performance report.

    Args:
        session_path: Path to session directory

    Returns:
        Dictionary with final cache summary, or None if unavailable
    """
    try:
        session_dir = Path(session_path)
        meta_path = session_dir / "session_metadata.json"

        if not meta_path.exists():
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        summary = meta.get("cache_summary")
        if not summary:
            return None

        # Calculate overall hit rate
        total_cacheable = summary["total_cache_write_tokens"] + summary["total_cache_read_tokens"]
        overall_hit_rate = (
            (summary["total_cache_read_tokens"] / total_cacheable * 100)
            if total_cacheable > 0
            else 0.0
        )

        # Calculate savings percentage
        savings_pct = (
            (summary["total_savings_usd"] / summary["total_baseline_cost_usd"] * 100)
            if summary["total_baseline_cost_usd"] > 0
            else 0.0
        )

        final_summary = {
            **summary,
            "overall_cache_hit_rate_pct": round(overall_hit_rate, 1),
            "overall_savings_pct": round(savings_pct, 1),
            "finalized_at": datetime.now().isoformat(),
        }

        # Update metadata with final summary
        meta["cache_summary"] = final_summary
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Session cache summary: {summary['total_api_calls']} calls, "
            f"hit_rate={overall_hit_rate:.1f}%, "
            f"savings=${summary['total_savings_usd']:.4f} ({savings_pct:.1f}%)"
        )

        return final_summary

    except Exception as e:
        logger.warning(f"Failed to finalize session cache summary: {e}")
        return None
