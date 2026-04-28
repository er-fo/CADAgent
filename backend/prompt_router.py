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
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI

from .prompt_structure import get_tool_catalog_text, CLUSTER_TOOL_MAPPING

logger = logging.getLogger(__name__)

def _build_routing_client(api_keys: Optional[Dict[str, str]] = None) -> Optional[AsyncOpenAI]:
    """Create routing client from per-session BYOK key with env fallback."""
    if _router_uses_bedrock():
        api_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if not api_key or not api_key.strip():
            return None
        return AsyncOpenAI(api_key=api_key.strip(), base_url=ROUTER_BEDROCK_BASE_URL)

    api_key = None
    if api_keys:
        api_key = api_keys.get("openai_api_key") or api_keys.get("OPENAI_API_KEY")
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        return None

    client_kwargs: Dict[str, Any] = {"api_key": api_key.strip()}
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


def _router_uses_bedrock() -> bool:
    """Return True when the configured router model should use Bedrock."""
    model_name = (ROUTER_MODEL or "").strip().lower()
    return any(model_name.startswith(prefix) for prefix in _BEDROCK_ROUTER_PREFIXES)

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
   - jump_to_timeline_position, delete_feature → timeline
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
        logger.info(f"Routing request: {user_request[:100]}...")

        routing_client = _build_routing_client(api_keys)
        if routing_client is None:
            logger.warning("No OpenAI API key available for prompt routing; using fallback routing")
            return _fallback_routing(user_request)

        # Get tool catalog for router
        tool_catalog = get_tool_catalog_text()

        # Build router prompt (system role) and send user request separately
        system_prompt = ROUTER_PROMPT.format(tool_catalog=tool_catalog)

        # Extract recent conversation context (last 2-3 turns) to help router understand follow-ups
        context_summary = _extract_conversation_context(conversation_history)

        # Extract build plan operations if available
        build_plan_context = _extract_operations_from_build_plan(build_plan)

        # Build user message with optional context
        message_parts = []

        if context_summary:
            message_parts.append(f"**Recent conversation context:**\n{context_summary}")

        if build_plan_context:
            message_parts.append(build_plan_context)

        message_parts.append(f"**Current request:**\n{user_request}")

        user_message = "\n\n".join(message_parts)

        # Call the configured OpenAI-compatible routing LLM
        response = await routing_client.chat.completions.create(
            model=ROUTER_MODEL,
            # GPT-5.1 requires max_completion_tokens instead of max_tokens
            max_completion_tokens=ROUTER_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        # Extract response text
        response_text = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        routing_result = json.loads(response_text)

        # Validate result structure
        if "required" not in routing_result:
            routing_result["required"] = []
        if "optional" not in routing_result:
            routing_result["optional"] = []
        if "reasoning" not in routing_result:
            routing_result["reasoning"] = "No reasoning provided"

        # Always ensure 'core' is included
        if "core" not in routing_result["required"]:
            routing_result["required"].insert(0, "core")

        # Remove duplicates
        routing_result["required"] = list(dict.fromkeys(routing_result["required"]))
        routing_result["optional"] = list(dict.fromkeys(routing_result["optional"]))

        # Add confidence estimate based on cluster count
        total_clusters = len(routing_result["required"]) + len(routing_result["optional"])
        if total_clusters <= 4:
            routing_result["confidence"] = "high"
        elif total_clusters <= 7:
            routing_result["confidence"] = "medium"
        else:
            routing_result["confidence"] = "low"

        logger.info(
            f"Routing complete: {len(routing_result['required'])} required, "
            f"{len(routing_result['optional'])} optional clusters "
            f"(confidence: {routing_result['confidence']})"
        )
        logger.debug(f"Required: {routing_result['required']}")
        logger.debug(f"Optional: {routing_result['optional']}")
        logger.debug(f"Reasoning: {routing_result['reasoning']}")

        return routing_result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse router JSON response: {e}")
        logger.error(f"Response text: {response_text}")
        return _fallback_routing(user_request)

    except Exception as e:
        logger.error(f"Routing failed: {e}", exc_info=True)
        return _fallback_routing(user_request)


def _fallback_routing(user_request: str) -> Dict[str, Any]:
    """
    Fallback routing when AI router fails.

    Uses simple keyword matching to make a best-effort routing decision.
    Better than loading ALL clusters but not as smart as AI routing.

    Args:
        user_request: The user's request

    Returns:
        Conservative routing result that should cover most use cases
    """
    logger.warning("Using fallback routing based on keywords")

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
        "reasoning": "Fallback routing using keyword matching (AI router failed)",
        "confidence": "low",
        "fallback": True
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
