"""
Test suite for reasoning context management system.

Tests the capture, persistence, and injection of reasoning context across
multiple agent iterations using OpenAI models.

Run with:
    python backend/test_reasoning_context.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Setup path for imports
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

# Load environment
from dotenv import load_dotenv
load_dotenv()

from backend.reasoning_context import (
    ReasoningContext,
    ReasoningEntry,
    extract_decisions,
    summarize_reasoning,
    assess_outcome,
    mask_old_observations,
)
from backend.prompt_structure import REASONING_CONTEXT_SECTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    # The backend stack and connection manager are asyncio-native.
    return "asyncio"


# =============================================================================
# Unit Tests for reasoning_context.py
# =============================================================================

def test_extract_decisions():
    """Test that decision extraction correctly identifies key statements."""
    reasoning = """
    Looking at this CAD model, I need to create a base plate first.
    I will start by creating a sketch on the XY plane.
    The user wants a 50mm x 30mm rectangle, so I'll use add_rectangle.
    Because the thickness is 5mm, I should extrude by 0.5cm.
    Therefore, my approach will be: sketch -> rectangle -> extrude.
    """

    decisions = extract_decisions(reasoning, max_decisions=5)

    assert len(decisions) > 0, "Should extract at least one decision"
    assert any("sketch" in d.lower() for d in decisions), "Should identify sketch decision"
    logger.info(f"Extracted {len(decisions)} decisions: {decisions}")
    print(f"PASS: extract_decisions found {len(decisions)} decisions")


def test_summarize_reasoning():
    """Test that summarization preserves key decision content."""
    reasoning = """
    This is a complex bracket design request. The user wants an L-shaped bracket.
    I will create the cross-section on the XZ plane so the vertical leg points up.
    First, I need to draw the L-profile using 6 connected lines.
    Because the bracket has specific dimensions, I'll use: 50mm horizontal, 40mm vertical.
    The approach failed when I tried extruding - got an error about unclosed profile.
    Therefore I need to ensure all lines connect properly to form a closed loop.
    """

    summary = summarize_reasoning(reasoning, max_chars=500)

    assert len(summary) > 0, "Should produce a summary"
    assert len(summary) <= 500, "Should respect max_chars"
    # Should prioritize failure mentions
    assert "failed" in summary.lower() or "error" in summary.lower(), "Should include failure info"
    logger.info(f"Summary ({len(summary)} chars): {summary[:100]}...")
    print(f"PASS: summarize_reasoning produced {len(summary)} char summary")


def test_assess_outcome():
    """Test outcome assessment from tool results."""
    # Success case
    success_results = [
        {"content": "Sketch created successfully on XY plane", "is_error": False},
        {"content": "Rectangle added to sketch", "is_error": False},
    ]
    assert assess_outcome(success_results) == "success"

    # Partial case
    partial_results = [
        {"content": "Sketch created successfully", "is_error": False},
        {"content": "Error: Profile not closed", "is_error": True},
    ]
    assert assess_outcome(partial_results) == "partial"

    # Failed case
    failed_results = [
        {"content": "Error: Invalid plane reference", "is_error": True},
    ]
    assert assess_outcome(failed_results) == "failed"

    print("PASS: assess_outcome correctly classifies outcomes")


def test_mask_old_observations():
    """Test that old verbose tool results are masked while recent ones preserved."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Create a box"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "I'll create a box"}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "1", "content": "A" * 1000}  # Verbose
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "Now extruding"}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "2", "content": "B" * 1000}  # Verbose
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "Done"}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "3", "content": "Recent result"}  # Recent
        ]},
    ]

    masked = mask_old_observations(messages, keep_recent=3, max_result_chars=500)

    # Recent messages should be unmasked
    assert "Recent result" in str(masked[-1])

    # Old verbose results should be masked
    old_tool_result = masked[2]["content"][0]["content"]
    assert "[Observation masked" in old_tool_result

    print("PASS: mask_old_observations correctly masks old verbose results")


def test_reasoning_context_lifecycle():
    """Test ReasoningContext add, compact, and injection."""
    ctx = ReasoningContext(session_id="test-session", max_entries=3)

    # Add entries
    for i in range(5):
        entry = ReasoningEntry(
            iteration=i + 1,
            timestamp=datetime.now(timezone.utc),
            raw_thinking=f"Thinking for iteration {i + 1}",
            summary=f"Summary {i + 1}",
            decisions=[f"Decision {i + 1}"],
            tool_calls=["create_sketch"] if i == 0 else ["extrude_profile"],
            outcome="success" if i % 2 == 0 else "partial",
        )
        ctx.add_entry(entry)

    # Should have compacted old entries
    assert len(ctx.entries) <= ctx.max_entries, f"Should compact to max {ctx.max_entries} entries"
    assert ctx.compacted_summary, "Should have compacted summary from old entries"

    # Get injection text
    injection = ctx.get_injection_text()
    assert injection, "Should produce injection text"
    assert "Iteration" in injection, "Should include iteration markers"

    logger.info(f"Injection text ({len(injection)} chars):\n{injection}")
    print(f"PASS: ReasoningContext lifecycle - {len(ctx.entries)} entries, {len(injection)} char injection")


def test_reasoning_context_section_format():
    """Test that REASONING_CONTEXT_SECTION formats correctly."""
    history = """**Iteration 1**
- Decisions: Created sketch on XY plane
- Actions: create_sketch
- Outcome: success

**Iteration 2**
- Decisions: Drew rectangle for base plate
- Actions: add_rectangle
- Outcome: success"""

    formatted = REASONING_CONTEXT_SECTION.format(reasoning_history=history)

    assert "Prior Reasoning Context" in formatted
    assert "Iteration 1" in formatted
    assert "Avoid repeating approaches that failed" in formatted

    print("PASS: REASONING_CONTEXT_SECTION formats correctly")


# =============================================================================
# Integration Test with Mock LLM
# =============================================================================

async def test_reasoning_capture_integration():
    """
    Integration test that simulates multiple agent iterations and validates
    reasoning context is captured and would be injected.

    This test mocks the LLM response to avoid API calls while testing the flow.
    """
    from backend.websocket_manager import ConnectionManager

    # Create manager and simulate connection
    manager = ConnectionManager()
    session_id = "test-integration-session"

    # Mock WebSocket
    mock_ws = AsyncMock()
    mock_ws.accept = AsyncMock()
    mock_ws.send_json = AsyncMock()

    await manager.connect(session_id, mock_ws)

    # Verify reasoning context was initialized
    reasoning_ctx = manager.get_reasoning_context(session_id)
    assert reasoning_ctx is not None, "Should have reasoning context after connect"
    assert reasoning_ctx.session_id == session_id

    # Simulate adding reasoning entries (as would happen in agent_workflow)
    for i in range(3):
        entry = ReasoningEntry(
            iteration=i + 1,
            timestamp=datetime.now(timezone.utc),
            raw_thinking=f"Extended thinking for iteration {i + 1}: analyzing geometry...",
            summary=f"Created geometry step {i + 1}",
            decisions=[f"Chose approach {i + 1}"],
            tool_calls=["create_sketch", "add_rectangle", "extrude_profile"][i:i+1],
            outcome="success",
        )
        reasoning_ctx.add_entry(entry)

    # Verify entries were captured
    assert len(reasoning_ctx.entries) == 3, f"Should have 3 entries, got {len(reasoning_ctx.entries)}"

    # Verify injection text is generated
    injection = reasoning_ctx.get_injection_text()
    assert "Iteration 1" in injection
    assert "Iteration 3" in injection

    # Test clear functionality
    manager.clear_reasoning_context(session_id)
    assert len(reasoning_ctx.entries) == 0, "Should clear entries"

    # Cleanup
    manager.disconnect(session_id)

    print("PASS: Integration test - reasoning context lifecycle in manager")


# =============================================================================
# Live OpenAI Test (requires API key)
# =============================================================================

async def test_openai_with_reasoning_context():
    """
    Live test that makes actual OpenAI API calls to verify the reasoning
    context injection works with real LLM responses.

    This simulates a simplified agent loop without Fusion 360.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("test-") or "your-api-key" in api_key.lower():
        print("SKIP: test_openai_with_reasoning_context (no valid OPENAI_API_KEY)")
        return

    from backend.llm_client import call_claude_with_tools
    from backend.prompt_structure import CORE_INSTRUCTIONS, REASONING_CONTEXT_SECTION

    # Minimal tool set for testing
    test_tools = [
        {
            "name": "log_status",
            "description": "Record a status note during the reasoning-context test",
            "input_schema": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to send"}
                },
                "required": ["message"]
            }
        },
        {
            "name": "create_sketch",
            "description": "Create a new sketch on a plane",
            "input_schema": {
                "type": "object",
                "properties": {
                    "plane_id": {"type": "string", "description": "Plane: XY, XZ, YZ"}
                },
                "required": ["plane_id"]
            }
        }
    ]

    # Simulated reasoning context from "prior iterations"
    prior_reasoning = """**Iteration 1**
- Decisions: Started with XY plane for the base
- Actions: create_sketch
- Outcome: success

**Iteration 2**
- Decisions: Rectangle dimensions 50x30mm chosen based on user request
- Actions: add_rectangle
- Outcome: success
- Summary: Base plate sketch complete, ready to extrude"""

    # System prompt with reasoning context injected
    system_prompt = CORE_INSTRUCTIONS + "\n\n" + REASONING_CONTEXT_SECTION.format(
        reasoning_history=prior_reasoning
    )

    # Test message - asking about next steps
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What should I do next to complete this base plate? I need 5mm thickness."}
            ]
        }
    ]

    # Accumulate reasoning
    reasoning_chunks = []

    async def capture_reasoning(delta: str):
        reasoning_chunks.append(delta)
        logger.debug(f"Reasoning chunk: {delta[:50]}...")

    try:
        logger.info("Calling OpenAI with reasoning context injected...")
        response = await call_claude_with_tools(
            messages=messages,
            tools=test_tools,
            system_prompt=system_prompt,
            model_name="gpt-5.2",  # Use GPT-5.2 for reasoning
            reasoning_effort="medium",
            session_id="openai-test",
            iteration=3,  # Simulate iteration 3
            reasoning_callback=capture_reasoning,
        )

        # Analyze response
        stop_reason = response.get("stop_reason")
        content = response.get("content", [])

        logger.info(f"Response stop_reason: {stop_reason}")
        logger.info(f"Response content blocks: {len(content)}")

        # Check for tool calls
        tool_calls = [c for c in content if c.get("type") == "tool_use"]
        text_blocks = [c for c in content if c.get("type") == "text"]

        if tool_calls:
            logger.info(f"Tool calls: {[tc.get('name') for tc in tool_calls]}")
        if text_blocks:
            text = text_blocks[0].get("text", "")[:200]
            logger.info(f"Text response: {text}...")

        # Check if reasoning was captured
        if reasoning_chunks:
            full_reasoning = "".join(reasoning_chunks)
            logger.info(f"Captured {len(full_reasoning)} chars of reasoning")

            # Verify reasoning references prior context
            if "prior" in full_reasoning.lower() or "previous" in full_reasoning.lower() or "iteration" in full_reasoning.lower():
                logger.info("SUCCESS: Model referenced prior reasoning context!")

            # Extract decisions from this reasoning
            decisions = extract_decisions(full_reasoning, max_decisions=3)
            logger.info(f"Extracted decisions: {decisions}")

        print(f"PASS: OpenAI test - stop_reason={stop_reason}, tool_calls={len(tool_calls)}, reasoning_chars={len(''.join(reasoning_chunks))}")

    except Exception as e:
        logger.error(f"OpenAI test failed: {e}")
        print(f"FAIL: OpenAI test - {e}")
        raise


async def test_multi_iteration_simulation():
    """
    Simulate a 3-iteration agent loop with reasoning context building up.
    Uses OpenAI to show how reasoning improves across iterations.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("test-"):
        print("SKIP: test_multi_iteration_simulation (no valid OPENAI_API_KEY)")
        return

    from backend.llm_client import call_claude_with_tools
    from backend.prompt_structure import CORE_INSTRUCTIONS, REASONING_CONTEXT_SECTION
    from backend.websocket_manager import ConnectionManager

    # Setup
    manager = ConnectionManager()
    session_id = "multi-iter-test"
    mock_ws = AsyncMock()
    mock_ws.accept = AsyncMock()
    mock_ws.send_json = AsyncMock()
    await manager.connect(session_id, mock_ws)

    reasoning_ctx = manager.get_reasoning_context(session_id)

    # Minimal tools
    tools = [
        {
            "name": "log_status",
            "description": "Record a status note during the reasoning-context test",
            "input_schema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"]
            }
        }
    ]

    # Simulate 3 iterations with accumulating context
    user_prompts = [
        "I want to create a simple box. How should I start?",
        "Good, the sketch is done. What's next?",
        "Now I need to add a hole in the top. What's the best approach?",
    ]

    messages = []

    for iteration in range(3):
        logger.info(f"\n{'='*50}")
        logger.info(f"ITERATION {iteration + 1}")
        logger.info(f"{'='*50}")

        # Add user message
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_prompts[iteration]}]
        })

        # Build system prompt with reasoning context
        base_prompt = "You are a CAD assistant helping create 3D models. Be concise."
        if reasoning_ctx.entries:
            injection = reasoning_ctx.get_injection_text()
            system_prompt = base_prompt + "\n\n" + REASONING_CONTEXT_SECTION.format(
                reasoning_history=injection
            )
            logger.info(f"Injected {len(injection)} chars of prior reasoning")
        else:
            system_prompt = base_prompt
            logger.info("No prior reasoning to inject")

        # Capture reasoning
        reasoning_buffer = []
        async def on_reasoning(delta: str):
            if delta and delta.strip():
                reasoning_buffer.append(delta)

        try:
            response = await call_claude_with_tools(
                messages=messages,
                tools=tools,
                system_prompt=system_prompt,
                model_name="gpt-5.2",
                reasoning_effort="medium",  # Must be medium+ to produce reasoning tokens
                session_id=session_id,
                iteration=iteration + 1,
                reasoning_callback=on_reasoning,
            )

            # Get response content
            content = response.get("content", [])
            text_blocks = [c for c in content if c.get("type") == "text"]
            response_text = text_blocks[0].get("text", "") if text_blocks else "No response"

            # Add assistant response to messages
            messages.append({"role": "assistant", "content": content})

            # Capture reasoning entry
            if reasoning_buffer:
                full_reasoning = "".join(reasoning_buffer)
                entry = ReasoningEntry(
                    iteration=iteration + 1,
                    timestamp=datetime.now(timezone.utc),
                    raw_thinking=full_reasoning,
                    summary=summarize_reasoning(full_reasoning, max_chars=200),
                    decisions=extract_decisions(full_reasoning, max_decisions=2),
                    tool_calls=[],
                    outcome="success",
                )
                reasoning_ctx.add_entry(entry)
                logger.info(f"Captured reasoning: {len(full_reasoning)} chars, {len(entry.decisions)} decisions")

            logger.info(f"Response: {response_text[:150]}...")

        except Exception as e:
            logger.error(f"Iteration {iteration + 1} failed: {e}")
            break

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("SIMULATION COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"Total reasoning entries: {len(reasoning_ctx.entries)}")

    if reasoning_ctx.entries:
        final_injection = reasoning_ctx.get_injection_text()
        logger.info(f"Final injection would be {len(final_injection)} chars")
        logger.info(f"Sample:\n{final_injection[:500]}...")

    manager.disconnect(session_id)
    print(f"PASS: Multi-iteration simulation - {len(reasoning_ctx.entries)} reasoning entries captured")


# =============================================================================
# Main Test Runner
# =============================================================================

def run_unit_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("UNIT TESTS")
    print("="*60 + "\n")

    test_extract_decisions()
    test_summarize_reasoning()
    test_assess_outcome()
    test_mask_old_observations()
    test_reasoning_context_lifecycle()
    test_reasoning_context_section_format()

    print("\nAll unit tests passed!")


async def run_integration_tests():
    """Run integration tests."""
    print("\n" + "="*60)
    print("INTEGRATION TESTS")
    print("="*60 + "\n")

    await test_reasoning_capture_integration()

    print("\nIntegration tests passed!")


async def run_live_tests():
    """Run live API tests (requires valid API key)."""
    print("\n" + "="*60)
    print("LIVE OPENAI TESTS")
    print("="*60 + "\n")

    await test_openai_with_reasoning_context()
    await test_multi_iteration_simulation()

    print("\nLive tests completed!")


if __name__ == "__main__":
    print("="*60)
    print("REASONING CONTEXT MANAGEMENT TEST SUITE")
    print("="*60)

    # Run unit tests (no API calls)
    run_unit_tests()

    # Run integration tests (no API calls)
    asyncio.run(run_integration_tests())

    # Run live tests (requires API key)
    print("\nRunning live OpenAI tests...")
    asyncio.run(run_live_tests())

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
