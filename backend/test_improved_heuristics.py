"""
Test improved heuristics against real reasoning samples from logs.

This test validates that the updated keyword categories and scoring
algorithm better capture decisions from actual Claude and GPT-5.x reasoning.
"""

import sys
sys.path.insert(0, '/Users/erikfornlund/Desktop/CADAgent-Projects/cadagent-reasoning-efficently/backend')

from backend.reasoning_context import (
    extract_decisions,
    summarize_reasoning,
    extract_section_headers,
    score_sentence,
    DECISION_KEYWORDS,
    PROCEDURAL_KEYWORDS,
    CONSTRAINT_KEYWORDS,
    DELIBERATION_KEYWORDS,
    RESOLUTION_KEYWORDS,
    CAD_DOMAIN_KEYWORDS,
    ALL_KEYWORDS,
)

# Real samples extracted from cadagent_reasoning.log
SAMPLE_1_GPT_CUBE = """**Creating a simple cube**

The user wants me to create a simple cube without any questions, so I'll proceed with defaults. The rules indicate using centimeters for all coordinates, even though some defaults mention millimeters. Since a reasonable default size could be a 20 mm cube, I'll convert that to 2.0 cm. To keep things straightforward, I'll create a sketch on the XY plane. Instead of a rectangle, a square will be drawn as an easier option, centered at the origin. I'll make adjustments accordingly to extrude it properly. Let's get started!"""

SAMPLE_2_GPT_HOLES = """**Planning hole creation**

I need to ensure that the `create_simple_hole` operation references the correct face from the Design Entities context, which I'll get after the extrusion creates the body and faces. Following the developer instructions, I should execute each step and check for errors. The first step is to create a sketch on the XZ plane, named "L-bracket profile sketch," using cm units for coordinates. In this context, I'll define the U axis as corresponding to X and V to Z, so I can effectively draw the L-profile with the right dimensions."""

SAMPLE_3_GPT_DELIBERATION = """**Clarifying the sketches' location**

I think it's important to ask where the sketches are. But I also need to consider that asking questions can slow things down, which goes against being proactive. Since I can't find the sketches, I'll respond to the user asking for clarification on which face the sketches are on or if they can provide the sketch entities. However, the user requested no questions, so maybe I should assume the holes are on the top face, but it's risky since positions are unknown.

I think a better approach is to explain that there are no hole sketches available. Since the user preferred not to have questions asked, I should propose a default action: adding a center hole on the top face with a standard diameter."""

SAMPLE_4_GPT_CONSTRAINT = """**Following developer instructions**

I need to keep following the developer's guidelines, working one operation at a time. The assistant has already created a sketch on the XY plane and extruded the profile to a distance of 2.5 cm. The user's measurements were 90mm x 50mm x 25mm with a wall thickness of 2.5mm, so I need to convert those units to cm: 90mm becomes 9.0 cm, 50mm turns into 5.0 cm, and 25mm remains 2.5 cm. The wall thickness converts to 0.25 cm. Everything is aligning well!"""

SAMPLE_5_GPT_GEOMETRY = """**Setting Up the D Plate Profile**

I need to create a sketch on the XY plane and add a rectangle with corners at (0, -1.5) to (6, 1.5) cm, as well as a circle centered at (6, 0) with a radius of 1.5 cm. I must batch these multiple sketch commands in a single response, as the toolkit allows for multiple tool calls in one go. However, I need to call create_sketch first, ensuring it completes before adding the rectangle and circle together in a subsequent call."""


def test_header_extraction():
    """Test that GPT-style headers are properly extracted."""
    print("\n=== Testing Header Extraction ===")

    headers = extract_section_headers(SAMPLE_1_GPT_CUBE)
    print(f"Sample 1 headers: {headers}")
    assert "Creating a simple cube" in headers

    headers = extract_section_headers(SAMPLE_2_GPT_HOLES)
    print(f"Sample 2 headers: {headers}")
    assert "Planning hole creation" in headers

    headers = extract_section_headers(SAMPLE_3_GPT_DELIBERATION)
    print(f"Sample 3 headers: {headers}")
    assert "Clarifying the sketches' location" in headers

    print("PASSED: Header extraction works correctly")


def test_decision_extraction_quality():
    """Test that decisions are properly extracted with improved scoring."""
    print("\n=== Testing Decision Extraction Quality ===")

    # Sample 1: Should extract the decision to use defaults and XY plane
    decisions = extract_decisions(SAMPLE_1_GPT_CUBE)
    print(f"\nSample 1 decisions ({len(decisions)}):")
    for d in decisions:
        print(f"  - {d[:80]}...")

    # Should have the header and key decisions
    assert any("Creating a simple cube" in d for d in decisions), "Should capture header"
    assert any("proceed with defaults" in d.lower() or "i'll" in d.lower() for d in decisions), "Should capture intent"

    # Sample 2: Should extract procedural steps
    decisions = extract_decisions(SAMPLE_2_GPT_HOLES)
    print(f"\nSample 2 decisions ({len(decisions)}):")
    for d in decisions:
        print(f"  - {d[:80]}...")

    assert any("Planning hole creation" in d for d in decisions), "Should capture header"

    # Sample 3: Should capture the deliberation and resolution
    decisions = extract_decisions(SAMPLE_3_GPT_DELIBERATION)
    print(f"\nSample 3 decisions ({len(decisions)}):")
    for d in decisions:
        print(f"  - {d[:80]}...")

    # Should capture the resolution
    assert any("better approach" in d.lower() or "propose" in d.lower() for d in decisions), "Should capture resolution"

    print("\nPASSED: Decision extraction captures key statements")


def test_sentence_scoring():
    """Test that scoring prioritizes the right patterns."""
    print("\n=== Testing Sentence Scoring ===")

    # High-score sentences (decision + procedural + domain)
    high_score_sentence = "I'll create a sketch on the XY plane and then extrude it to 2.5 cm."
    score = score_sentence(high_score_sentence)
    print(f"High-score sentence: {score:.2f}")
    assert score > 5.0, "Decision + procedural + domain should score high"

    # Medium-score sentence (just decision)
    medium_sentence = "I need to figure out what dimensions to use."
    score = score_sentence(medium_sentence)
    print(f"Medium-score sentence: {score:.2f}")
    assert 2.0 < score < 8.0, "Simple decision should score medium"

    # Low-score sentence (generic)
    low_sentence = "This is a test."
    score = score_sentence(low_sentence)
    print(f"Low-score sentence: {score:.2f}")
    assert score < 2.0, "Generic sentence should score low"

    # Constraint reference (should boost)
    constraint_sentence = "According to the developer instructions, I should use centimeters."
    score = score_sentence(constraint_sentence)
    print(f"Constraint sentence: {score:.2f}")
    assert score > 3.0, "Constraint reference should boost score"

    print("PASSED: Sentence scoring differentiates patterns correctly")


def test_summarize_with_headers():
    """Test that summaries include headers and key decisions."""
    print("\n=== Testing Summary Generation ===")

    summary = summarize_reasoning(SAMPLE_4_GPT_CONSTRAINT, max_chars=300)
    print(f"\nSummary (Sample 4):\n{summary}")

    # Should include the header context
    assert "[" in summary or "developer" in summary.lower(), "Summary should include context"

    # Should include key information
    assert "cm" in summary.lower() or "convert" in summary.lower(), "Should preserve unit info"

    summary = summarize_reasoning(SAMPLE_5_GPT_GEOMETRY, max_chars=400)
    print(f"\nSummary (Sample 5):\n{summary}")

    # Should include geometric details
    assert "sketch" in summary.lower() or "rectangle" in summary.lower(), "Should preserve geometry info"

    print("PASSED: Summaries capture key information")


def test_keyword_coverage():
    """Verify keyword categories cover observed patterns."""
    print("\n=== Testing Keyword Coverage ===")

    # Check that real patterns are covered
    observed_patterns = [
        "i need to",
        "i'll",
        "first",
        "according to",
        "however",
        "centered",
        "sketch",
        "so i'll",
        "i'll proceed",
    ]

    for pattern in observed_patterns:
        found = any(pattern in kw.lower() for kw in ALL_KEYWORDS)
        print(f"  '{pattern}': {'Found' if found else 'MISSING'}")
        assert found, f"Pattern '{pattern}' should be in keywords"

    print("PASSED: Keyword categories cover observed patterns")


def run_comparison_test():
    """Compare old vs new extraction on real samples."""
    print("\n=== Comparison: Before vs After ===")

    # Test on the most complex sample
    sample = SAMPLE_3_GPT_DELIBERATION

    decisions = extract_decisions(sample, max_decisions=5)
    print(f"\nExtracted {len(decisions)} decisions from deliberation sample:")
    for i, d in enumerate(decisions, 1):
        print(f"  {i}. {d[:100]}...")

    summary = summarize_reasoning(sample, max_chars=350)
    print(f"\nSummary:\n{summary}")

    # Verify it captures the key points
    assert len(decisions) >= 2, "Should extract multiple decisions"
    assert len(summary) > 100, "Summary should be substantial"

    print("\nPASSED: Improved extraction captures complex reasoning")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Improved Reasoning Heuristics")
    print("=" * 60)

    test_header_extraction()
    test_sentence_scoring()
    test_decision_extraction_quality()
    test_summarize_with_headers()
    test_keyword_coverage()
    run_comparison_test()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
