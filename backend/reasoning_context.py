"""
Reasoning Context Management for CADAgent

Maintains continuous chain-of-thought context across agentic loop iterations.
Instead of losing reasoning between tool calls, this module captures, summarizes,
and re-injects reasoning history to improve coherence on multi-step tasks.

Based on research from:
- ReAct (interleaved reasoning + action)
- JetBrains observation masking study
- Anthropic context engineering guidelines
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# =============================================================================
# KEYWORD CATEGORIES FOR HEURISTIC EXTRACTION
# Derived from analysis of 81K+ lines of actual Claude and GPT-5.x reasoning
# =============================================================================

# Core decision/intent markers (highest signal)
DECISION_KEYWORDS = [
    # Direct intent
    "i will", "i'll", "i need to", "i should", "i must", "i can",
    "i'm going to", "let me", "my plan is",
    # Decision confirmation
    "decided", "choosing", "selected", "going with", "opting for",
    # Causal reasoning
    "because", "therefore", "this means", "so i'll", "which means",
    "since", "given that", "as a result",
    # Planning markers
    "the approach", "my plan", "next step", "the strategy",
    "this requires", "to achieve", "in order to",
    # Outcome indicators
    "failed", "didn't work", "error", "issue", "problem",
    "success", "worked", "completed", "done",
]

# Procedural/sequencing markers (important for multi-step reasoning)
PROCEDURAL_KEYWORDS = [
    "first", "then", "next", "after", "finally", "lastly",
    "starting with", "following", "before", "once",
    "step 1", "step 2", "the first step", "the next step",
]

# Constraint/rule reference markers (common in GPT-5.x reasoning)
CONSTRAINT_KEYWORDS = [
    "according to", "the rules", "the instructions", "developer instructions",
    "the specification", "as specified", "per the", "following the",
    "the guidelines", "requirements state",
]

# Hesitation/consideration markers (indicates deliberation)
DELIBERATION_KEYWORDS = [
    "but", "however", "although", "might be", "seems like",
    "i wonder", "should i", "i'm not sure", "uncertain",
    "on the other hand", "alternatively", "instead",
    "tricky", "confusing", "ambiguous",
]

# Resolution markers (confirms a decision was made)
RESOLUTION_KEYWORDS = [
    "so i'll", "i'll go ahead", "i'll proceed", "this should",
    "that's the direction", "makes sense", "aligns with",
    "i've decided", "i've determined", "the solution is",
]

# CAD/spatial domain keywords (specific to this application)
CAD_DOMAIN_KEYWORDS = [
    # Geometry
    "centered", "radius", "diameter", "corner", "edge", "vertex",
    "face", "profile", "sketch", "extrude", "hole", "fillet",
    # Planes/coordinates
    "xy plane", "xz plane", "yz plane", "coordinates", "origin",
    "axis", "normal", "dimension", "position",
    # Operations
    "create", "draw", "add", "remove", "cut", "shell", "pattern",
    # Units/conversion
    "convert", "converting", "mm to cm", "cm to mm", "millimeters", "centimeters",
]

# All keywords combined for backward compatibility
ALL_KEYWORDS = (
    DECISION_KEYWORDS + PROCEDURAL_KEYWORDS + CONSTRAINT_KEYWORDS +
    DELIBERATION_KEYWORDS + RESOLUTION_KEYWORDS + CAD_DOMAIN_KEYWORDS
)


@dataclass
class ReasoningEntry:
    """Single reasoning capture from one iteration."""

    iteration: int
    timestamp: datetime
    raw_thinking: str
    summary: str
    decisions: List[str]
    tool_calls: List[str]
    outcome: Optional[str] = None  # "success" | "partial" | "failed" | None

    def to_injection_text(self) -> str:
        """Format this entry for context injection."""
        lines = [f"**Iteration {self.iteration}**"]

        if self.decisions:
            lines.append(f"- Decisions: {'; '.join(self.decisions[:3])}")

        if self.tool_calls:
            lines.append(f"- Actions: {', '.join(self.tool_calls)}")

        if self.outcome:
            lines.append(f"- Outcome: {self.outcome}")

        if self.summary and self.summary != "No reasoning captured":
            # Only include summary if it adds value beyond decisions
            if len(self.summary) > 50:
                lines.append(f"- Summary: {self.summary}")

        return "\n".join(lines)


@dataclass
class ReasoningContext:
    """
    Session-scoped reasoning history with automatic compaction.

    Maintains a rolling window of reasoning entries, summarizing older
    entries to prevent context rot while preserving key decisions.
    """

    session_id: str
    entries: List[ReasoningEntry] = field(default_factory=list)
    max_entries: int = 10
    max_injection_tokens: int = 2000
    compacted_summary: str = ""

    def add_entry(self, entry: ReasoningEntry) -> None:
        """Add a new reasoning entry, compacting if necessary."""
        self.entries.append(entry)
        logger.debug(
            f"[{self.session_id}] Added reasoning entry for iteration {entry.iteration}, "
            f"total entries: {len(self.entries)}"
        )

        if len(self.entries) > self.max_entries:
            self._compact_oldest()

    def _compact_oldest(self) -> None:
        """Merge oldest entries into compacted summary."""
        if len(self.entries) <= self.max_entries // 2:
            return

        # Take oldest half and summarize
        to_compact = self.entries[: len(self.entries) // 2]
        self.entries = self.entries[len(self.entries) // 2 :]

        # Build compacted summary from old entries
        all_decisions = []
        all_outcomes = []
        for entry in to_compact:
            all_decisions.extend(entry.decisions[:2])
            if entry.outcome:
                all_outcomes.append(f"iter{entry.iteration}:{entry.outcome}")

        compact_text = f"Earlier iterations ({to_compact[0].iteration}-{to_compact[-1].iteration}): "
        if all_decisions:
            compact_text += f"Decisions: {'; '.join(all_decisions[:5])}. "
        if all_outcomes:
            compact_text += f"Outcomes: {', '.join(all_outcomes)}."

        if self.compacted_summary:
            self.compacted_summary = f"{self.compacted_summary} | {compact_text}"
        else:
            self.compacted_summary = compact_text

        # Truncate compacted summary if too long
        if len(self.compacted_summary) > 1000:
            self.compacted_summary = self.compacted_summary[-1000:]

        logger.debug(
            f"[{self.session_id}] Compacted {len(to_compact)} entries, "
            f"remaining: {len(self.entries)}"
        )

    def get_injection_text(self) -> str:
        """
        Generate text block for system prompt injection.

        Returns formatted reasoning history suitable for inclusion
        in the system prompt before the next LLM call.
        """
        if not self.entries and not self.compacted_summary:
            return ""

        parts = []

        # Include compacted summary of older iterations
        if self.compacted_summary:
            parts.append(f"*{self.compacted_summary}*")
            parts.append("")

        # Include recent entries in detail
        for entry in self.entries:
            parts.append(entry.to_injection_text())
            parts.append("")

        full_text = "\n".join(parts).strip()

        # Rough token limit (4 chars ~= 1 token)
        max_chars = self.max_injection_tokens * 4
        if len(full_text) > max_chars:
            # Truncate from the beginning (keep most recent)
            full_text = "..." + full_text[-(max_chars - 3) :]

        return full_text

    def clear(self) -> None:
        """Clear all reasoning context (e.g., on task reset)."""
        self.entries.clear()
        self.compacted_summary = ""
        logger.debug(f"[{self.session_id}] Reasoning context cleared")

    def get_last_outcome(self) -> Optional[str]:
        """Get the outcome of the most recent iteration."""
        if self.entries:
            return self.entries[-1].outcome
        return None


def extract_section_headers(reasoning_text: str) -> List[str]:
    """
    Extract GPT-style section headers (**Header Text**) from reasoning.

    GPT-5.x models commonly use markdown bold headers to mark major
    reasoning sections. These are high-signal decision/planning markers.

    Args:
        reasoning_text: Raw reasoning text

    Returns:
        List of header texts (without the ** markers)
    """
    # Match **Header Text** pattern (GPT-style section headers)
    header_pattern = r'\*\*([^*]+)\*\*'
    matches = re.findall(header_pattern, reasoning_text)
    return [h.strip() for h in matches if len(h.strip()) > 3]


def score_sentence(sentence: str, position_ratio: float = 0.5) -> float:
    """
    Score a sentence by decision-relevance using weighted keyword categories.

    Weights derived from empirical analysis of which patterns best indicate
    actionable decisions vs. general deliberation.

    Args:
        sentence: The sentence to score
        position_ratio: Position in text (0=start, 1=end), used for tie-breaking

    Returns:
        Relevance score (higher = more likely a key decision)
    """
    sentence_lower = sentence.lower()
    score = 0.0

    # Category weights (derived from pattern analysis)
    weights = {
        'decision': 3.0,      # Core decision/intent - highest signal
        'resolution': 2.5,    # Confirms decision was made
        'procedural': 2.0,    # Multi-step planning
        'constraint': 1.5,    # Rule references
        'cad_domain': 1.5,    # Domain-specific (relevant context)
        'deliberation': 1.0,  # Indicates thinking but not decision
    }

    # Check each category
    for kw in DECISION_KEYWORDS:
        if kw in sentence_lower:
            score += weights['decision']

    for kw in RESOLUTION_KEYWORDS:
        if kw in sentence_lower:
            score += weights['resolution']

    for kw in PROCEDURAL_KEYWORDS:
        if kw in sentence_lower:
            score += weights['procedural']

    for kw in CONSTRAINT_KEYWORDS:
        if kw in sentence_lower:
            score += weights['constraint']

    for kw in CAD_DOMAIN_KEYWORDS:
        if kw in sentence_lower:
            score += weights['cad_domain']

    for kw in DELIBERATION_KEYWORDS:
        if kw in sentence_lower:
            score += weights['deliberation']

    # Boost for sentences with multiple keywords (compound decisions)
    keyword_count = sum(1 for kw in ALL_KEYWORDS if kw in sentence_lower)
    if keyword_count > 2:
        score *= 1.2

    # Penalty for very short sentences (less informative)
    if len(sentence) < 30:
        score *= 0.5
    elif len(sentence) < 50:
        score *= 0.8

    # Slight boost for later sentences (conclusions tend to come later)
    score += position_ratio * 0.5

    return score


def extract_decisions(reasoning_text: str, max_decisions: int = 5) -> List[str]:
    """
    Extract key decision statements from reasoning text.

    Uses weighted keyword heuristics to identify sentences containing decisions,
    plans, or conclusions without requiring an additional LLM call.

    Improved algorithm based on analysis of 81K+ lines of actual LLM reasoning:
    1. Detects GPT-style **Headers** as high-priority section markers
    2. Scores sentences using weighted keyword categories
    3. Prioritizes resolution/decision markers over general deliberation

    Args:
        reasoning_text: Raw extended thinking/reasoning text
        max_decisions: Maximum number of decisions to extract

    Returns:
        List of decision statements (cleaned and truncated)
    """
    if not reasoning_text:
        return []

    decisions = []

    # First, extract section headers (high-signal in GPT-5.x reasoning)
    headers = extract_section_headers(reasoning_text)
    for header in headers[:2]:  # Take up to 2 headers
        if 10 < len(header) < 100:
            decisions.append(f"[{header}]")

    # Split into sentences (handle multiple punctuation patterns)
    sentences = re.split(r'(?<=[.!?])\s+', reasoning_text)
    total_sentences = len(sentences)

    # Score and collect candidate sentences
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        # Skip sentences that are just headers (already extracted)
        if sentence.strip().startswith('**') and sentence.strip().endswith('**'):
            continue

        position_ratio = i / max(total_sentences, 1)
        score = score_sentence(sentence, position_ratio)

        if score > 1.0:  # Minimum threshold
            cleaned = sentence.strip()
            # Remove leading ** from header continuations
            cleaned = re.sub(r'^\*\*[^*]+\*\*\s*', '', cleaned)

            if 20 < len(cleaned) < 300:
                scored_sentences.append((score, cleaned))

    # Sort by score descending and take top candidates
    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    remaining_slots = max_decisions - len(decisions)
    for score, sentence in scored_sentences[:remaining_slots]:
        # Truncate if too long
        if len(sentence) > 150:
            sentence = sentence[:147] + "..."
        decisions.append(sentence)

    return decisions


def summarize_reasoning(
    reasoning_text: str,
    max_chars: int = 500,
    include_failures: bool = True
) -> str:
    """
    Create a heuristic summary of reasoning without LLM call.

    Extracts the most decision-relevant sentences and combines them
    into a coherent summary paragraph.

    Improved algorithm based on analysis of actual LLM reasoning patterns:
    1. Uses weighted keyword scoring from score_sentence()
    2. Includes section headers as context
    3. Prioritizes failure/error mentions for learning
    4. Preserves procedural flow when possible

    Args:
        reasoning_text: Raw extended thinking/reasoning text
        max_chars: Maximum characters in summary
        include_failures: Whether to prioritize failure/error mentions

    Returns:
        Summarized reasoning text
    """
    if not reasoning_text:
        return "No reasoning captured"

    summary_parts = []
    current_length = 0

    # Extract section headers first (provide context)
    headers = extract_section_headers(reasoning_text)
    if headers:
        header_text = f"[{headers[0]}]"
        if len(header_text) < max_chars // 4:  # Don't let headers dominate
            summary_parts.append(header_text)
            current_length = len(header_text) + 1

    sentences = re.split(r'(?<=[.!?])\s+', reasoning_text)
    total_sentences = len(sentences)

    # Score sentences using the improved algorithm
    scored = []
    for i, sentence in enumerate(sentences):
        # Skip pure headers
        if sentence.strip().startswith('**') and sentence.strip().endswith('**'):
            continue

        sentence_lower = sentence.lower()
        position_ratio = i / max(total_sentences, 1)

        # Base score from weighted categories
        score = score_sentence(sentence, position_ratio)

        # Additional boost for failure/error mentions (important for learning)
        if include_failures:
            error_terms = ["fail", "error", "issue", "problem", "wrong", "can't", "doesn't"]
            if any(w in sentence_lower for w in error_terms):
                score += 4.0

        # Additional boost for explicit success mentions
        success_terms = ["success", "worked", "complete", "done", "created"]
        if any(w in sentence_lower for w in success_terms):
            score += 3.0

        if score > 1.0:
            cleaned = sentence.strip()
            cleaned = re.sub(r'^\*\*[^*]+\*\*\s*', '', cleaned)  # Remove header prefix
            if len(cleaned) > 20:
                scored.append((score, cleaned, i))  # Keep position for ordering

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top sentences but maintain some temporal order
    selected = scored[:8]  # Candidate pool
    # Sort selected by original position to maintain flow
    selected.sort(key=lambda x: x[2])

    for score, sentence, _ in selected:
        if current_length + len(sentence) + 2 > max_chars:
            break
        summary_parts.append(sentence)
        current_length += len(sentence) + 2

    if not summary_parts:
        # Fallback: take first substantial sentence
        for sentence in sentences:
            if len(sentence) > 30:
                return sentence[:max_chars]
        return reasoning_text[:max_chars]

    return " ".join(summary_parts)


def assess_outcome(tool_results: List[Dict[str, Any]]) -> Optional[str]:
    """
    Assess the outcome of an iteration based on tool results.

    Args:
        tool_results: List of tool result dictionaries

    Returns:
        "success" | "partial" | "failed" | None
    """
    if not tool_results:
        return None

    errors = 0
    successes = 0

    for result in tool_results:
        content = str(result.get("content", "")).lower()
        is_error = result.get("is_error", False)

        if is_error or "error" in content or "failed" in content:
            errors += 1
        elif "success" in content or "created" in content or "completed" in content:
            successes += 1
        else:
            # Neutral result counts as partial success
            successes += 0.5

    if errors == 0 and successes > 0:
        return "success"
    elif errors > 0 and successes > 0:
        return "partial"
    elif errors > 0:
        return "failed"

    return None


def mask_old_observations(
    messages: List[Dict[str, Any]],
    keep_recent: int = 3,
    max_result_chars: int = 500
) -> List[Dict[str, Any]]:
    """
    Mask verbose tool results in older messages while preserving reasoning.

    Based on JetBrains research showing observation masking outperforms
    full summarization for agent context management.

    Args:
        messages: Full conversation history
        keep_recent: Number of recent messages to leave unmasked
        max_result_chars: Max chars before masking a tool result

    Returns:
        Messages with old verbose tool results masked
    """
    if len(messages) <= keep_recent:
        return messages

    # Work on a copy to avoid mutating original
    import copy
    masked = copy.deepcopy(messages)

    for msg in masked[:-keep_recent]:
        if msg.get("role") != "user":
            continue

        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") != "tool_result":
                continue

            result_content = item.get("content", "")
            if not isinstance(result_content, str):
                result_content = str(result_content)

            if len(result_content) > max_result_chars:
                # Preserve first/last bits for context
                preview = result_content[:100]
                is_error = item.get("is_error", False)
                status = "error" if is_error else "success"

                item["content"] = (
                    f"[Observation masked - {len(result_content)} chars, {status}]\n"
                    f"Preview: {preview}..."
                )

    return masked
