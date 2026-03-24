"""
layer1_regex.py

Layer 1 of the RAG injection scanning pipeline.
Runs fast, free regex pattern matching on every chunk.
Detects known, explicit injection patterns across 7 attack categories.
Output: flagged (bool) + matched patterns per chunk.

This layer catches known explicit attacks. Subtle, paraphrased,
and obfuscated attacks are handled by Layer 2 and Layer 3.
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Each category has a name and a list of regex patterns.
# Patterns are compiled once at module load — not on every scan call.
# re.IGNORECASE ensures "IGNORE PREVIOUS" and "ignore previous" both match.

INJECTION_PATTERNS: list[dict[str, Any]] = [
    {
        "category": "instruction_override",
        "patterns": [
            re.compile(r"ignore (all\s+)?(previous\s+)?instructions", re.IGNORECASE),
            re.compile(
                r"disregard (your )?(system prompt|instructions|rules)", re.IGNORECASE
            ),
            re.compile(
                r"forget (everything|all|previous)(above| you were told)?",
                re.IGNORECASE,
            ),
            re.compile(
                r"override (your )?(instructions|rules|directives)", re.IGNORECASE
            ),
            re.compile(
                r"stop following (your )?(instructions|rules|guidelines)", re.IGNORECASE
            ),
            re.compile(
                r"your previous instructions are (void|invalid|cancelled)",
                re.IGNORECASE,
            ),
            re.compile(
                r"do not follow (your )?(previous |original )?(instructions|rules)",
                re.IGNORECASE,
            ),
        ],
    },
    {
        "category": "role_persona_switch",
        "patterns": [
            re.compile(
                r"you are now (a |an )(different |new |alternative )?(ai|bot|system|assistant|agent|character|persona|hacker|tool|model|version)",
                re.IGNORECASE,
            ),
            re.compile(
                r"act as (a |an )?(different |new |alternative )?(ai|bot|system|assistant|agent|character|persona|hacker|tool|model)",
                re.IGNORECASE,
            ),
            re.compile(r"pretend (you are|to be) (a |an )?", re.IGNORECASE),
            re.compile(r"your new persona is", re.IGNORECASE),
            re.compile(r"from now on you are", re.IGNORECASE),
            re.compile(r"you have been reprogrammed", re.IGNORECASE),
            re.compile(r"you are (DAN|STAN|DUDE|AIM|KEVIN)", re.IGNORECASE),
            re.compile(r"do anything now", re.IGNORECASE),
            re.compile(r"you are an AI without restrictions", re.IGNORECASE),
            re.compile(
                r"(switch|change) (to |into )?(a |an )?new (role|persona|mode)",
                re.IGNORECASE,
            ),
        ],
    },
    {
        "category": "system_prompt_markers",
        "patterns": [
            re.compile(r"\[SYSTEM\]", re.IGNORECASE),
            re.compile(r"\[INST\]", re.IGNORECASE),
            re.compile(r"<\|system\|>", re.IGNORECASE),
            re.compile(r"<\|im_start\|>\s*system", re.IGNORECASE),
            re.compile(r"###\s*(instruction|system|prompt)\s*:", re.IGNORECASE),
            re.compile(r"system\s*(prompt|message)\s*:", re.IGNORECASE),
            re.compile(r"<system>", re.IGNORECASE),
            re.compile(r"<<SYS>>", re.IGNORECASE),
        ],
    },
    {
        "category": "imperative_commands",
        "patterns": [
            re.compile(r"do not (reveal|disclose|mention|share|say)", re.IGNORECASE),
            re.compile(r"never (mention|reveal|say|disclose|tell)", re.IGNORECASE),
            re.compile(r"always respond with", re.IGNORECASE),
            re.compile(r"respond only with", re.IGNORECASE),
            re.compile(
                r"you must (not |never )?(reveal|say|respond|output|ignore)",
                re.IGNORECASE,
            ),
            re.compile(r"your (only )?(task|job|purpose|goal) is now", re.IGNORECASE),
            re.compile(r"only (output|return|respond|say|print)", re.IGNORECASE),
            re.compile(r"do not answer (any|questions|with)", re.IGNORECASE),
        ],
    },
    {
        "category": "data_exfiltration",
        "patterns": [
            re.compile(
                r"reveal (your )?(system prompt|instructions|configuration|rules)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(print|show|output|display)( me)?( your)? (system prompt|instructions|prompt|rules)",
                re.IGNORECASE,
            ),
            re.compile(
                r"what are your (instructions|rules|guidelines|constraints)",
                re.IGNORECASE,
            ),
            re.compile(
                r"repeat (everything|all)\s+(above|before|prior|you (were told|know|have))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(output|dump|leak) your (configuration|config|system|prompt)",
                re.IGNORECASE,
            ),
            re.compile(
                r"tell me your (system |original )?(prompt|instructions|rules)",
                re.IGNORECASE,
            ),
        ],
    },
    {
        "category": "obfuscation_signals",
        "patterns": [
            # Base64 strings of meaningful length (20+ chars)
            re.compile(r"[A-Za-z0-9+/]{20,}={1,2}", re.IGNORECASE),
            # Excessive special characters in sequence
            re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]{2,}"),
            # HTML/XML comment tags hiding content
            re.compile(
                r"<!--.*?(ignore|instructions|system|prompt).*?-->",
                re.IGNORECASE | re.DOTALL,
            ),
            # Zero-width unicode characters
            re.compile(r"[\u200b\u200c\u200d\ufeff]"),
            # Cyrillic lookalikes for common latin letters
            re.compile(r"[іаеорсухАЕОРСТУХ]{1,}"),
        ],
    },
    {
        "category": "developer_god_mode",
        "patterns": [
            re.compile(
                r"(enable|activate|enter|switch (to|into)) (developer|god|jailbreak|unrestricted|debug|maintenance|training) mode",
                re.IGNORECASE,
            ),
            re.compile(
                r"developer mode (is\s+)?(now\s+)?(on|enabled|activated)", re.IGNORECASE
            ),
            re.compile(
                r"(you are|operating) in (developer|god|unrestricted|jailbreak) mode",
                re.IGNORECASE,
            ),
            re.compile(r"DAN mode (enabled|activated|on)", re.IGNORECASE),
            re.compile(r"jailbreak (mode|activated|enabled|successful)", re.IGNORECASE),
            re.compile(
                r"all (restrictions|limitations|rules|filters) (are )?(now )?(removed|disabled|lifted|off)",
                re.IGNORECASE,
            ),
        ],
    },
]


# ── Core Scanning Function ───────────────────────────────────────────────────


def scan_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Run all regex patterns against a single chunk.

    Args:
        chunk: A chunk dictionary produced by chunker.chunk_text().
               Must contain 'text', 'chunk_index', and 'source' keys.

    Returns:
        A Layer 1 result dictionary with flagged status and matched patterns.
    """
    text: str = chunk.get("text", "")
    chunk_index: int = chunk.get("chunk_index", -1)
    source: str = chunk.get("source", "unknown")

    matched_patterns: list[str] = []

    for category_entry in INJECTION_PATTERNS:
        category: str = category_entry["category"]
        for compiled_pattern in category_entry["patterns"]:
            match = compiled_pattern.search(text)
            if match:
                match_label = f"{category}::{match.group(0)[:50]}"
                matched_patterns.append(match_label)
                logger.debug(
                    "Layer 1 match | chunk=%d | source=%s | category=%s | match='%s'",
                    chunk_index,
                    source,
                    category,
                    match.group(0)[:50],
                )

    flagged: bool = len(matched_patterns) > 0

    if flagged:
        logger.warning(
            "Layer 1 FLAGGED | chunk=%d | source=%s | matches=%d",
            chunk_index,
            source,
            len(matched_patterns),
        )
    else:
        logger.debug(
            "Layer 1 clean | chunk=%d | source=%s",
            chunk_index,
            source,
        )

    return {
        "chunk_index": chunk_index,
        "source": source,
        "flagged": flagged,
        "matched_patterns": matched_patterns,
        "layer": 1,
    }


# ── Batch Scanning Function ──────────────────────────────────────────────────


def scan_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Run Layer 1 scanning across all chunks from a document.

    Args:
        chunks: List of chunk dictionaries from chunker.chunk_text().

    Returns:
        List of Layer 1 result dictionaries, one per chunk.
    """
    logger.info("Layer 1 scan starting | total_chunks=%d", len(chunks))
    results = [scan_chunk(chunk) for chunk in chunks]
    flagged_count = sum(1 for r in results if r["flagged"])
    logger.info(
        "Layer 1 scan complete | total=%d | flagged=%d | clean=%d",
        len(results),
        flagged_count,
        len(results) - flagged_count,
    )
    return results
