"""
classifier.py

Risk Classifier — Module 4 of the RAG injection scanning pipeline.
Combines Layer 1, Layer 2, and Layer 3 signals into a final risk verdict.

Three possible risk levels:
    CLEAN      — no evidence of injection, safe to ingest
    SUSPICIOUS — conflicting signals or uncertain LLM result, needs review
    DANGEROUS  — LLM confirmed INSTRUCTION with high confidence, block ingestion

Decision logic:
    If Layer 3 ran:
        INSTRUCTION classification            → DANGEROUS
        UNCERTAIN or low confidence (<0.70)   → SUSPICIOUS
        DATA with high confidence             →
            Layer 1 also flagged              → SUSPICIOUS (conflicting)
            No other flags                    → CLEAN
    If Layer 3 skipped:
        Layer 1 flagged OR Layer 2 >= 0.40    → SUSPICIOUS
        Neither                               → CLEAN
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Risk level labels
RISK_CLEAN:      str = "CLEAN"
RISK_SUSPICIOUS: str = "SUSPICIOUS"
RISK_DANGEROUS:  str = "DANGEROUS"

# Minimum Layer 3 confidence to trust a DATA classification
MIN_L3_CONFIDENCE: float = 0.70

# Layer 2 escalation threshold — must match layer2_heuristic.py
L2_ESCALATION_THRESHOLD: float = 0.40


# ── Core Classifier ───────────────────────────────────────────────────────────

def classify_chunk(
    chunk: dict[str, Any],
    layer1_result: dict[str, Any] | None,
    layer2_result: dict[str, Any] | None,
    layer3_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Classify a single chunk based on all available layer signals.

    Args:
        chunk:         Original chunk dictionary from the chunker.
        layer1_result: Layer 1 regex scan result for this chunk.
        layer2_result: Layer 2 heuristic score result for this chunk.
        layer3_result: Layer 3 LLM judge result, or None if skipped.

    Returns:
        A classification result dictionary with risk level and reason.
    """
    chunk_index: int = chunk.get("chunk_index", -1)
    source: str = chunk.get("source", "unknown")

    # ── Extract signals ───────────────────────────────────────────────────────
    l1_flagged: bool = bool(layer1_result and layer1_result.get("flagged", False))
    l1_patterns: list[str] = (
        layer1_result.get("matched_patterns", []) if layer1_result else []
    )

    l2_score: float = float(
        layer2_result.get("risk_score", 0.0) if layer2_result else 0.0
    )
    l2_escalated: bool = bool(
        layer2_result and layer2_result.get("escalate", False)
    )

    l3_ran: bool = layer3_result is not None
    l3_classification: str = (
        layer3_result.get("classification", "UNCERTAIN") if l3_ran else "SKIPPED"
    )
    l3_confidence: float = float(
        layer3_result.get("confidence", 0.0) if l3_ran else 0.0
    )
    l3_reasoning: str = (
        layer3_result.get("reasoning", "") if l3_ran else ""
    )

    # ── Decision logic ────────────────────────────────────────────────────────
    risk_level: str
    reason: str

    if l3_ran:
        if l3_classification == "INSTRUCTION":
            risk_level = RISK_DANGEROUS
            reason = (
                f"Layer 3 classified as INSTRUCTION with confidence "
                f"{l3_confidence:.2f}. Reasoning: {l3_reasoning}"
            )

        elif l3_classification == "UNCERTAIN" or l3_confidence < MIN_L3_CONFIDENCE:
            risk_level = RISK_SUSPICIOUS
            reason = (
                f"Layer 3 returned uncertain result "
                f"(classification={l3_classification}, "
                f"confidence={l3_confidence:.2f}). "
                f"Manual review recommended."
            )

        elif l3_classification == "DATA" and l3_confidence >= MIN_L3_CONFIDENCE:
            if l1_flagged:
                risk_level = RISK_SUSPICIOUS
                reason = (
                    f"Layer 3 classified as DATA (confidence={l3_confidence:.2f}) "
                    f"but Layer 1 flagged {len(l1_patterns)} pattern(s): "
                    f"{l1_patterns[:3]}. Conflicting signals — review recommended."
                )
            else:
                risk_level = RISK_CLEAN
                reason = (
                    f"Layer 3 classified as DATA with confidence "
                    f"{l3_confidence:.2f}. No conflicting signals."
                )

        else:
            # Fallback — should never reach here but fail safe
            risk_level = RISK_SUSPICIOUS
            reason = f"Unexpected Layer 3 state: {l3_classification}"

    else:
        # Layer 3 was not run
        if l1_flagged and l2_escalated:
            risk_level = RISK_SUSPICIOUS
            reason = (
                f"Layer 1 flagged {len(l1_patterns)} pattern(s) and "
                f"Layer 2 scored {l2_score:.4f} (>= {L2_ESCALATION_THRESHOLD}). "
                f"Layer 3 not run."
            )
        elif l1_flagged:
            risk_level = RISK_SUSPICIOUS
            reason = (
                f"Layer 1 flagged {len(l1_patterns)} pattern(s): "
                f"{l1_patterns[:3]}. Layer 3 not run."
            )
        elif l2_escalated:
            risk_level = RISK_SUSPICIOUS
            reason = (
                f"Layer 2 scored {l2_score:.4f} "
                f"(>= {L2_ESCALATION_THRESHOLD}). Layer 3 not run."
            )
        else:
            risk_level = RISK_CLEAN
            reason = (
                f"No flags from Layer 1. "
                f"Layer 2 score {l2_score:.4f} below threshold. "
                f"Layer 3 not required."
            )

    logger.info(
        "Classified | chunk=%d | source=%s | risk=%s",
        chunk_index, source, risk_level,
    )

    return {
        "chunk_index":          chunk_index,
        "source":               source,
        "risk_level":           risk_level,
        "reason":               reason,
        "layer1_flagged":       l1_flagged,
        "layer1_patterns":      l1_patterns,
        "layer2_score":         l2_score,
        "layer3_ran":           l3_ran,
        "layer3_classification": l3_classification,
        "layer3_confidence":    l3_confidence,
    }


# ── Batch Classifier ──────────────────────────────────────────────────────────

def classify_all_chunks(
    chunks: list[dict[str, Any]],
    layer1_results: list[dict[str, Any]],
    layer2_results: list[dict[str, Any]],
    layer3_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Classify all chunks by combining signals from all three layers.

    Args:
        chunks:         All chunks from the document.
        layer1_results: Layer 1 results aligned by chunk index.
        layer2_results: Layer 2 results aligned by chunk index.
        layer3_results: Layer 3 results — only for escalated chunks.

    Returns:
        List of classification results, one per chunk.
    """
    # Build a lookup for Layer 3 results by chunk_index
    # Layer 3 only runs on flagged chunks so we can't index by position
    l3_lookup: dict[int, dict[str, Any]] = {
        r["chunk_index"]: r for r in layer3_results
    }

    results: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        l1 = layer1_results[i] if i < len(layer1_results) else None
        l2 = layer2_results[i] if i < len(layer2_results) else None
        l3 = l3_lookup.get(i)

        result = classify_chunk(chunk, l1, l2, l3)
        results.append(result)

    # Summary logging
    counts = {
        RISK_CLEAN:      sum(1 for r in results if r["risk_level"] == RISK_CLEAN),
        RISK_SUSPICIOUS: sum(1 for r in results if r["risk_level"] == RISK_SUSPICIOUS),
        RISK_DANGEROUS:  sum(1 for r in results if r["risk_level"] == RISK_DANGEROUS),
    }

    logger.info(
        "Classification complete | total=%d | clean=%d | suspicious=%d | dangerous=%d",
        len(results), counts[RISK_CLEAN],
        counts[RISK_SUSPICIOUS], counts[RISK_DANGEROUS],
    )

    return results