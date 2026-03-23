"""
test_classifier.py
Pytest test suite for the Risk Classifier.
Tests all decision tree paths without calling any external APIs.
"""

import pytest
from rag_scanner.classifier import classify_chunk, classify_all_chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(index: int = 0, source: str = "test.txt") -> dict:
    return {"chunk_index": index, "source": source, "text": "some text"}


def make_l1(flagged: bool, patterns: list = None) -> dict:
    return {
        "chunk_index": 0,
        "flagged": flagged,
        "matched_patterns": patterns or [],
        "layer": 1,
    }


def make_l2(score: float, escalate: bool) -> dict:
    return {
        "chunk_index": 0,
        "risk_score": score,
        "escalate": escalate,
        "layer": 2,
    }


def make_l3(classification: str, confidence: float, reasoning: str = "") -> dict:
    return {
        "chunk_index": 0,
        "classification": classification,
        "confidence": confidence,
        "reasoning": reasoning or f"Test reasoning for {classification}",
        "layer": 3,
    }


# ── Output structure tests ────────────────────────────────────────────────────

def test_classify_chunk_returns_required_fields():
    """classify_chunk result contains all required fields."""
    result = classify_chunk(
        make_chunk(),
        make_l1(False),
        make_l2(0.1, False),
        None,
    )
    assert "chunk_index" in result
    assert "source" in result
    assert "risk_level" in result
    assert "reason" in result
    assert "layer1_flagged" in result
    assert "layer2_score" in result
    assert "layer3_ran" in result
    assert "layer3_classification" in result
    assert "layer3_confidence" in result


# ── CLEAN paths ───────────────────────────────────────────────────────────────

def test_all_clear_is_clean():
    """No flags from any layer = CLEAN."""
    result = classify_chunk(
        make_chunk(),
        make_l1(False),
        make_l2(0.1, False),
        None,
    )
    assert result["risk_level"] == "CLEAN"


def test_l3_data_high_confidence_no_l1_is_clean():
    """Layer 3 DATA with high confidence and no L1 flag = CLEAN."""
    result = classify_chunk(
        make_chunk(),
        make_l1(False),
        make_l2(0.45, True),
        make_l3("DATA", 0.95),
    )
    assert result["risk_level"] == "CLEAN"


# ── SUSPICIOUS paths ──────────────────────────────────────────────────────────

def test_l1_flagged_no_l3_is_suspicious():
    """Layer 1 flagged, Layer 3 not run = SUSPICIOUS."""
    result = classify_chunk(
        make_chunk(),
        make_l1(True, ["instruction_override::ignore"]),
        make_l2(0.2, False),
        None,
    )
    assert result["risk_level"] == "SUSPICIOUS"


def test_l2_escalated_no_l3_is_suspicious():
    """Layer 2 escalated, Layer 3 not run = SUSPICIOUS."""
    result = classify_chunk(
        make_chunk(),
        make_l1(False),
        make_l2(0.55, True),
        None,
    )
    assert result["risk_level"] == "SUSPICIOUS"


def test_l3_uncertain_is_suspicious():
    """Layer 3 UNCERTAIN classification = SUSPICIOUS."""
    result = classify_chunk(
        make_chunk(),
        make_l1(False),
        make_l2(0.45, True),
        make_l3("UNCERTAIN", 0.0),
    )
    assert result["risk_level"] == "SUSPICIOUS"


def test_l3_low_confidence_data_is_suspicious():
    """Layer 3 DATA with confidence below threshold = SUSPICIOUS."""
    result = classify_chunk(
        make_chunk(),
        make_l1(False),
        make_l2(0.45, True),
        make_l3("DATA", 0.50),  # Below 0.70 threshold
    )
    assert result["risk_level"] == "SUSPICIOUS"


def test_l3_data_but_l1_flagged_is_suspicious():
    """Layer 3 says DATA but Layer 1 flagged = SUSPICIOUS (conflicting)."""
    result = classify_chunk(
        make_chunk(),
        make_l1(True, ["instruction_override::ignore all"]),
        make_l2(0.45, True),
        make_l3("DATA", 0.95),
    )
    assert result["risk_level"] == "SUSPICIOUS"


# ── DANGEROUS paths ───────────────────────────────────────────────────────────

def test_l3_instruction_is_dangerous():
    """Layer 3 INSTRUCTION classification = DANGEROUS."""
    result = classify_chunk(
        make_chunk(),
        make_l1(True, ["instruction_override::ignore"]),
        make_l2(0.75, True),
        make_l3("INSTRUCTION", 0.99),
    )
    assert result["risk_level"] == "DANGEROUS"


def test_l3_instruction_low_l2_still_dangerous():
    """Layer 3 INSTRUCTION is DANGEROUS regardless of L2 score."""
    result = classify_chunk(
        make_chunk(),
        make_l1(False),
        make_l2(0.1, False),
        make_l3("INSTRUCTION", 0.85),
    )
    assert result["risk_level"] == "DANGEROUS"


def test_l3_instruction_confidence_1_is_dangerous():
    """Layer 3 INSTRUCTION with confidence 1.0 = DANGEROUS."""
    result = classify_chunk(
        make_chunk(),
        make_l1(True, ["role_persona_switch::you are now"]),
        make_l2(0.80, True),
        make_l3("INSTRUCTION", 1.0),
    )
    assert result["risk_level"] == "DANGEROUS"


# ── Reason string tests ───────────────────────────────────────────────────────

def test_dangerous_reason_contains_confidence():
    """DANGEROUS reason string includes L3 confidence."""
    result = classify_chunk(
        make_chunk(),
        make_l1(False),
        make_l2(0.5, True),
        make_l3("INSTRUCTION", 0.97, "Direct commands detected"),
    )
    assert "0.97" in result["reason"]


def test_suspicious_reason_contains_score():
    """SUSPICIOUS reason from L2 includes the score."""
    result = classify_chunk(
        make_chunk(),
        make_l1(False),
        make_l2(0.55, True),
        None,
    )
    assert "0.55" in result["reason"]


def test_clean_reason_is_descriptive():
    """CLEAN reason explains why chunk was cleared."""
    result = classify_chunk(
        make_chunk(),
        make_l1(False),
        make_l2(0.1, False),
        None,
    )
    assert len(result["reason"]) > 0


# ── Batch classifier tests ────────────────────────────────────────────────────

def test_classify_all_chunks_returns_one_per_chunk():
    """classify_all_chunks returns one result per input chunk."""
    chunks = [make_chunk(i) for i in range(3)]
    l1 = [make_l1(False) for _ in range(3)]
    l2 = [make_l2(0.1, False) for _ in range(3)]
    results = classify_all_chunks(chunks, l1, l2, [])
    assert len(results) == 3


def test_classify_all_chunks_l3_lookup_by_index():
    """classify_all_chunks correctly matches L3 results by chunk_index."""
    chunks = [make_chunk(i) for i in range(3)]
    l1 = [make_l1(False) for _ in range(3)]
    l2 = [make_l2(0.1, False) for _ in range(3)]

    # Only chunk index 1 has an L3 result
    l3_result = make_l3("INSTRUCTION", 0.99)
    l3_result["chunk_index"] = 1

    results = classify_all_chunks(chunks, l1, l2, [l3_result])

    assert results[0]["risk_level"] == "CLEAN"
    assert results[1]["risk_level"] == "DANGEROUS"
    assert results[2]["risk_level"] == "CLEAN"