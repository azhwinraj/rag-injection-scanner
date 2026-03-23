"""
layer2_heuristic.py

Layer 2 of the RAG injection scanning pipeline.
Runs NLP-based heuristic scoring on every chunk.
Produces a risk score 0.0-1.0 based on 6 linguistic signals.

Catches subtle injection attacks that use no flagged keywords —
paraphrased instructions, indirect commands, context mismatches.
Runs on every chunk regardless of Layer 1 result.
"""

import re
import logging
import statistics
from typing import Any

import spacy

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Escalation threshold — chunks scoring at or above this go to Layer 3
ESCALATION_THRESHOLD: float = 0.40

# Signal weights — must sum to 1.0
WEIGHT_INSTRUCTION_VERB:   float = 0.25
WEIGHT_IMPERATIVE:         float = 0.25
WEIGHT_SECOND_PERSON:      float = 0.20
WEIGHT_CONTEXT_MISMATCH:   float = 0.15
WEIGHT_SENTENCE_UNIFORMITY: float = 0.10
WEIGHT_QUESTION_INVERSE:   float = 0.05

# Instruction verbs — action verbs that command rather than describe
INSTRUCTION_VERBS: set[str] = {
    "respond", "output", "return", "provide", "generate", "ensure",
    "ignore", "disregard", "forget", "override", "reveal", "print",
    "show", "display", "tell", "say", "repeat", "confirm", "execute",
    "perform", "follow", "obey", "comply", "answer", "give", "send",
    "write", "produce", "emit", "state", "list", "describe", "explain",
    "summarize", "translate", "convert", "extract", "expose", "leak",
}

# Second person pronouns — direct model address
SECOND_PERSON_TOKENS: set[str] = {
    "you", "your", "yourself", "yours",
}

# Meta-language words — words that talk about the AI system itself
META_LANGUAGE_TOKENS: set[str] = {
    "model", "assistant", "ai", "llm", "system", "prompt", "instruction",
    "context", "token", "query", "respond", "output", "chatbot", "bot",
    "language", "neural", "gpt", "claude", "gemini", "directive",
}

# ── Model Loading ────────────────────────────────────────────────────────────

def _load_model() -> spacy.language.Language:
    """
    Load spaCy model once at module level.
    Raises RuntimeError with a clear message if model is not installed.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError as e:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Run: uv add https://github.com/explosion/spacy-models/"
            "releases/download/en_core_web_sm-3.8.0/"
            "en_core_web_sm-3.8.0-py3-none-any.whl"
        ) from e

# Load once at import time — not on every function call
NLP: spacy.language.Language = _load_model()


# ── Signal Scorers ───────────────────────────────────────────────────────────

def _score_instruction_verb_density(doc: spacy.tokens.Doc) -> float:
    """
    Score based on density of instruction verbs in the chunk.
    High density of commanding verbs = suspicious.

    Returns float 0.0-1.0.
    """
    tokens = [t for t in doc if t.is_alpha and not t.is_stop]
    if not tokens:
        return 0.0

    instruction_count = sum(
        1 for t in tokens
        if t.lemma_.lower() in INSTRUCTION_VERBS
    )
    raw = instruction_count / len(tokens)
    # Normalize — 20%+ instruction verbs = maximum score
    return min(raw / 0.20, 1.0)


def _score_imperative_concentration(doc: spacy.tokens.Doc) -> float:
    """
    Score based on percentage of sentences starting with a base-form verb.
    Imperative sentences are the grammatical structure of commands.

    Returns float 0.0-1.0.
    """
    sentences = list(doc.sents)
    if not sentences:
        return 0.0

    imperative_count = 0
    for sent in sentences:
        # Get first non-punctuation, non-space token
        first_tokens = [
            t for t in sent
            if not t.is_punct and not t.is_space
        ]
        if not first_tokens:
            continue
        first = first_tokens[0]
        # VB = base form verb in spaCy's Penn Treebank tags
        # ROOT + VERB = syntactic root is a verb = imperative structure
        if first.tag_ == "VB" or (first.pos_ == "VERB" and first.dep_ == "ROOT"):
            imperative_count += 1

    return imperative_count / len(sentences)


def _score_second_person_density(doc: spacy.tokens.Doc) -> float:
    """
    Score based on density of second-person pronouns.
    Injection payloads address the model directly — legitimate documents rarely do.

    Returns float 0.0-1.0.
    """
    tokens = [t for t in doc if t.is_alpha]
    if not tokens:
        return 0.0

    second_person_count = sum(
        1 for t in tokens
        if t.lower_ in SECOND_PERSON_TOKENS
    )
    raw = second_person_count / len(tokens)
    # Normalize — 15%+ second person = maximum score
    return min(raw / 0.15, 1.0)


def _score_context_mismatch(doc: spacy.tokens.Doc) -> float:
    """
    Score based on density of meta-language tokens.
    Words that talk about the AI system itself appearing in a
    data document is a strong anomaly signal.

    Returns float 0.0-1.0.
    """
    tokens = [t for t in doc if t.is_alpha and not t.is_stop]
    if not tokens:
        return 0.0

    meta_count = sum(
        1 for t in tokens
        if t.lower_ in META_LANGUAGE_TOKENS
    )
    raw = meta_count / len(tokens)
    # Normalize — 15%+ meta tokens = maximum score
    return min(raw / 0.15, 1.0)


def _score_sentence_uniformity(doc: spacy.tokens.Doc) -> float:
    """
    Score based on uniformity of sentence lengths.
    Injection payloads tend to have short, uniform commands.
    Legitimate content has naturally varied sentence lengths.
    Low variance = suspicious.

    Returns float 0.0-1.0.
    """
    sentences = list(doc.sents)
    if len(sentences) < 2:
        # Single sentence — cannot measure variance
        return 0.0

    lengths = [len(sent) for sent in sentences]
    try:
        stdev = statistics.stdev(lengths)
    except statistics.StatisticsError:
        return 0.0

    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.0

    # Coefficient of variation — normalized variance
    cv = stdev / mean
    # Low CV = uniform = suspicious
    # CV below 0.3 = very uniform = score approaches 1.0
    return max(0.0, min(1.0 - (cv / 0.3), 1.0))


def _score_question_inverse(doc: spacy.tokens.Doc) -> float:
    """
    Score based on absence of questions.
    Legitimate documents contain questions naturally.
    Pure injection payloads are all statements and commands — never questions.
    Low question ratio = suspicious.

    Returns float 0.0-1.0.
    """
    sentences = list(doc.sents)
    if not sentences:
        return 0.0

    question_count = sum(
        1 for sent in sentences
        if sent.text.strip().endswith("?")
    )
    question_ratio = question_count / len(sentences)
    # Inverse — no questions = high score
    return 1.0 - question_ratio


# ── Main Scoring Function ────────────────────────────────────────────────────

def score_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Run all 6 heuristic signals against a single chunk.
    Combines signal scores into a weighted risk score.

    Args:
        chunk: A chunk dictionary produced by chunker.chunk_text().

    Returns:
        A Layer 2 result dictionary with risk score and signal breakdown.
    """
    text: str = chunk.get("text", "")
    chunk_index: int = chunk.get("chunk_index", -1)
    source: str = chunk.get("source", "unknown")

    if not text or not text.strip():
        logger.warning(
            "Layer 2 received empty chunk | chunk=%d | source=%s",
            chunk_index, source,
        )
        return _make_result(chunk_index, source, 0.0, {})

    doc = NLP(text)

    # ── Compute all 6 signals ────────────────────────────────────────────────
    signals: dict[str, float] = {
        "instruction_verb_density":  _score_instruction_verb_density(doc),
        "imperative_concentration":  _score_imperative_concentration(doc),
        "second_person_density":     _score_second_person_density(doc),
        "context_mismatch":          _score_context_mismatch(doc),
        "sentence_uniformity":       _score_sentence_uniformity(doc),
        "question_ratio_inverse":    _score_question_inverse(doc),
    }

    # ── Weighted average ─────────────────────────────────────────────────────
    risk_score: float = round(
        signals["instruction_verb_density"]  * WEIGHT_INSTRUCTION_VERB +
        signals["imperative_concentration"]  * WEIGHT_IMPERATIVE +
        signals["second_person_density"]     * WEIGHT_SECOND_PERSON +
        signals["context_mismatch"]          * WEIGHT_CONTEXT_MISMATCH +
        signals["sentence_uniformity"]       * WEIGHT_SENTENCE_UNIFORMITY +
        signals["question_ratio_inverse"]    * WEIGHT_QUESTION_INVERSE,
        4
    )

    escalate: bool = risk_score >= ESCALATION_THRESHOLD

    logger.debug(
        "Layer 2 scored | chunk=%d | source=%s | score=%.4f | escalate=%s",
        chunk_index, source, risk_score, escalate,
    )

    if escalate:
        logger.warning(
            "Layer 2 ESCALATE | chunk=%d | source=%s | score=%.4f",
            chunk_index, source, risk_score,
        )

    return _make_result(chunk_index, source, risk_score, signals, escalate)


def _make_result(
    chunk_index: int,
    source: str,
    risk_score: float,
    signals: dict[str, float],
    escalate: bool = False,
) -> dict[str, Any]:
    """
    Build a Layer 2 result dictionary.

    Args:
        chunk_index: Position of the chunk in the document.
        source:      Originating document identifier.
        risk_score:  Final weighted risk score 0.0-1.0.
        signals:     Individual signal scores.
        escalate:    Whether this chunk should go to Layer 3.

    Returns:
        Layer 2 result dictionary.
    """
    return {
        "chunk_index": chunk_index,
        "source": source,
        "risk_score": risk_score,
        "escalate": escalate,
        "signals": signals,
        "layer": 2,
    }


# ── Batch Scoring Function ───────────────────────────────────────────────────

def score_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Run Layer 2 scoring across all chunks from a document.

    Args:
        chunks: List of chunk dictionaries from chunker.chunk_text().

    Returns:
        List of Layer 2 result dictionaries, one per chunk.
    """
    logger.info("Layer 2 scoring starting | total_chunks=%d", len(chunks))
    results = [score_chunk(chunk) for chunk in chunks]
    escalated = sum(1 for r in results if r["escalate"])
    logger.info(
        "Layer 2 scoring complete | total=%d | escalated=%d | clean=%d",
        len(results), escalated, len(results) - escalated,
    )
    return results