"""Deterministic lexicon topic tagger.

This is the baseline tagger used at ingestion time. Its precision/recall is measured by the
topic-tagging eval against a labelled passage set, so lexicon changes are regression-tested.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import TOPICS, Topic, TopicTag

LEXICON: dict[Topic, dict[str, float]] = {
    Topic.HOUSING: {
        "housing": 3,
        "housing affordability": 4,
        "affordable housing": 4,
        "social housing": 4,
        "public housing": 4,
        "homelessness": 3,
        "homeless": 2,
        "rent": 1.5,
        "rental": 2,
        "renters": 2.5,
        "tenants": 2,
        "tenancy": 2,
        "landlord": 2,
        "mortgage": 2,
        "first home": 3,
        "home buyers": 3,
        "homebuyers": 3,
        "home ownership": 3,
        "dwellings": 2.5,
        "planning scheme": 2,
        "zoning": 1.5,
        "negative gearing": 3,
        "stamp duty": 2,
        "build-to-rent": 3,
        "housing supply": 4,
        "new homes": 2.5,
    },
    Topic.HEALTHCARE: {
        "health": 1.5,
        "healthcare": 3,
        "health care": 3,
        "hospital": 3,
        "hospitals": 3,
        "medicare": 4,
        "bulk billing": 4,
        "bulk-billing": 4,
        "gp": 2,
        "gps": 2,
        "general practitioner": 3,
        "nurses": 2.5,
        "nursing": 2,
        "doctors": 2,
        "patients": 2,
        "mental health": 4,
        "ambulance": 3,
        "elective surgery": 4,
        "emergency department": 3,
        "pbs": 3,
        "pharmaceutical benefits": 4,
        "aged care": 2.5,
        "dental": 2.5,
        "ndis": 1.5,
        "pandemic": 1.5,
        "vaccination": 2.5,
        "vaccine": 2,
    },
    Topic.ECONOMY: {
        "economy": 3,
        "economic": 2,
        "budget": 2,
        "deficit": 2.5,
        "surplus": 2.5,
        "debt": 1.5,
        "inflation": 3,
        "interest rates": 3,
        "cost of living": 3,
        "unemployment": 3,
        "jobs": 1.5,
        "employment": 1.5,
        "wages": 2.5,
        "productivity": 2.5,
        "gdp": 3,
        "growth": 1,
        "tax": 1.5,
        "taxation": 2,
        "taxes": 1.5,
        "gst": 2.5,
        "small business": 2,
        "investment": 1.5,
        "industry": 1,
        "manufacturing": 2,
        "trade": 1,
        "exports": 2,
        "recession": 3,
        "treasury": 2,
        "fiscal": 2.5,
    },
    Topic.EDUCATION: {
        "education": 3,
        "school": 2,
        "schools": 2.5,
        "students": 2,
        "teachers": 2.5,
        "teaching": 1.5,
        "university": 2.5,
        "universities": 2.5,
        "tafe": 3,
        "vocational": 2.5,
        "apprenticeships": 2.5,
        "early childhood": 3,
        "kindergarten": 3,
        "childcare": 2,
        "child care": 2,
        "curriculum": 3,
        "naplan": 3,
        "gonski": 4,
        "hecs": 3,
        "student debt": 3,
        "literacy": 2,
        "numeracy": 2,
        "classroom": 2,
    },
}

_PATTERNS: dict[Topic, list[tuple[str, float, re.Pattern[str]]]] = {
    topic: [(term, weight, re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)) for term, weight in terms.items()]
    for topic, terms in LEXICON.items()
}

TAG_THRESHOLD = 4.0


def score_topics(text: str) -> dict[Topic, tuple[float, list[str]]]:
    """Score text against each topic; score is weight * log-damped hit count, normalised by length."""
    words = max(len(text.split()), 1)
    length_norm = min(1.0, 200 / words) ** 0.5
    results: dict[Topic, tuple[float, list[str]]] = {}
    for topic, patterns in _PATTERNS.items():
        total = 0.0
        matched: list[str] = []
        for term, weight, pattern in patterns:
            hits = len(pattern.findall(text))
            if hits:
                matched.append(term)
                total += weight * (1 + 0.5 * (hits - 1) if hits < 5 else 3.0)
        results[topic] = (round(total * length_norm, 3), matched)
    return results


def tag_passage(text: str, threshold: float = TAG_THRESHOLD) -> list[TopicTag]:
    return [
        TopicTag(topic=topic, score=score, matched_terms=terms)
        for topic, (score, terms) in score_topics(text).items()
        if score >= threshold
    ]


REPORTED_SPEECH = re.compile(
    r"\b(?:the (?:honourable |hon\.? )?(?:member|senator|minister|leader|shadow|opposition|premier|treasurer)\b[^.?!]{0,80}?"
    r"|(?:he|she|they)\s+)(?:has |have |had )?(?:said|says|spoke|claimed|claims|stated|argued|suggested|told|asserted)\b",
    re.IGNORECASE,
)


def best_sentences(text: str, topic: Topic, sentences: Iterable[str], limit: int = 1) -> list[str]:
    """Rank sentences of a passage by topic-term density, for extractive quotes.

    Sentences reporting what someone else said are skipped so they are not cited as the member's own position.
    """
    patterns = _PATTERNS[topic]
    scored = []
    for idx, sentence in enumerate(sentences):
        if len(sentence) < 40 or len(sentence) > 400 or REPORTED_SPEECH.search(sentence):
            continue
        score = sum(weight for _, weight, pattern in patterns if pattern.search(sentence))
        if score:
            scored.append((score, -idx, sentence))
    scored.sort(reverse=True)
    return [s for _, _, s in scored[:limit]]


__all__ = ["LEXICON", "REPORTED_SPEECH", "TAG_THRESHOLD", "TOPICS", "best_sentences", "score_topics", "tag_passage"]
