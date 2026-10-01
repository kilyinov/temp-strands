"""Deterministic profile checks. Used by offline evals, online scoring and as hard CI gates."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ..evidence import own_words, quote_in_own_words, within_membership
from ..models import TOPICS, Claim, Disambiguation, MPProfile, Parliament, ProfileResponse, Topic
from ..store import SpeechStore
from ..text import fold_quote_text

EVALUATIVE_TERMS = re.compile(
    r"\b(champion\w*|hypocri\w*|disgrace\w*|radical|extremist\w*|brilliant\w*|heroic\w*|shameful\w*|reckless\w*|"
    r"admirabl\w*|woke|liar\w*|lies|incompeten\w*|visionary|courageous\w*|dishonest\w*)\b",
    re.IGNORECASE,
)


class GoldenCase(BaseModel):
    name: str
    query: str
    expect_status: str
    expect_person_id: str | None = None
    expect_candidates: list[str] = Field(default_factory=list)
    expect_claims_topics: list[Topic] = Field(default_factory=list)
    expect_no_claims_topics: list[Topic] = Field(default_factory=list)
    expect_parliaments: list[Parliament] = Field(default_factory=list)
    forbidden_quotes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class CheckResult:
    name: str
    score: float
    passed: bool
    detail: str = ""


def _claims(profile: MPProfile) -> list[tuple[Topic, Claim]]:
    return [(c.category, claim) for c in profile.categories for p in c.periods for claim in p.claims]


def _rate(name: str, good: int, total: int, threshold: float = 1.0, detail: str = "") -> CheckResult:
    score = good / total if total else 1.0
    return CheckResult(name, round(score, 4), score >= threshold, detail or f"{good}/{total}")


def evidence_checks(store: SpeechStore, profile: MPProfile) -> list[CheckResult]:
    claims = _claims(profile)
    passages = {c.ref.passage_id: store.get_passage(c.ref.passage_id) for _, c in claims}
    quote_ok = attributed = on_topic = in_office = 0
    for topic, claim in claims:
        passage = passages[claim.ref.passage_id]
        if passage is None:
            continue
        quote_ok += quote_in_own_words(claim.quote, passage)
        attributed += store.person_for_source(passage.source_speaker_id) == profile.person.person_id
        on_topic += topic in {t.topic for t in store.passage_topics(passage.passage_id)}
        in_office += within_membership(profile.person, passage)
    total = len(claims)
    return [
        _rate("quote_fidelity", quote_ok, total),
        _rate("speaker_attribution", attributed, total, threshold=0.99),
        _rate("topic_relevance", on_topic, total, threshold=0.9),
        _rate("membership_alignment", in_office, total),
    ]


def structure_checks(profile: MPProfile) -> list[CheckResult]:
    order = [c.category for c in profile.categories]
    results = [CheckResult("four_categories", float(order == list(TOPICS)), order == list(TOPICS), str(order))]
    honest = [
        c.category.value
        for c in profile.categories
        if c.passage_count == 0 and (any(p.claims for p in c.periods) or not c.gaps)
    ]
    results.append(CheckResult("gap_honesty", float(not honest), not honest, f"violations: {honest}"))
    texts = [c.career_summary for c in profile.categories] + [c.evolution or "" for c in profile.categories]
    texts += [p.summary for c in profile.categories for p in c.periods]
    texts += [claim.paraphrase for _, claim in _claims(profile)]
    flagged = sorted({m.group(0).lower() for t in texts for m in EVALUATIVE_TERMS.finditer(t)})
    results.append(CheckResult("neutrality_lexicon", float(not flagged), not flagged, f"terms: {flagged}"))
    return results


def golden_checks(store: SpeechStore, response: ProfileResponse, case: GoldenCase) -> list[CheckResult]:
    results = [
        CheckResult(
            "status",
            float(response.status == case.expect_status),
            response.status == case.expect_status,
            f"expected {case.expect_status}, got {response.status}",
        )
    ]
    if isinstance(response, Disambiguation):
        ids = {c.person_id for c in response.candidates}
        missing = [c for c in case.expect_candidates if c not in ids]
        results.append(CheckResult("disambiguation_candidates", float(not missing), not missing, f"missing {missing}"))
        return results
    if case.expect_person_id:
        ok = response.person.person_id == case.expect_person_id
        results.append(CheckResult("person_resolution", float(ok), ok, response.person.person_id))
    results += evidence_checks(store, response) + structure_checks(response)
    by_topic = {c.category: c for c in response.categories}
    claims = _claims(response)
    for topic in case.expect_claims_topics:
        n = sum(1 for p in by_topic[topic].periods for _ in p.claims)
        results.append(CheckResult(f"has_claims:{topic.value}", float(n > 0), n > 0, f"{n} claims"))
    for topic in case.expect_no_claims_topics:
        n = sum(1 for p in by_topic[topic].periods for _ in p.claims)
        ok = n == 0 and bool(by_topic[topic].gaps)
        results.append(
            CheckResult(f"no_fabrication:{topic.value}", float(ok), ok, f"{n} claims, gaps={by_topic[topic].gaps}")
        )
    cited = {c.ref.parliament for _, c in claims}
    for parliament in case.expect_parliaments:
        ok = parliament in cited
        results.append(CheckResult(f"jurisdiction_coverage:{parliament}", float(ok), ok, f"cited {sorted(cited)}"))
    leaked = [
        q for q in case.forbidden_quotes if any(fold_quote_text(q) in fold_quote_text(c.quote) for _, c in claims)
    ]
    results.append(CheckResult("context_fidelity", float(not leaked), not leaked, f"leaked {leaked}"))
    return results


def trajectory_check(response: ProfileResponse, trajectory: list[str]) -> CheckResult:
    if isinstance(response, Disambiguation):
        ok = trajectory == ["resolve"]
    else:
        middle = set(trajectory[2:-1])
        ok = (
            trajectory[:2] == ["resolve", "plan"]
            and trajectory[-1:] == ["verify"]
            and middle == {t.value for t in TOPICS}
        )
    return CheckResult("trajectory", float(ok), ok, " -> ".join(trajectory))


def own_words_ratio(store: SpeechStore, profile: MPProfile) -> float:
    """Share of cited passage text that is the member's own words (low values hint at quote-heavy evidence)."""
    lengths = [
        (len(own_words(p.text)), len(p.text))
        for _, c in _claims(profile)
        if (p := store.get_passage(c.ref.passage_id)) is not None
    ]
    total = sum(t for _, t in lengths)
    return sum(o for o, _ in lengths) / total if total else 1.0
