"""Deterministic verifier: every claim must be the resolved person's own verbatim words, on topic, in office."""

from __future__ import annotations

from ..evidence import build_ref, quote_in_own_words, within_membership
from ..models import CategorySummary, Claim, Period, Person, VerificationIssue
from ..store import SpeechStore
from ..telemetry import domain_metrics


def verify_claim(
    store: SpeechStore, person: Person, category: CategorySummary, claim: Claim
) -> tuple[Claim | None, VerificationIssue | None]:
    pid = claim.ref.passage_id

    def issue(check: str, detail: str) -> tuple[None, VerificationIssue]:
        domain_metrics().verifier_rejections.add(1, {"check": check, "topic": category.category.value})
        return None, VerificationIssue(category=category.category, passage_id=pid, check=check, detail=detail)  # type: ignore[arg-type]

    passage = store.get_passage(pid)
    if passage is None:
        return issue("unknown_passage", "cited passage does not exist in the speech store")
    if store.person_for_source(passage.source_speaker_id) != person.person_id:
        return issue("speaker_attribution", f"passage is attributed to {passage.speaker_name}")
    if not quote_in_own_words(claim.quote, passage):
        return issue("quote_fidelity", "quote is not a verbatim substring of the member's own words in the passage")
    if not within_membership(person, passage):
        return issue("membership_date", f"{passage.date} is outside the member's recorded service")
    if category.category not in {t.topic for t in store.passage_topics(pid)}:
        return issue("topic_relevance", f"passage is not tagged {category.category.value}")
    return claim.model_copy(update={"ref": build_ref(person, passage)}), None


def verify_category(
    store: SpeechStore, person: Person, category: CategorySummary
) -> tuple[CategorySummary, list[VerificationIssue]]:
    issues: list[VerificationIssue] = []
    periods: list[Period] = []
    for period in category.periods:
        kept = []
        for claim in period.claims:
            verified, problem = verify_claim(store, person, category, claim)
            if verified:
                kept.append(verified)
            if problem:
                issues.append(problem)
        periods.append(period.model_copy(update={"claims": kept}))
    gaps = list(category.gaps)
    if issues:
        gaps.append(f"{len(issues)} claim(s) removed by verification.")
    if any(c.ref.is_proof for p in periods for c in p.claims):
        gaps.append("Some citations are to proof (uncorrected) Hansard.")
    return category.model_copy(update={"periods": periods, "gaps": gaps}), issues
