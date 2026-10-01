"""Deterministic person resolution across federal and Victorian registries."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .models import Disambiguation, Parliament, Person, PersonCandidate
from .store import SpeechStore
from .text import name_tokens

CHAMBER_LABELS = {
    "house_of_reps": "House of Representatives",
    "senate": "Senate",
    "legislative_assembly": "Victorian Legislative Assembly",
    "legislative_council": "Victorian Legislative Council",
}

MIN_SCORE = 0.6
RESOLVE_SCORE = 0.85
SOLE_MATCH_SCORE = 0.75
RESOLVE_MARGIN = 0.1


def _variant_score(query: list[str], variant: list[str]) -> float:
    if not query or not variant:
        return 0.0
    q, v = set(query), set(variant)
    if q == v:
        return 1.0
    surname = variant[-1]
    if surname in q:
        others = [t for t in query if t != surname]
        if others and all(any(vt.startswith(t) or t.startswith(vt) for vt in variant[:-1]) for t in others):
            return 0.93
        if not others:
            return 0.75
    if q <= v:
        return 0.9 if len(q) >= 2 else 0.7
    return 0.8 * SequenceMatcher(None, " ".join(query), " ".join(variant)).ratio()


def career_summary(person: Person) -> str:
    parts = []
    for m in person.memberships:
        span = f"{m.start.year if m.start else '?'}–{m.end.year if m.end else 'present'}"
        if m.start is None and m.end is None:
            span = "dates not recorded"
        parts.append(f"{CHAMBER_LABELS[m.chamber]}, {m.seat} ({m.party}, {span})")
    return "; ".join(parts) or "no membership records"


def score_people(store: SpeechStore, query: str, parliament: Parliament | None = None) -> list[PersonCandidate]:
    q = name_tokens(query)
    candidates = []
    for person in store.list_people():
        if parliament and not any(m.parliament == parliament for m in person.memberships):
            continue
        score = max(_variant_score(q, name_tokens(n)) for n in [person.name, *person.aliases])
        if score >= MIN_SCORE:
            candidates.append(
                PersonCandidate(
                    person_id=person.person_id,
                    name=person.name,
                    score=round(score, 3),
                    summary=career_summary(person),
                )
            )
    return sorted(candidates, key=lambda c: (-c.score, c.name))


@dataclass
class Resolution:
    person: Person | None
    disambiguation: Disambiguation | None


def resolve_person(store: SpeechStore, query: str, parliament: Parliament | None = None) -> Resolution:
    """Resolve to exactly one person, or return a disambiguation response. Never guesses between close matches."""
    direct = store.get_person(query.strip())
    if direct:
        return Resolution(direct, None)
    candidates = score_people(store, query, parliament)
    if not candidates:
        return Resolution(
            None,
            Disambiguation(
                status="not_found",
                query=query,
                message=f"No Commonwealth or Victorian member matching '{query}' is in the speech store.",
            ),
        )
    top = candidates[0]
    runner_up = candidates[1].score if len(candidates) > 1 else 0.0
    exact_unique = top.score == 1.0 and runner_up < 1.0
    sole_match = len(candidates) == 1 and top.score >= SOLE_MATCH_SCORE
    if exact_unique or sole_match or (top.score >= RESOLVE_SCORE and top.score - runner_up >= RESOLVE_MARGIN):
        return Resolution(store.get_person(top.person_id), None)
    return Resolution(
        None,
        Disambiguation(
            status="needs_disambiguation",
            query=query,
            candidates=candidates[:8],
            message=f"'{query}' matches more than one member. Choose one by person_id.",
        ),
    )
