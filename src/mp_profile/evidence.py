"""Evidence helpers shared by summarisers and the verifier: own-words text, capacity, references."""

from __future__ import annotations

import re
from datetime import date

from .models import Capacity, HansardRef, Passage, Person
from .text import is_verbatim

_MINISTERIAL = re.compile(
    r"\b(minister|premier|treasurer|attorney-general|prime minister|parliamentary secretary)\b", re.I
)
_SHADOW = re.compile(r"\bshadow\b", re.I)


def own_words(text: str) -> str:
    """Passage text minus material the member was quoting or reading in (stored as '> ' lines)."""
    return "\n".join(line for line in text.splitlines() if not line.startswith("> "))


def quote_in_own_words(quote: str, passage: Passage) -> bool:
    return is_verbatim(quote, own_words(passage.text))


def _active(start: date | None, end: date | None, on: date) -> bool:
    return (start is None or start <= on) and (end is None or on <= end)


def role_at(person: Person, passage: Passage) -> str | None:
    if passage.stated_role:
        return passage.stated_role
    active = [r for r in person.roles if r.start and _active(r.start, r.end, passage.date)]
    return "; ".join(sorted({r.title for r in active})) or None


def capacity_at(person: Person, passage: Passage) -> Capacity:
    if passage.talktype == "procedural":
        return "presiding"
    role = role_at(person, passage) or ""
    if _SHADOW.search(role):
        return "shadow"
    if _MINISTERIAL.search(role):
        return "ministerial"
    if "committee" in passage.debate_title.lower():
        return "committee"
    return "personal"


def build_ref(person: Person, passage: Passage) -> HansardRef:
    return HansardRef(
        passage_id=passage.passage_id,
        parliament=passage.parliament,
        chamber=passage.chamber,
        date=passage.date,
        debate_title=passage.debate_title,
        url=passage.url,
        speaker_role=role_at(person, passage),
        is_proof=passage.is_proof,
        parse_confidence=passage.parse_confidence,
    )


def within_membership(person: Person, passage: Passage) -> bool:
    relevant = [m for m in person.memberships if m.parliament == passage.parliament]
    if not relevant:
        return False
    return any(_active(m.start, m.end, passage.date) for m in relevant)
