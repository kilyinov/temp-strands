"""Federal <-> Victorian person crosswalk.

People are never merged on name alone. `suggest` lists candidate pairs for human review; confirmed
pairs go into a CSV (canonical_person_id, alias_person_id, evidence) that `apply` loads.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .store import SpeechStore
from .text import name_tokens


@dataclass(frozen=True)
class CrosswalkSuggestion:
    federal_id: str
    federal_name: str
    victorian_id: str
    victorian_name: str
    reason: str


def _key(name: str) -> tuple[str, str] | None:
    tokens = name_tokens(name)
    return (tokens[0], tokens[-1]) if len(tokens) >= 2 else None


def suggest(store: SpeechStore) -> list[CrosswalkSuggestion]:
    federal: dict[tuple[str, str], list[tuple[str, str]]] = {}
    victorian: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for person in store.list_people():
        parliaments = {m.parliament for m in person.memberships}
        if len(parliaments) > 1:
            continue
        for name in {person.name, *person.aliases}:
            key = _key(name)
            if key is None:
                continue
            target = federal if "commonwealth" in parliaments else victorian
            target.setdefault(key, [])
            if (person.person_id, person.name) not in target[key]:
                target[key].append((person.person_id, person.name))
    out = []
    for key, feds in sorted(federal.items()):
        for vic_id, vic_name in victorian.get(key, []):
            for fed_id, fed_name in feds:
                out.append(CrosswalkSuggestion(fed_id, fed_name, vic_id, vic_name, f"first+last name match {key}"))
    return out


def apply(store: SpeechStore, path: Path) -> int:
    count = 0
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            canonical, alias = row["canonical_person_id"].strip(), row["alias_person_id"].strip()
            if not canonical or not alias:
                continue
            if store.get_person(canonical) is None or store.get_person(alias) is None:
                raise ValueError(f"crosswalk row references unknown person: {canonical} / {alias}")
            store.link_persons(canonical, alias, row.get("evidence", "").strip() or "manual")
            count += 1
    return count
