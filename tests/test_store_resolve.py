from datetime import date
from pathlib import Path

import pytest

from mp_profile import crosswalk
from mp_profile.models import Disambiguation, Topic
from mp_profile.resolve import resolve_person
from mp_profile.store import SpeechStore

from .conftest import ALEX, ALEX_VIC, PAT, SAM


def test_crosswalk_merges_federal_and_victorian_records(store: SpeechStore) -> None:
    assert store.canonical_id(ALEX_VIC) == ALEX
    assert set(store.person_group(ALEX)) == {ALEX, ALEX_VIC}
    alex = store.get_person(ALEX)
    assert alex is not None
    assert {m.parliament for m in alex.memberships} == {"commonwealth", "victoria"}
    assert all(p.person_id != ALEX_VIC for p in store.list_people())


def test_search_by_topic_spans_both_parliaments(store: SpeechStore) -> None:
    found = store.search_passages(ALEX, Topic.HOUSING)
    assert {p.parliament for p, _ in found} == {"commonwealth", "victoria"}
    scores = [s for _, s in found]
    assert scores == sorted(scores, reverse=True)


def test_search_filters_dates_query_and_procedural(store: SpeechStore) -> None:
    assert all(p.date >= date(2019, 1, 1) for p, _ in store.search_passages(ALEX, start=date(2019, 1, 1)))
    assert all("surplus" in p.text.lower() for p, _ in store.search_passages(ALEX, query="surplus"))
    assert store.search_passages("vic:member/9010") == []


def test_add_passages_is_idempotent(store: SpeechStore) -> None:
    before = store.topic_counts(ALEX)
    passage = store.search_passages(ALEX)[0][0]
    store.add_passages([passage])
    assert store.topic_counts(ALEX) == before


def test_summary_cache_roundtrip(store: SpeechStore) -> None:
    assert store.cache_get("k") is None
    store.cache_put("k", {"a": 1})
    assert store.cache_get("k") == {"a": 1}


def test_crosswalk_suggest_and_apply(tmp_path: Path, store: SpeechStore) -> None:
    suggestions = crosswalk.suggest(store)
    assert all(s.victorian_id != ALEX_VIC for s in suggestions)
    bad = tmp_path / "bad.csv"
    bad.write_text("canonical_person_id,alias_person_id,evidence\noa:person/20001,vic:member/99999,\n")
    with pytest.raises(ValueError):
        crosswalk.apply(store, bad)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Alex Morgan", ALEX),
        ("MORGAN, the Hon. Alex", ALEX),
        ("alex morgan mp", ALEX),
        ("Sam Taylor", SAM),
        ("Pat Lee", PAT),
        (ALEX_VIC, ALEX),
    ],
)
def test_resolve(store: SpeechStore, query: str, expected: str) -> None:
    result = resolve_person(store, query)
    assert result.person is not None and result.person.person_id == expected


def test_resolve_ambiguous_and_missing(store: SpeechStore) -> None:
    ambiguous = resolve_person(store, "Taylor").disambiguation
    assert isinstance(ambiguous, Disambiguation) and ambiguous.status == "needs_disambiguation"
    assert {c.person_id for c in ambiguous.candidates} >= {SAM, "vic:member/9002"}
    missing = resolve_person(store, "Nobody Real").disambiguation
    assert missing is not None and missing.status == "not_found"


def test_resolve_parliament_filter(store: SpeechStore) -> None:
    result = resolve_person(store, "Taylor", parliament="victoria")
    assert result.person is not None and result.person.person_id == "vic:member/9002"
