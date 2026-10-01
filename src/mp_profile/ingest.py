"""Ingestion jobs: registries and Hansard into the speech store. Idempotent and resumable per sitting."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date

from opentelemetry import trace

from .models import Chamber, Passage, Person
from .sources import handbook, openaustralia, victoria
from .sources.http import HttpClient
from .store import SpeechStore
from .telemetry import domain_metrics

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("mp_profile.ingest")


def ingest_people(store: SpeechStore, people: Iterable[Person]) -> int:
    count = 0
    for person in people:
        store.upsert_person(person)
        count += 1
    return count


def ingest_federal_people(store: SpeechStore, client: HttpClient) -> int:
    with tracer.start_as_current_span("ingest.federal_people") as span:
        count = ingest_people(store, openaustralia.fetch_people(client))
        span.set_attribute("mp_profile.people", count)
        return count


def ingest_victorian_people(store: SpeechStore, client: HttpClient) -> int:
    with tracer.start_as_current_span("ingest.victorian_people") as span:
        count = ingest_people(store, victoria.fetch_members(client))
        span.set_attribute("mp_profile.people", count)
        return count


def link_handbook(store: SpeechStore, client: HttpClient) -> int:
    """Attach APH PHIDs to federal people as extra source ids (aph:<PHID>)."""
    records = handbook.fetch_records(client)
    linked = 0
    for person in store.list_people():
        if not person.person_id.startswith("oa:") or len(store.person_group(person.person_id)) > 1:
            continue
        phid = handbook.match_phid(person, records)
        if phid:
            person.source_ids = sorted({*person.source_ids, f"aph:{phid}"})
            store.upsert_person(person)
            linked += 1
    return linked


def _record_attribution(store: SpeechStore, passages: list[Passage], source: str) -> int:
    unattributed = sum(1 for p in passages if store.person_for_source(p.source_speaker_id) is None)
    domain_metrics().unattributed_passages.add(unattributed, {"source": source})
    return unattributed


def ingest_federal_hansard(
    store: SpeechStore,
    client: HttpClient,
    start: date,
    end: date,
    chambers: Iterable[Chamber] = ("house_of_reps", "senate"),
    force: bool = False,
) -> int:
    total = 0
    for chamber in chambers:
        for sitting_date in openaustralia.list_sitting_dates(client, chamber, start, end):
            if not force and store.has_sitting("openaustralia", chamber, sitting_date):
                continue
            with tracer.start_as_current_span("ingest.federal_sitting") as span:
                span.set_attributes({"mp_profile.chamber": chamber, "mp_profile.date": sitting_date.isoformat()})
                passages = openaustralia.fetch_debates(client, chamber, sitting_date)
                missing = _record_attribution(store, passages, "openaustralia")
                store.add_passages(passages)
                store.record_sitting("openaustralia", "commonwealth", chamber, sitting_date, len(passages))
                span.set_attributes({"mp_profile.passages": len(passages), "mp_profile.unattributed": missing})
                domain_metrics().passages_ingested.add(len(passages), {"source": "openaustralia", "chamber": chamber})
                total += len(passages)
                logger.info("federal %s %s: %d passages", chamber, sitting_date, len(passages))
    return total


def ingest_victorian_hansard(
    store: SpeechStore, client: HttpClient, start: date, end: date, chamber: Chamber | None = None, force: bool = False
) -> int:
    total = 0
    for sitting in victoria.list_sittings(client, start, end, chamber):
        if not force and store.has_sitting("vic_hansard", sitting.chamber, sitting.date):
            continue
        with tracer.start_as_current_span("ingest.victorian_sitting") as span:
            span.set_attributes({"mp_profile.chamber": sitting.chamber, "mp_profile.date": sitting.date.isoformat()})
            passages = victoria.fetch_sitting(client, sitting)
            missing = _record_attribution(store, passages, "vic_hansard")
            store.add_passages(passages)
            store.record_sitting("vic_hansard", "victoria", sitting.chamber, sitting.date, len(passages))
            span.set_attributes({"mp_profile.passages": len(passages), "mp_profile.unattributed": missing})
            domain_metrics().passages_ingested.add(len(passages), {"source": "vic_hansard", "chamber": sitting.chamber})
            total += len(passages)
            logger.info("victoria %s %s: %d passages", sitting.chamber, sitting.date, len(passages))
    return total
