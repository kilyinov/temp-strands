"""Parliamentary Handbook API (handbookapi.aph.gov.au): authoritative federal identity (PHID).

Used to attach the official PHID to OpenAustralia person records, which gives a stable federal key
for the federal <-> Victorian crosswalk and for joining pre-2006 Hansard (keyed by PHID in ParlInfo).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

from ..models import Person
from ..text import name_tokens
from .http import HttpClient

BASE = "https://handbookapi.aph.gov.au/api/individuals"


@dataclass(frozen=True)
class HandbookRecord:
    phid: str
    display_name: str
    service_start: date | None


def _parse(record: dict[str, Any]) -> HandbookRecord:
    start = record.get("ServiceHistory_Start")
    return HandbookRecord(
        phid=str(record["PHID"]),
        display_name=str(record.get("DisplayName", "")),
        service_start=date.fromisoformat(start[:10]) if start else None,
    )


def fetch_records(client: HttpClient, page_size: int = 200) -> list[HandbookRecord]:
    records: list[HandbookRecord] = []
    skip = 0
    while True:
        body = client.get_text(
            BASE, params={"$top": page_size, "$skip": skip, "$select": "PHID,DisplayName,ServiceHistory_Start"}
        )
        page = json.loads(body).get("value", [])
        records.extend(_parse(r) for r in page)
        if len(page) < page_size:
            return records
        skip += page_size


def match_phid(person: Person, records: list[HandbookRecord], max_start_gap_days: int = 400) -> str | None:
    """Match on surname + a given-name token + first federal service start within ~a year. Unique match only."""
    tokens = name_tokens(person.name)
    if not tokens:
        return None
    starts = [m.start for m in person.memberships if m.parliament == "commonwealth" and m.start]
    first_start = min(starts) if starts else None
    hits = []
    for r in records:
        # DisplayName looks like "PLIBERSEK, the Hon. Tanya Joan"
        surname_part, _, given_part = r.display_name.partition(",")
        if name_tokens(surname_part)[-1:] != tokens[-1:]:
            continue
        if not set(name_tokens(given_part)) & set(tokens[:-1]):
            continue
        if first_start and r.service_start and abs((r.service_start - first_start).days) > max_start_gap_days:
            continue
        hits.append(r.phid)
    return hits[0] if len(hits) == 1 else None
