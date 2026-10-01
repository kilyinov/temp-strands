"""Commonwealth Hansard and member registry from the OpenAustralia XML mirror (2006 onwards).

https://data.openaustralia.org.au/ publishes the parsed Hansard used by openaustralia.org.au:
  members/people.xml, representatives.xml, senators.xml, ministers.xml
  scrapedxml/{representatives,senate}_debates/YYYY-MM-DD.xml
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date

from lxml import etree

from ..models import Chamber, Membership, Passage, Person, Role, TalkType
from ..text import normalize_whitespace
from .http import HttpClient

BASE = "https://data.openaustralia.org.au"
DEBATE_DIRS: dict[Chamber, str] = {"house_of_reps": "representatives_debates", "senate": "senate_debates"}
MEMBER_FILES: dict[Chamber, str] = {"house_of_reps": "representatives.xml", "senate": "senators.xml"}
OPEN_ENDED = "9999-12-31"
_PRESIDING = re.compile(r"^(the )?(deputy |acting )?(speaker|president|chair|temporary chairman)\b", re.IGNORECASE)
_DATE_HREF = re.compile(r'href="(\d{4}-\d{2}-\d{2})\.xml"')


def _date(value: str | None) -> date | None:
    if not value or value == OPEN_ENDED:
        return None
    return date.fromisoformat(value)


def person_id_for(oa_person_id: str) -> str:
    return "oa:" + oa_person_id.removeprefix("uk.org.publicwhip/")


def parse_people(
    people_xml: bytes, member_xml: dict[Chamber, bytes], ministers_xml: bytes | None = None
) -> list[Person]:
    office_to_person: dict[str, str] = {}
    latest_name: dict[str, str] = {}
    for el in etree.fromstring(people_xml).iter("person"):
        pid = el.get("id", "")
        latest_name[pid] = el.get("latestname", "")
        for office in el.iter("office"):
            office_to_person[office.get("id", "")] = pid

    memberships: dict[str, list[Membership]] = defaultdict(list)
    names: dict[str, set[str]] = defaultdict(set)
    for chamber, xml in member_xml.items():
        for el in etree.fromstring(xml).iter("member"):
            office_id = el.get("id", "")
            owner = office_to_person.get(office_id)
            if not owner:
                continue
            memberships[owner].append(
                Membership(
                    source_id=office_id,
                    parliament="commonwealth",
                    chamber=chamber,
                    seat=el.get("division", ""),
                    party=el.get("party", ""),
                    start=_date(el.get("fromdate")),
                    end=_date(el.get("todate")),
                )
            )
            names[owner].add(normalize_whitespace(f"{el.get('firstname', '')} {el.get('lastname', '')}"))

    roles: dict[str, list[Role]] = defaultdict(list)
    if ministers_xml:
        for el in etree.fromstring(ministers_xml).iter("moffice"):
            holder = office_to_person.get(el.get("matchid", ""))
            if not holder:
                continue
            title = el.get("position", "")
            roles[holder].append(
                Role(
                    title=title,
                    kind="shadow" if "shadow" in title.lower() else "ministerial",
                    start=_date(el.get("fromdate")),
                    end=_date(el.get("todate")),
                )
            )
            names[holder].add(normalize_whitespace(el.get("name", "")))

    people = []
    for pid, mships in memberships.items():
        office_ids = sorted({m.source_id for m in mships})
        people.append(
            Person(
                person_id=person_id_for(pid),
                name=latest_name.get(pid) or sorted(names[pid])[0],
                aliases=sorted(n for n in names[pid] if n),
                source_ids=office_ids,
                memberships=sorted(mships, key=lambda m: m.start or date.min),
                roles=roles.get(pid, []),
            )
        )
    return people


def _text(el: etree._Element) -> str:
    return normalize_whitespace("".join(str(t) for t in el.itertext()))


def _paragraphs(speech: etree._Element) -> list[str]:
    paras = []
    for p in speech.iter("p"):
        text = _text(p)
        if not text:
            continue
        css = p.get("class", "")
        paras.append(f"> {text}" if "italic" in css or "indent" in css else text)
    return paras


def parse_debates(xml: bytes, chamber: Chamber, sitting_date: date) -> list[Passage]:
    root = etree.fromstring(xml)
    major = minor = ""
    passages = []
    for el in root:
        tag = el.tag
        if tag == "major-heading":
            major, minor = _text(el), ""
        elif tag == "minor-heading":
            minor = _text(el)
        elif tag == "speech":
            talktype = el.get("talktype", "speech")
            speaker = el.get("speakerid", "")
            if talktype == "interjection" or not speaker or speaker == "unknown":
                continue
            text = "\n".join(_paragraphs(el))
            if not text:
                continue
            speaker_name = el.get("speakername", "")
            kind: TalkType = (
                "procedural"
                if _PRESIDING.match(speaker_name)
                else ("continuation" if talktype == "continuation" else "speech")
            )
            passages.append(
                Passage(
                    passage_id=el.get("id", ""),
                    source_speaker_id=speaker,
                    speaker_name=speaker_name,
                    parliament="commonwealth",
                    chamber=chamber,
                    date=sitting_date,
                    debate_title=" — ".join(t for t in (major, minor) if t) or "Untitled debate",
                    text=text,
                    url=el.get("url", ""),
                    talktype=kind,
                )
            )
    return passages


def list_sitting_dates(client: HttpClient, chamber: Chamber, start: date, end: date) -> list[date]:
    listing = client.get_text(f"{BASE}/scrapedxml/{DEBATE_DIRS[chamber]}/", use_cache=False)
    dates = sorted({date.fromisoformat(d) for d in _DATE_HREF.findall(listing)})
    return [d for d in dates if start <= d <= end]


def fetch_debates(client: HttpClient, chamber: Chamber, sitting_date: date) -> list[Passage]:
    xml = client.get_bytes(f"{BASE}/scrapedxml/{DEBATE_DIRS[chamber]}/{sitting_date.isoformat()}.xml")
    return parse_debates(xml, chamber, sitting_date)


def fetch_people(client: HttpClient) -> list[Person]:
    people_xml = client.get_bytes(f"{BASE}/members/people.xml")
    member_xml = {ch: client.get_bytes(f"{BASE}/members/{name}") for ch, name in MEMBER_FILES.items()}
    ministers_xml = client.get_bytes(f"{BASE}/members/ministers.xml")
    return parse_people(people_xml, member_xml, ministers_xml)
