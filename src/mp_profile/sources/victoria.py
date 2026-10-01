"""Parliament of Victoria Hansard (progressive HTML, Dec 2018 onwards) and member registry.

There is no documented public API. The website's own JSON search endpoints are used for listing:
  /api/search/debate?page=N&pageSize=M[&hansard-house=10|20]  -> sitting days (newest first)
  /api/search/members?member-status=current|former&page=N&pageSize=M
Each sitting day links to a table of contents of debate items; each item page holds
<div data-js-hook="speech-wrapper" id="{memberId}[X{n}]"> blocks with the speaker anchor.
Pre-2018 Hansard is PDF only (OCR needed before 1990s) and is not ingested by this adapter.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from lxml import html as lxml_html
from lxml.html import HtmlElement

from ..models import Chamber, Membership, Passage, Person, TalkType
from ..text import normalize_whitespace
from .http import HttpClient

BASE = "https://www.parliament.vic.gov.au"
HOUSES: dict[str, Chamber] = {"10": "legislative_assembly", "20": "legislative_council"}
HOUSE_CODES: dict[Chamber, str] = {v: k for k, v in HOUSES.items()}
HOUSE_NAMES: dict[str, Chamber] = {
    "Legislative Assembly": "legislative_assembly",
    "Legislative Council": "legislative_council",
}
_PHOTO_ID = re.compile(r"member-photo/(\d+)\.")
_ITEM_LINK = re.compile(r"/parliamentary-activity/hansard/hansard-details/(HANSARD-\d+-\d+)")
_PRESIDING = re.compile(r"^the (deputy |acting )?(speaker|president|chair)", re.IGNORECASE)
_RECORD_NOTE = re.compile(
    r"^(motion agreed to|question agreed to|house adjourned|council adjourned|sitting suspended|debate adjourned|"
    r"bill read a|the (deputy |acting )?(speaker|president|chair) took the chair)",
    re.IGNORECASE,
)
_BY_LABEL = re.compile(r"^(?P<name>[^(:]+?)\s*(\((?P<detail>[^)]*)\))?\s*(\(\d{1,2}:\d{2}\))?\s*:?\s*$")
QUOTED_CLASSES = ("Indent1", "Indent2", "Quote", "List")


def person_id_for(member_id: str) -> str:
    return f"vic:member/{member_id}"


@dataclass(frozen=True)
class Sitting:
    hansard_id: str
    date: date
    chamber: Chamber
    status: str
    toc_url: str


def parse_member_hits(payload: dict[str, Any], former: bool) -> list[Person]:
    people = []
    for hit in payload.get("result", {}).get("hits", []):
        photo = (hit.get("image") or {}).get("src") or ""
        match = _PHOTO_ID.search(photo)
        if not match:
            continue
        member_id = match.group(1)
        details = {m.get("title"): m.get("details") or [] for m in hit.get("memberships", [])}
        house_name = (details.get("House") or [""])[0]
        chamber = HOUSE_NAMES.get(house_name) or HOUSES.get(str(hit.get("house")), "legislative_assembly")
        pid = person_id_for(member_id)
        people.append(
            Person(
                person_id=pid,
                name=normalize_whitespace(hit.get("title", "")),
                source_ids=[pid],
                memberships=[
                    Membership(
                        source_id=pid,
                        parliament="victoria",
                        chamber=chamber,
                        seat=(details.get("Member for") or ["Unknown"])[0],
                        party=(details.get("Party") or ["Unknown"])[0],
                        start=None,
                        end=None,
                    )
                ],
            )
        )
    return people


def fetch_members(client: HttpClient, page_size: int = 100) -> list[Person]:
    people: list[Person] = []
    for status in ("current", "former"):
        page = 1
        while True:
            body = client.get_text(
                f"{BASE}/api/search/members",
                params={"member-status": status, "page": page, "pageSize": page_size},
                use_cache=False,
            )
            payload = json.loads(body)
            batch = parse_member_hits(payload, former=status == "former")
            people.extend(batch)
            total = payload.get("result", {}).get("totalMatching", 0)
            if page * page_size >= total or not payload.get("result", {}).get("hits"):
                break
            page += 1
    return people


def parse_sitting_hits(payload: dict[str, Any]) -> list[Sitting]:
    sittings = []
    for hit in payload.get("result", {}).get("hits", []):
        chamber = HOUSES.get(str(hit.get("house")))
        href = (hit.get("onlineButton") or {}).get("href")
        if not chamber or not href or not hit.get("date1"):
            continue
        sittings.append(
            Sitting(
                hansard_id=hit["id"],
                date=date.fromisoformat(hit["date1"][:10]),
                chamber=chamber,
                status=str(hit.get("status") or ""),
                toc_url=BASE + href,
            )
        )
    return sittings


def list_sittings(client: HttpClient, start: date, end: date, chamber: Chamber | None = None) -> list[Sitting]:
    found: list[Sitting] = []
    page = 1
    while True:
        params: dict[str, Any] = {"page": page, "pageSize": 50}
        if chamber:
            params["hansard-house"] = HOUSE_CODES[chamber]
        payload = json.loads(client.get_text(f"{BASE}/api/search/debate", params=params, use_cache=False))
        batch = parse_sitting_hits(payload)
        if not batch:
            break
        found.extend(s for s in batch if start <= s.date <= end)
        if min(s.date for s in batch) < start:
            break
        page += 1
    return sorted(found, key=lambda s: (s.date, s.chamber))


def parse_toc(html: str) -> list[str]:
    """Unique debate-item URLs for a sitting day, in order."""
    seen: dict[str, None] = {}
    for item in _ITEM_LINK.findall(html):
        seen.setdefault(item, None)
    return [f"{BASE}/parliamentary-activity/hansard/hansard-details/{item}" for item in seen]


def _split_label(label: str) -> tuple[str, str | None]:
    match = _BY_LABEL.match(normalize_whitespace(label))
    if not match:
        return normalize_whitespace(label.rstrip(":")), None
    detail = match.group("detail")
    role = None
    if detail and "–" in detail:
        role = detail.split("–", 1)[1].strip()
    elif detail and " - " in detail:
        role = detail.split(" - ", 1)[1].strip()
    return match.group("name").strip(), role


def _find(node: HtmlElement, expr: str) -> list[HtmlElement]:
    return [el for el in node.xpath(expr) if isinstance(el, HtmlElement)]


def parse_item(html: str, url: str, chamber: Chamber, sitting_date: date, is_proof: bool) -> list[Passage]:
    doc = lxml_html.fromstring(html)
    title_el = _find(doc, "//h2[contains(@class,'text-2xl')]")
    category_el = _find(doc, "//span[contains(@class,'text-uppercase')]")
    subject = normalize_whitespace(title_el[0].text_content()) if title_el else ""
    category = normalize_whitespace(category_el[0].text_content()) if category_el else ""
    debate_title = " — ".join(t for t in (category, subject) if t) or "Untitled debate"
    item_id = url.rstrip("/").rsplit("/", 1)[-1]
    passages = []
    last_kind: dict[str, TalkType] = {}
    last_role: dict[str, str] = {}
    for wrapper in _find(doc, "//div[@data-js-hook='speech-wrapper']"):
        wrapper_id = wrapper.get("id", "")
        member_id = wrapper_id.split("X", 1)[0]
        if not member_id.isdigit():
            continue
        anchor = _find(wrapper, ".//span[contains(@class,'HpsBy')]//a[@href]")
        anchor_id = anchor[0].get("href", "").rstrip("/").rsplit("/", 1)[-1] if anchor else None
        by = _find(wrapper, ".//span[contains(@class,'HpsBy')]")
        label = normalize_whitespace(by[0].text_content()) if by else ""
        speaker_name, stated_role = _split_label(label)
        if stated_role:
            last_role[member_id] = stated_role
        elif "(" not in label:
            stated_role = last_role.get(member_id)
        for el in _find(wrapper, ".//span[contains(@class,'HpsBy')] | .//span[@data-js-hook='speaker-info']"):
            el.drop_tree()
        paragraphs = []
        for p in _find(wrapper, ".//p"):
            if _find(p, ".//span[contains(@class,'HpsTerm')]"):
                continue
            text = normalize_whitespace(p.text_content())
            if not text or _RECORD_NOTE.match(text):
                continue
            classes = " ".join(s.get("class", "") for s in _find(p, "./span"))
            paragraphs.append(f"> {text}" if any(c in classes for c in QUOTED_CLASSES) else text)
        if not paragraphs:
            continue
        talktype: TalkType
        if _PRESIDING.match(speaker_name):
            talktype = "procedural"
        elif category.lower().startswith("petitions"):
            talktype = "petition"
        elif category.lower().startswith("questions"):
            talktype = "answer" if stated_role else last_kind.get(member_id, "question")
            last_kind[member_id] = talktype
        else:
            talktype = "speech" if "(" in label else "continuation"
        if anchor_id == member_id:
            attribution = 1.0
        elif anchor_id is None:
            attribution = 0.9
        else:
            attribution = 0.5
        passages.append(
            Passage(
                passage_id=f"vic:{item_id}#{wrapper_id}",
                source_speaker_id=person_id_for(member_id),
                speaker_name=speaker_name,
                parliament="victoria",
                chamber=chamber,
                date=sitting_date,
                debate_title=debate_title,
                text="\n".join(paragraphs),
                url=f"{url}#{wrapper_id}",
                talktype=talktype,
                stated_role=stated_role,
                is_proof=is_proof,
                attribution_confidence=attribution,
            )
        )
    return passages


def fetch_sitting(client: HttpClient, sitting: Sitting) -> list[Passage]:
    toc = client.get_text(sitting.toc_url)
    passages: list[Passage] = []
    is_proof = sitting.status.lower() == "proof"
    for item_url in parse_toc(toc):
        passages.extend(parse_item(client.get_text(item_url), item_url, sitting.chamber, sitting.date, is_proof))
    return passages
