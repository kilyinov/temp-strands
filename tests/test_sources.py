import json
from datetime import date

from mp_profile.sample import SAMPLE_DIR
from mp_profile.sources import openaustralia, victoria


def _vic_item(item_id: str) -> str:
    return (SAMPLE_DIR / f"vic_item_{item_id}.html").read_text()


def test_openaustralia_people_and_roles() -> None:
    people = openaustralia.parse_people(
        (SAMPLE_DIR / "oa_people.xml").read_bytes(),
        {
            "house_of_reps": (SAMPLE_DIR / "oa_representatives.xml").read_bytes(),
            "senate": (SAMPLE_DIR / "oa_senators.xml").read_bytes(),
        },
        (SAMPLE_DIR / "oa_ministers.xml").read_bytes(),
    )
    alex = next(p for p in people if p.person_id == "oa:person/20001")
    assert alex.memberships[0].seat == "Riverbend"
    assert alex.memberships[0].start == date(2007, 11, 24)
    assert any("Housing" in r.title for r in alex.roles)


def test_openaustralia_debates_exclude_interjections_and_mark_quotes() -> None:
    path = SAMPLE_DIR / "oa_representatives_debates_2012-03-20.xml"
    passages = openaustralia.parse_debates(path.read_bytes(), "house_of_reps", date(2012, 3, 20))
    assert passages
    assert all("It will push up rents" not in p.text for p in passages)
    assert any(line.startswith("> ") for p in passages for line in p.text.splitlines())
    assert {p.talktype for p in passages} <= {"speech", "continuation", "procedural"}
    assert all(p.parliament == "commonwealth" and p.date == date(2012, 3, 20) for p in passages)
    assert all(p.debate_title for p in passages)


def test_victoria_item_parsing() -> None:
    url = "https://www.parliament.vic.gov.au/parliamentary-activity/hansard/hansard-details/HANSARD-2000000001-10002"
    passages = victoria.parse_item(
        _vic_item("HANSARD-2000000001-10002"), url, "legislative_assembly", date(2019, 8, 14), False
    )
    by_id = {p.passage_id.rsplit("#", 1)[-1]: p for p in passages}
    assert set(by_id) == {"9001", "9003", "9010", "9001X1"}
    assert by_id["9010"].talktype == "procedural"
    assert by_id["9001"].stated_role == "Minister for Housing"
    assert by_id["9001X1"].talktype == "continuation"
    assert by_id["9001X1"].stated_role == "Minister for Housing"
    assert "> The housing crisis is a myth" in by_id["9001X1"].text
    assert "interjecting" not in by_id["9003"].text
    assert "Motion agreed to" not in by_id["9001X1"].text
    assert by_id["9001"].source_speaker_id == "vic:member/9001"
    assert by_id["9001"].debate_title == "Bills — Rental Reform Bill 2019"


def test_victoria_question_time_classification() -> None:
    url = "https://www.parliament.vic.gov.au/parliamentary-activity/hansard/hansard-details/HANSARD-2000000001-10003"
    passages = victoria.parse_item(
        _vic_item("HANSARD-2000000001-10003"), url, "legislative_assembly", date(2019, 8, 14), True
    )
    kinds = [(p.source_speaker_id, p.talktype) for p in passages]
    assert kinds == [("vic:member/9003", "question"), ("vic:member/9004", "answer"), ("vic:member/9003", "question")]
    assert all(p.is_proof for p in passages)


def test_victoria_toc_and_members() -> None:
    items = victoria.parse_toc((SAMPLE_DIR / "vic_toc.html").read_text())
    assert [i.rsplit("/", 1)[-1] for i in items] == ["HANSARD-2000000001-10002", "HANSARD-2000000001-10003"]
    people = victoria.parse_member_hits(json.loads((SAMPLE_DIR / "vic_members_current.json").read_text()), former=False)
    sam = next(p for p in people if p.person_id == "vic:member/9002")
    assert sam.memberships[0].chamber == "legislative_council"
