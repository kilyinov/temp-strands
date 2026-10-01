"""Load the bundled fictional sample dataset through the real parsers (no network)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from . import crosswalk
from .ingest import ingest_people
from .sources import openaustralia, victoria
from .store import SpeechStore

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def load_sample(store: SpeechStore, sample_dir: Path = SAMPLE_DIR) -> None:
    people = openaustralia.parse_people(
        (sample_dir / "oa_people.xml").read_bytes(),
        {
            "house_of_reps": (sample_dir / "oa_representatives.xml").read_bytes(),
            "senate": (sample_dir / "oa_senators.xml").read_bytes(),
        },
        (sample_dir / "oa_ministers.xml").read_bytes(),
    )
    ingest_people(store, people)
    members = json.loads((sample_dir / "vic_members_current.json").read_text())
    ingest_people(store, victoria.parse_member_hits(members, former=False))

    for path in sorted(sample_dir.glob("oa_representatives_debates_*.xml")):
        sitting_date = date.fromisoformat(path.stem.rsplit("_", 1)[-1])
        passages = openaustralia.parse_debates(path.read_bytes(), "house_of_reps", sitting_date)
        store.add_passages(passages)
        store.record_sitting("openaustralia", "commonwealth", "house_of_reps", sitting_date, len(passages))

    for sitting in victoria.parse_sitting_hits(json.loads((sample_dir / "vic_debate_search.json").read_text())):
        total = 0
        for item_url in victoria.parse_toc((sample_dir / "vic_toc.html").read_text()):
            item_id = item_url.rsplit("/", 1)[-1]
            html = (sample_dir / f"vic_item_{item_id}.html").read_text()
            passages = victoria.parse_item(html, item_url, sitting.chamber, sitting.date, sitting.status == "Proof")
            store.add_passages(passages)
            total += len(passages)
        store.record_sitting("vic_hansard", "victoria", sitting.chamber, sitting.date, total)

    crosswalk.apply(store, sample_dir / "crosswalk.csv")
