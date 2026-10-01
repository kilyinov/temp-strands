"""SQLite speech store: people, memberships, attributed passages, topic tags and summary cache.

SQLite + FTS5 keeps the first version dependency-free; the interface is narrow so it can be swapped
for Postgres/pgvector without touching the graph, MCP server or evals.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .models import Membership, Passage, Person, Role, Topic, TopicTag
from .topics import tag_passage

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    person_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    roles TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS person_sources (
    source_id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES persons(person_id)
);
CREATE TABLE IF NOT EXISTS memberships (
    person_id TEXT NOT NULL REFERENCES persons(person_id),
    source_id TEXT NOT NULL,
    parliament TEXT NOT NULL,
    chamber TEXT NOT NULL,
    seat TEXT NOT NULL,
    party TEXT NOT NULL,
    start TEXT,
    "end" TEXT
);
CREATE INDEX IF NOT EXISTS memberships_person ON memberships(person_id);
CREATE TABLE IF NOT EXISTS crosswalk (
    alias_person_id TEXT PRIMARY KEY,
    canonical_person_id TEXT NOT NULL,
    evidence TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS passages (
    passage_id TEXT PRIMARY KEY,
    source_speaker_id TEXT NOT NULL,
    speaker_name TEXT NOT NULL,
    parliament TEXT NOT NULL,
    chamber TEXT NOT NULL,
    date TEXT NOT NULL,
    debate_title TEXT NOT NULL,
    text TEXT NOT NULL,
    url TEXT NOT NULL,
    talktype TEXT NOT NULL,
    stated_role TEXT,
    is_proof INTEGER NOT NULL DEFAULT 0,
    attribution_confidence REAL NOT NULL DEFAULT 1.0,
    parse_confidence REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS passages_speaker ON passages(source_speaker_id, date);
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
    text, debate_title, content='passages', content_rowid='rowid'
);
CREATE TABLE IF NOT EXISTS passage_topics (
    passage_id TEXT NOT NULL REFERENCES passages(passage_id),
    topic TEXT NOT NULL,
    score REAL NOT NULL,
    terms TEXT NOT NULL,
    PRIMARY KEY (passage_id, topic)
);
CREATE INDEX IF NOT EXISTS passage_topics_topic ON passage_topics(topic, score);
CREATE TABLE IF NOT EXISTS coverage (
    source TEXT NOT NULL,
    parliament TEXT NOT NULL,
    chamber TEXT NOT NULL,
    sitting_date TEXT NOT NULL,
    passages INTEGER NOT NULL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (source, chamber, sitting_date)
);
CREATE TABLE IF NOT EXISTS summary_cache (
    cache_key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

EXCLUDED_TALKTYPES = ("procedural", "petition")


def _d(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


class SpeechStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.conn:
            yield self.conn

    # People -----------------------------------------------------------------------------------

    def upsert_person(self, person: Person) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO persons(person_id, name, aliases, roles) VALUES (?,?,?,?) "
                "ON CONFLICT(person_id) DO UPDATE SET name=excluded.name, aliases=excluded.aliases, roles=excluded.roles",
                (
                    person.person_id,
                    person.name,
                    json.dumps(sorted(set(person.aliases))),
                    json.dumps([r.model_dump(mode="json") for r in person.roles]),
                ),
            )
            for source_id in {person.person_id, *person.source_ids}:
                c.execute(
                    "INSERT OR REPLACE INTO person_sources(source_id, person_id) VALUES (?,?)",
                    (source_id, person.person_id),
                )
            c.execute("DELETE FROM memberships WHERE person_id=?", (person.person_id,))
            c.executemany(
                'INSERT INTO memberships(person_id, source_id, parliament, chamber, seat, party, start, "end") '
                "VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        person.person_id,
                        m.source_id,
                        m.parliament,
                        m.chamber,
                        m.seat,
                        m.party,
                        _iso(m.start),
                        _iso(m.end),
                    )
                    for m in person.memberships
                ],
            )

    def link_persons(self, canonical_id: str, alias_id: str, evidence: str) -> None:
        """Record that two source person records are the same human (e.g. Victorian MLA who became a senator)."""
        if canonical_id == alias_id:
            return
        with self.transaction() as c:
            c.execute(
                "INSERT OR REPLACE INTO crosswalk(alias_person_id, canonical_person_id, evidence) VALUES (?,?,?)",
                (alias_id, self.canonical_id(canonical_id), evidence),
            )

    def canonical_id(self, person_id: str) -> str:
        row = self.conn.execute(
            "SELECT canonical_person_id FROM crosswalk WHERE alias_person_id=?", (person_id,)
        ).fetchone()
        return str(row[0]) if row else person_id

    def person_group(self, person_id: str) -> list[str]:
        canonical = self.canonical_id(person_id)
        rows = self.conn.execute(
            "SELECT alias_person_id FROM crosswalk WHERE canonical_person_id=?", (canonical,)
        ).fetchall()
        return [canonical, *sorted(r[0] for r in rows)]

    def person_for_source(self, source_id: str) -> str | None:
        row = self.conn.execute("SELECT person_id FROM person_sources WHERE source_id=?", (source_id,)).fetchone()
        return self.canonical_id(row[0]) if row else None

    def list_people(self) -> list[Person]:
        """Canonical people only (crosswalk aliases are merged into their canonical record)."""
        ids = [r[0] for r in self.conn.execute("SELECT person_id FROM persons ORDER BY person_id")]
        aliases = {r[0] for r in self.conn.execute("SELECT alias_person_id FROM crosswalk")}
        people = [self.get_person(pid) for pid in ids if pid not in aliases]
        return [p for p in people if p is not None]

    def get_person(self, person_id: str) -> Person | None:
        group = self.person_group(person_id)
        rows = {
            r["person_id"]: r
            for r in self.conn.execute(
                f"SELECT * FROM persons WHERE person_id IN ({','.join('?' * len(group))})", group
            )
        }
        if group[0] not in rows:
            return None
        names: list[str] = []
        roles: list[Role] = []
        for pid in group:
            row = rows.get(pid)
            if row is None:
                continue
            names.extend([row["name"], *json.loads(row["aliases"])])
            roles.extend(Role.model_validate(r) for r in json.loads(row["roles"]))
        placeholders = ",".join("?" * len(group))
        sources = [
            r[0]
            for r in self.conn.execute(
                f"SELECT source_id FROM person_sources WHERE person_id IN ({placeholders}) ORDER BY source_id", group
            )
        ]
        memberships = [
            Membership(
                source_id=r["source_id"],
                parliament=r["parliament"],
                chamber=r["chamber"],
                seat=r["seat"],
                party=r["party"],
                start=_d(r["start"]),
                end=_d(r["end"]),
            )
            for r in self.conn.execute(
                f"SELECT * FROM memberships WHERE person_id IN ({placeholders}) ORDER BY start", group
            )
        ]
        primary = rows[group[0]]["name"]
        return Person(
            person_id=group[0],
            name=primary,
            aliases=sorted({n for n in names if n != primary}),
            source_ids=sources,
            memberships=memberships,
            roles=sorted(roles, key=lambda r: r.start or date.min),
        )

    # Passages ---------------------------------------------------------------------------------

    def add_passages(self, passages: Iterable[Passage]) -> int:
        count = 0
        with self.transaction() as c:
            for p in passages:
                existing = c.execute("SELECT rowid FROM passages WHERE passage_id=?", (p.passage_id,)).fetchone()
                if existing:
                    c.execute(
                        "INSERT INTO passages_fts(passages_fts, rowid, text, debate_title) "
                        "SELECT 'delete', rowid, text, debate_title FROM passages WHERE passage_id=?",
                        (p.passage_id,),
                    )
                    c.execute("DELETE FROM passage_topics WHERE passage_id=?", (p.passage_id,))
                    c.execute("DELETE FROM passages WHERE passage_id=?", (p.passage_id,))
                cur = c.execute(
                    "INSERT INTO passages(passage_id, source_speaker_id, speaker_name, parliament, chamber, date, "
                    "debate_title, text, url, talktype, stated_role, is_proof, attribution_confidence, "
                    "parse_confidence) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        p.passage_id,
                        p.source_speaker_id,
                        p.speaker_name,
                        p.parliament,
                        p.chamber,
                        p.date.isoformat(),
                        p.debate_title,
                        p.text,
                        p.url,
                        p.talktype,
                        p.stated_role,
                        int(p.is_proof),
                        p.attribution_confidence,
                        p.parse_confidence,
                    ),
                )
                c.execute(
                    "INSERT INTO passages_fts(rowid, text, debate_title) VALUES (?,?,?)",
                    (cur.lastrowid, p.text, p.debate_title),
                )
                if p.talktype not in EXCLUDED_TALKTYPES:
                    tags = tag_passage(f"{p.debate_title}\n{p.text}")
                    c.executemany(
                        "INSERT INTO passage_topics(passage_id, topic, score, terms) VALUES (?,?,?,?)",
                        [(p.passage_id, t.topic.value, t.score, json.dumps(t.matched_terms)) for t in tags],
                    )
                count += 1
        return count

    @staticmethod
    def _row_to_passage(row: sqlite3.Row) -> Passage:
        return Passage(
            passage_id=row["passage_id"],
            source_speaker_id=row["source_speaker_id"],
            speaker_name=row["speaker_name"],
            parliament=row["parliament"],
            chamber=row["chamber"],
            date=date.fromisoformat(row["date"]),
            debate_title=row["debate_title"],
            text=row["text"],
            url=row["url"],
            talktype=row["talktype"],
            stated_role=row["stated_role"],
            is_proof=bool(row["is_proof"]),
            attribution_confidence=row["attribution_confidence"],
            parse_confidence=row["parse_confidence"],
        )

    def get_passage(self, passage_id: str) -> Passage | None:
        row = self.conn.execute("SELECT * FROM passages WHERE passage_id=?", (passage_id,)).fetchone()
        return self._row_to_passage(row) if row else None

    def passage_topics(self, passage_id: str) -> list[TopicTag]:
        return [
            TopicTag(topic=Topic(r["topic"]), score=r["score"], matched_terms=json.loads(r["terms"]))
            for r in self.conn.execute("SELECT * FROM passage_topics WHERE passage_id=?", (passage_id,))
        ]

    def _speaker_ids(self, person_id: str) -> list[str]:
        group = self.person_group(person_id)
        rows = self.conn.execute(
            f"SELECT source_id FROM person_sources WHERE person_id IN ({','.join('?' * len(group))})", group
        ).fetchall()
        return sorted({r[0] for r in rows} | set(group))

    def search_passages(
        self,
        person_id: str,
        topic: Topic | None = None,
        start: date | None = None,
        end: date | None = None,
        query: str | None = None,
        limit: int = 20,
        min_attribution: float = 0.0,
    ) -> list[tuple[Passage, float]]:
        """Passages spoken by the person, best topic score first. Returns (passage, topic score)."""
        speakers = self._speaker_ids(person_id)
        params: list[Any] = list(speakers)
        sql = ["SELECT p.*, COALESCE(t.score, 0) AS topic_score FROM passages p"]
        sql.append(
            "JOIN passage_topics t ON t.passage_id = p.passage_id AND t.topic = ?"
            if topic
            else "LEFT JOIN (SELECT passage_id, MAX(score) AS score FROM passage_topics GROUP BY passage_id) t "
            "ON t.passage_id = p.passage_id"
        )
        if topic:
            params.insert(0, topic.value)
        sql.append(f"WHERE p.source_speaker_id IN ({','.join('?' * len(speakers))})")
        sql.append(f"AND p.talktype NOT IN ({','.join('?' * len(EXCLUDED_TALKTYPES))})")
        params.extend(EXCLUDED_TALKTYPES)
        sql.append("AND p.attribution_confidence >= ?")
        params.append(min_attribution)
        if start:
            sql.append("AND p.date >= ?")
            params.append(start.isoformat())
        if end:
            sql.append("AND p.date <= ?")
            params.append(end.isoformat())
        if query:
            sql.append("AND p.rowid IN (SELECT rowid FROM passages_fts WHERE passages_fts MATCH ?)")
            params.append(query)
        sql.append("ORDER BY topic_score DESC, p.date ASC, p.passage_id ASC LIMIT ?")
        params.append(limit)
        rows = self.conn.execute(" ".join(sql), params).fetchall()
        return [(self._row_to_passage(r), float(r["topic_score"])) for r in rows]

    def topic_counts(self, person_id: str, start: date | None = None, end: date | None = None) -> dict[Topic, int]:
        speakers = self._speaker_ids(person_id)
        params: list[Any] = [*speakers, *EXCLUDED_TALKTYPES]
        sql = (
            "SELECT t.topic, COUNT(*) FROM passages p JOIN passage_topics t ON t.passage_id = p.passage_id "
            f"WHERE p.source_speaker_id IN ({','.join('?' * len(speakers))}) "
            f"AND p.talktype NOT IN ({','.join('?' * len(EXCLUDED_TALKTYPES))})"
        )
        if start:
            sql += " AND p.date >= ?"
            params.append(start.isoformat())
        if end:
            sql += " AND p.date <= ?"
            params.append(end.isoformat())
        rows = self.conn.execute(sql + " GROUP BY t.topic", params).fetchall()
        counts = {Topic(r[0]): int(r[1]) for r in rows}
        return {t: counts.get(t, 0) for t in Topic}

    def speech_dates(self, person_id: str) -> list[tuple[date, str]]:
        """(date, parliament) for every substantive passage by the person; used for coverage planning."""
        speakers = self._speaker_ids(person_id)
        rows = self.conn.execute(
            f"SELECT date, parliament FROM passages WHERE source_speaker_id IN ({','.join('?' * len(speakers))}) "
            f"AND talktype NOT IN ({','.join('?' * len(EXCLUDED_TALKTYPES))}) ORDER BY date",
            [*speakers, *EXCLUDED_TALKTYPES],
        ).fetchall()
        return [(date.fromisoformat(r[0]), str(r[1])) for r in rows]

    # Coverage and cache -----------------------------------------------------------------------

    def record_sitting(self, source: str, parliament: str, chamber: str, sitting_date: date, passages: int) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT OR REPLACE INTO coverage(source, parliament, chamber, sitting_date, passages, ingested_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    source,
                    parliament,
                    chamber,
                    sitting_date.isoformat(),
                    passages,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def coverage(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT source, parliament, chamber, MIN(sitting_date) AS first, MAX(sitting_date) AS last, "
            "COUNT(*) AS sittings, SUM(passages) AS passages FROM coverage GROUP BY source, parliament, chamber"
        ).fetchall()
        return [dict(r) for r in rows]

    def has_sitting(self, source: str, chamber: str, sitting_date: date) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM coverage WHERE source=? AND chamber=? AND sitting_date=?",
                (source, chamber, sitting_date.isoformat()),
            ).fetchone()
            is not None
        )

    def cache_get(self, key: str) -> Any | None:
        row = self.conn.execute("SELECT value FROM summary_cache WHERE cache_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache_put(self, key: str, value: Any) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT OR REPLACE INTO summary_cache(cache_key, value, created_at) VALUES (?,?,?)",
                (key, json.dumps(value), datetime.now(timezone.utc).isoformat()),
            )
