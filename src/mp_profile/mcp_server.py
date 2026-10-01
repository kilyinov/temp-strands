"""Parliament MCP server over the local speech store (MCP Python SDK v2 `MCPServer`, stdio transport).

Run: `mp-profile mcp` (or `python -m mp_profile.mcp_server`).
"""

from __future__ import annotations

import time
from datetime import date

from mcp.server.mcpserver import MCPServer

from .config import Settings
from .models import Parliament, Topic
from .resolve import score_people
from .service import build_profile
from .store import SpeechStore
from .telemetry import domain_metrics, setup_telemetry

MAX_TEXT = 2000

INSTRUCTIONS = (
    "Hansard speeches of Australian federal (House, Senate) and Victorian (Assembly, Council) members. "
    "Resolve a member with search_members first, then use person_id with the other tools. "
    "Lines starting with '> ' in passage text are material the member was quoting, not their own words."
)


class ParliamentTools:
    def __init__(self, store: SpeechStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    def _timed(self, tool: str, started: float) -> None:
        domain_metrics().tool_duration.record(time.perf_counter() - started, {"tool": tool, "agent": "mcp"})

    def search_members(self, name: str, parliament: Parliament | None = None) -> list[dict[str, object]]:
        """Find Commonwealth or Victorian members by name. Returns candidates with person_id, score and career summary."""
        started = time.perf_counter()
        try:
            return [c.model_dump(mode="json") for c in score_people(self.store, name, parliament)[:10]]
        finally:
            self._timed("search_members", started)

    def get_member(self, person_id: str) -> dict[str, object]:
        """Memberships (parliament, chamber, seat, party, dates) and offices held for a person_id."""
        person = self.store.get_person(person_id)
        if person is None:
            raise ValueError(f"unknown person_id {person_id}")
        return person.model_dump(mode="json")

    def search_speeches(
        self,
        person_id: str,
        topic: Topic | None = None,
        start: date | None = None,
        end: date | None = None,
        query: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, object]]:
        """Speech passages by a member, optionally filtered by topic (housing, healthcare, economy, education),
        date range and full-text query (SQLite FTS5 syntax). Best topic matches first."""
        started = time.perf_counter()
        try:
            results = self.store.search_passages(person_id, topic, start, end, query, limit=min(limit, 50))
            out = []
            for passage, score in results:
                record = passage.model_dump(mode="json")
                record["text"] = passage.text[:MAX_TEXT]
                record["topic_score"] = score
                out.append(record)
            return out
        finally:
            self._timed("search_speeches", started)

    def get_passage(self, passage_id: str) -> dict[str, object]:
        """Full text, source URL and topic tags for one Hansard passage."""
        passage = self.store.get_passage(passage_id)
        if passage is None:
            raise ValueError(f"unknown passage_id {passage_id}")
        record = passage.model_dump(mode="json")
        record["topics"] = [t.model_dump(mode="json") for t in self.store.passage_topics(passage_id)]
        return record

    def coverage(self) -> list[dict[str, object]]:
        """Which sources, chambers and date ranges have been ingested."""
        return self.store.coverage()

    async def mp_profile(self, name_or_person_id: str, parliament: Parliament | None = None) -> dict[str, object]:
        """Whole-career, cited summary of what a member said about housing, healthcare, economy and education.
        Returns a disambiguation list instead if the name matches several members."""
        result = await build_profile(self.store, name_or_person_id, self.settings, parliament=parliament)
        return result.model_dump(mode="json")


def create_server(store: SpeechStore, settings: Settings) -> MCPServer:
    tools = ParliamentTools(store, settings)
    server = MCPServer(name="parliament-au", instructions=INSTRUCTIONS)
    for fn in (
        tools.search_members,
        tools.get_member,
        tools.search_speeches,
        tools.get_passage,
        tools.coverage,
        tools.mp_profile,
    ):
        server.tool()(fn)
    return server


def main() -> None:
    settings = Settings()
    setup_telemetry("mp-profile-mcp")
    create_server(SpeechStore(settings.db_path), settings).run(transport="stdio")


if __name__ == "__main__":
    main()
