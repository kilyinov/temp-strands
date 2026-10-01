import json

from mcp.client.client import Client

from mp_profile.config import Settings
from mp_profile.mcp_server import create_server
from mp_profile.store import SpeechStore

from .conftest import ALEX


async def test_mcp_tools_end_to_end(store: SpeechStore, settings: Settings) -> None:
    server = create_server(store, settings)
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert names == {"search_members", "get_member", "search_speeches", "get_passage", "coverage", "mp_profile"}

        found = await client.call_tool("search_members", {"name": "Alex Morgan"})
        assert not found.is_error
        candidates = json.loads(found.content[0].text)  # type: ignore[union-attr]
        assert (candidates[0] if isinstance(candidates, list) else candidates)["person_id"] == ALEX

        speeches = await client.call_tool("search_speeches", {"person_id": ALEX, "topic": "housing", "limit": 2})
        assert not speeches.is_error
        first = json.loads(speeches.content[0].text)  # type: ignore[union-attr]
        passage_id = (first[0] if isinstance(first, list) else first)["passage_id"]

        passage = await client.call_tool("get_passage", {"passage_id": passage_id})
        assert "housing" in passage.content[0].text  # type: ignore[union-attr]

        missing = await client.call_tool("get_passage", {"passage_id": "nope"})
        assert missing.is_error

        profile = await client.call_tool("mp_profile", {"name_or_person_id": "Taylor"})
        assert "needs_disambiguation" in profile.content[0].text  # type: ignore[union-attr]
