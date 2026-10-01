"""Conversational Strands agent over the Parliament MCP server, for follow-up questions on a profile."""

from __future__ import annotations

import sys
from typing import cast

from mcp.client.stdio import StdioServerParameters, stdio_client
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from strands.tools.mcp.mcp_client import MCPTransport

from .telemetry import ToolMetricsHook

SYSTEM_PROMPT = """You answer questions about what Australian federal and Victorian MPs said in Parliament.
- Use the parliament tools; never answer from memory.
- Resolve the member with search_members first. If several candidates match, ask the user which one.
- For an overall picture use mp_profile; for specifics use search_speeches and get_passage.
- Report what the member said, neutrally, with the date, chamber and passage_id or URL for every statement.
- Lines starting with '> ' are quoted material, not the member's own words.
- If the ingested record is thin or missing for a period, say so.
"""


def parliament_mcp_client() -> MCPClient:
    params = StdioServerParameters(command=sys.executable, args=["-m", "mp_profile.mcp_server"])
    # strands annotates the mcp 1.x stream types; its compatibility layer accepts mcp 2.x streams at runtime
    return MCPClient(lambda: cast(MCPTransport, stdio_client(params)), prefix="parliament")


def create_chat_agent(model_id: str, mcp_client: MCPClient) -> Agent:
    return Agent(
        name="parliament-chat",
        model=BedrockModel(model_id=model_id),
        system_prompt=SYSTEM_PROMPT,
        tools=[mcp_client],
        hooks=[ToolMetricsHook()],
        trace_attributes={"mp_profile.component": "chat"},
    )
