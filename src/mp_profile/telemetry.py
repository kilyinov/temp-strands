"""OpenTelemetry setup, domain metrics and Strands hooks.

Strands already emits GenAI-semconv spans for graph, node, agent, cycle, model and tool calls plus
token/latency metrics. This module adds the domain signals the evals and dashboards need.

Env:
  OTEL_EXPORTER_OTLP_ENDPOINT     send traces + metrics to a collector (Langfuse, Jaeger, CloudWatch ...)
  MP_PROFILE_CONSOLE_TELEMETRY=1  print spans and metrics to stderr (stdout stays clean for JSON and MCP stdio)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, MetricReader, PeriodicExportingMetricReader
from strands.hooks import (
    AfterInvocationEvent,
    AfterNodeCallEvent,
    AfterToolCallEvent,
    BeforeNodeCallEvent,
    HookProvider,
    HookRegistry,
)
from strands.telemetry import StrandsTelemetry

_telemetry: StrandsTelemetry | None = None


def setup_telemetry(service_name: str = "mp-profile") -> StrandsTelemetry:
    """Configure exporters once per process (idempotent)."""
    global _telemetry
    if _telemetry is not None:
        return _telemetry
    os.environ.setdefault("OTEL_SERVICE_NAME", service_name)
    telemetry = StrandsTelemetry()
    otlp = bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
    console = os.environ.get("MP_PROFILE_CONSOLE_TELEMETRY") == "1"
    if otlp:
        telemetry.setup_otlp_exporter()
    if console:
        telemetry.setup_console_exporter(out=sys.stderr)
    readers: list[MetricReader] = []
    if console:
        readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter(out=sys.stderr)))
    if otlp:
        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
    telemetry.meter_provider = MeterProvider(resource=telemetry.resource, metric_readers=readers)
    metrics.set_meter_provider(telemetry.meter_provider)
    _telemetry = telemetry
    return telemetry


@dataclass(frozen=True)
class DomainMetrics:
    passages_ingested: metrics.Counter
    unattributed_passages: metrics.Counter
    profile_requests: metrics.Counter
    profile_latency: metrics.Histogram
    passages_considered: metrics.Histogram
    claims_emitted: metrics.Counter
    verifier_rejections: metrics.Counter
    summary_cache: metrics.Counter
    tool_calls: metrics.Counter
    tool_duration: metrics.Histogram
    node_duration: metrics.Histogram
    agent_tokens: metrics.Counter
    eval_scores: metrics.Histogram


@lru_cache(maxsize=1)
def domain_metrics() -> DomainMetrics:
    meter = metrics.get_meter("mp_profile")
    return DomainMetrics(
        passages_ingested=meter.create_counter("mp_profile.ingest.passages", unit="{passage}"),
        unattributed_passages=meter.create_counter("mp_profile.ingest.unattributed", unit="{passage}"),
        profile_requests=meter.create_counter("mp_profile.profile.requests", unit="{request}"),
        profile_latency=meter.create_histogram("mp_profile.profile.duration", unit="s"),
        passages_considered=meter.create_histogram("mp_profile.topic.passages", unit="{passage}"),
        claims_emitted=meter.create_counter("mp_profile.topic.claims", unit="{claim}"),
        verifier_rejections=meter.create_counter("mp_profile.verifier.rejections", unit="{claim}"),
        summary_cache=meter.create_counter("mp_profile.summary_cache.lookups", unit="{lookup}"),
        tool_calls=meter.create_counter("mp_profile.tool.calls", unit="{call}"),
        tool_duration=meter.create_histogram("mp_profile.tool.duration", unit="s"),
        node_duration=meter.create_histogram("mp_profile.graph.node.duration", unit="s"),
        agent_tokens=meter.create_counter("mp_profile.agent.tokens", unit="{token}"),
        eval_scores=meter.create_histogram("mp_profile.eval.score", unit="1"),
    )


class ToolMetricsHook(HookProvider):
    """Counts tool calls/failures and token usage per agent, tagged with the agent name."""

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(AfterToolCallEvent, self._after_tool)
        registry.add_callback(AfterInvocationEvent, self._after_invocation)

    def _after_tool(self, event: AfterToolCallEvent) -> None:
        failed = isinstance(event.result, Exception) or event.result.get("status") == "error"
        attrs = {"tool": event.tool_use["name"], "agent": event.agent.name, "status": "error" if failed else "ok"}
        domain_metrics().tool_calls.add(1, attrs)
        if event.duration is not None:
            domain_metrics().tool_duration.record(event.duration, attrs)

    def _after_invocation(self, event: AfterInvocationEvent) -> None:
        if event.result is None:
            return
        usage = event.result.metrics.accumulated_usage
        attrs = {"agent": event.agent.name}
        domain_metrics().agent_tokens.add(usage["inputTokens"], {**attrs, "kind": "input"})
        domain_metrics().agent_tokens.add(usage["outputTokens"], {**attrs, "kind": "output"})


class NodeTimingHook(HookProvider):
    """Per-node wall time for the profile graph (complements the graph's own spans)."""

    def __init__(self) -> None:
        self._started: dict[str, float] = {}

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeNodeCallEvent, self._before)
        registry.add_callback(AfterNodeCallEvent, self._after)

    def _before(self, event: BeforeNodeCallEvent) -> None:
        self._started[event.node_id] = time.perf_counter()

    def _after(self, event: AfterNodeCallEvent) -> None:
        started = self._started.pop(event.node_id, None)
        if started is not None:
            domain_metrics().node_duration.record(time.perf_counter() - started, {"node": event.node_id})
