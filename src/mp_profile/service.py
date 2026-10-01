"""Public entry point: name -> MPProfile | Disambiguation, with tracing and an online-eval run log."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from opentelemetry import trace

from .config import Settings
from .models import Disambiguation, MPProfile, Parliament, ProfileResponse
from .pipeline.graph import build_graph
from .pipeline.nodes import ProfileContext
from .pipeline.summarizers import Summarizer, make_summarizer
from .resolve import resolve_person
from .store import SpeechStore
from .telemetry import domain_metrics

tracer = trace.get_tracer("mp_profile.service")


async def build_profile(
    store: SpeechStore,
    query: str,
    settings: Settings | None = None,
    summarizer: Summarizer | None = None,
    parliament: Parliament | None = None,
) -> ProfileResponse:
    response, _ = await run_profile(store, query, settings, summarizer, parliament)
    return response


async def run_profile(
    store: SpeechStore,
    query: str,
    settings: Settings | None = None,
    summarizer: Summarizer | None = None,
    parliament: Parliament | None = None,
) -> tuple[ProfileResponse, list[str]]:
    """Like build_profile but also returns the executed step trajectory (used by trajectory evals)."""
    settings = settings or Settings()
    summarizer = summarizer or make_summarizer(settings.model_id)
    started = time.perf_counter()
    with tracer.start_as_current_span("mp_profile.build_profile") as span:
        span.set_attributes({"mp_profile.query": query, "mp_profile.summarizer": summarizer.name})
        resolution = resolve_person(store, query, parliament)
        trajectory = ["resolve"]
        response: ProfileResponse
        if resolution.person is None:
            assert resolution.disambiguation is not None
            response = resolution.disambiguation
        else:
            span.set_attribute("mp_profile.person_id", resolution.person.person_id)
            ctx = ProfileContext(store=store, settings=settings, summarizer=summarizer, person=resolution.person)
            await build_graph().invoke_async(
                f"Build the speech profile for {resolution.person.person_id}", invocation_state={"mp_profile": ctx}
            )
            if ctx.profile is None:
                raise RuntimeError("profile graph finished without producing a profile")
            response = ctx.profile
            trajectory.extend(ctx.trajectory)
        span.set_attribute("mp_profile.status", response.status)
        elapsed = time.perf_counter() - started
        domain_metrics().profile_requests.add(1, {"status": response.status})
        domain_metrics().profile_latency.record(elapsed, {"status": response.status})
        if settings.run_log_path:
            _log_run(settings, query, response, span.get_span_context().trace_id, elapsed)
        return response, trajectory


def _log_run(
    settings: Settings, query: str, response: MPProfile | Disambiguation, trace_id: int, elapsed: float
) -> None:
    """Append one JSON line per request; `mp-profile eval online` scores these against the store."""
    assert settings.run_log_path is not None
    settings.run_log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": f"{trace_id:032x}",
        "query": query,
        "elapsed_s": round(elapsed, 3),
        "response": response.model_dump(mode="json"),
    }
    with settings.run_log_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
