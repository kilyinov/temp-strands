"""Online evals: re-score production profiles from the run log against the speech store.

Each JSONL record carries the trace_id of the request, so scores can be attached to the trace in the
observability backend (e.g. Langfuse scores) and failing traces sampled for human review.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from ..models import MPProfile
from ..store import SpeechStore
from ..telemetry import domain_metrics
from .checks import evidence_checks, structure_checks


class OnlineScore(BaseModel):
    trace_id: str
    query: str
    person_id: str
    elapsed_s: float
    checks: dict[str, float]
    failed: list[str]


def score_run_log(store: SpeechStore, path: Path) -> tuple[list[OnlineScore], dict[str, float]]:
    scores: list[OnlineScore] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["response"].get("status") not in ("ok", "partial"):
            continue
        profile = MPProfile.model_validate(record["response"])
        results = evidence_checks(store, profile) + structure_checks(profile)
        for r in results:
            domain_metrics().eval_scores.record(r.score, {"check": r.name, "mode": "online"})
        scores.append(
            OnlineScore(
                trace_id=record["trace_id"],
                query=record["query"],
                person_id=profile.person.person_id,
                elapsed_s=record["elapsed_s"],
                checks={r.name: r.score for r in results},
                failed=[r.name for r in results if not r.passed],
            )
        )
    totals: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        for name, value in s.checks.items():
            totals[name].append(value)
    summary = {name: round(sum(v) / len(v), 4) for name, v in totals.items()}
    return scores, summary
