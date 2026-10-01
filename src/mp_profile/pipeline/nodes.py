"""Custom Strands graph nodes. Shared state travels in invocation_state["mp_profile"] (a ProfileContext)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace
from strands.multiagent.base import MultiAgentBase, MultiAgentResult, Status
from strands.types.multiagent import MultiAgentInput

from ..config import Settings
from ..evidence import build_ref, capacity_at, own_words
from ..models import CategorySummary, Claim, Confidence, MPProfile, Passage, Period, Person, Topic, VerificationIssue
from ..store import SpeechStore
from ..telemetry import domain_metrics
from .planner import CoveragePlan, PlannedPeriod, plan_coverage
from .summarizers import PeriodDraft, Summarizer
from .verify import verify_category

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("mp_profile.pipeline")
MIN_WORDS = 30


@dataclass
class ProfileContext:
    store: SpeechStore
    settings: Settings
    summarizer: Summarizer
    person: Person
    plan: CoveragePlan | None = None
    categories: dict[Topic, CategorySummary] = field(default_factory=dict)
    failures: dict[Topic, str] = field(default_factory=dict)
    issues: list[VerificationIssue] = field(default_factory=list)
    profile: MPProfile | None = None
    trajectory: list[str] = field(default_factory=list)


def _ctx(invocation_state: dict[str, Any] | None) -> ProfileContext:
    ctx = (invocation_state or {}).get("mp_profile")
    if not isinstance(ctx, ProfileContext):
        raise ValueError("invocation_state['mp_profile'] must hold a ProfileContext")
    return ctx


class _Node(MultiAgentBase):
    node_name = "node"

    def __init__(self) -> None:
        super().__init__()
        self.id = self.node_name

    async def run(self, ctx: ProfileContext) -> None:
        raise NotImplementedError

    async def invoke_async(
        self, task: MultiAgentInput, invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> MultiAgentResult:
        started = time.perf_counter()
        ctx = _ctx(invocation_state)
        ctx.trajectory.append(self.node_name)
        await self.run(ctx)
        return MultiAgentResult(status=Status.COMPLETED, execution_time=int((time.perf_counter() - started) * 1000))


class PlanNode(_Node):
    node_name = "plan"

    async def run(self, ctx: ProfileContext) -> None:
        with tracer.start_as_current_span("mp_profile.plan_coverage") as span:
            ctx.plan = plan_coverage(ctx.store, ctx.person, ctx.settings.period_years)
            span.set_attributes(
                {
                    "mp_profile.person_id": ctx.person.person_id,
                    "mp_profile.periods": len(ctx.plan.periods),
                    "mp_profile.coverage_notes": len(ctx.plan.coverage_notes),
                }
            )


def _confidence(total: int, periods_with_evidence: int, attribution: float) -> Confidence:
    if total >= 20 and periods_with_evidence >= 2 and attribution >= 0.95:
        return "high"
    if total >= 5:
        return "medium"
    return "low"


class TopicAnalystNode(_Node):
    """Map: summarise each period with evidence (cached). Reduce: whole-career summary."""

    def __init__(self, topic: Topic) -> None:
        self.topic = topic
        self.node_name = topic.value
        super().__init__()

    def _cache_key(self, ctx: ProfileContext, period: PlannedPeriod, passages: list[Passage]) -> str:
        payload = json.dumps(
            [
                ctx.person.person_id,
                self.topic.value,
                period.label,
                ctx.summarizer.name,
                [p.passage_id for p in passages],
            ]
        )
        return "period:" + hashlib.sha256(payload.encode()).hexdigest()

    async def _period(self, ctx: ProfileContext, period: PlannedPeriod) -> Period:
        assert ctx.plan is not None
        found = ctx.store.search_passages(
            ctx.person.person_id, self.topic, period.start, period.end, limit=ctx.settings.passages_per_period * 3
        )
        passages = [p for p, _ in found if len(own_words(p.text).split()) >= MIN_WORDS][
            : ctx.settings.passages_per_period
        ]
        by_id = {p.passage_id: p for p in passages}
        domain_metrics().passages_considered.record(len(passages), {"topic": self.topic.value})
        if not passages:
            return Period(
                label=period.label,
                start=period.start,
                end=period.end,
                passage_count=period.topic_counts.get(self.topic, 0),
                summary="Only brief or procedural mentions in this period.",
            )
        key = self._cache_key(ctx, period, passages)
        cached = ctx.store.cache_get(key)
        domain_metrics().summary_cache.add(1, {"hit": cached is not None, "topic": self.topic.value})
        if cached is not None:
            draft = PeriodDraft.model_validate(cached)
        else:
            draft = await ctx.summarizer.summarize_period(
                ctx.person, self.topic, period, passages, ctx.settings.claims_per_period
            )
            ctx.store.cache_put(key, draft.model_dump(mode="json"))
        claims = []
        for d in draft.claims[: ctx.settings.claims_per_period]:
            passage = by_id.get(d.passage_id) or ctx.store.get_passage(d.passage_id)
            if passage is None:
                ctx.issues.append(
                    VerificationIssue(
                        category=self.topic,
                        passage_id=d.passage_id,
                        check="unknown_passage",
                        detail="model cited an unknown passage",
                    )
                )
                continue
            claims.append(
                Claim(
                    paraphrase=d.paraphrase,
                    quote=d.quote,
                    ref=build_ref(ctx.person, passage),
                    capacity=capacity_at(ctx.person, passage),
                )
            )
        domain_metrics().claims_emitted.add(len(claims), {"topic": self.topic.value})
        return Period(
            label=period.label,
            start=period.start,
            end=period.end,
            passage_count=period.topic_counts.get(self.topic, 0),
            summary=draft.summary,
            claims=claims,
        )

    async def run(self, ctx: ProfileContext) -> None:
        assert ctx.plan is not None
        with tracer.start_as_current_span(f"mp_profile.topic.{self.topic.value}") as span:
            try:
                ctx.categories[self.topic] = await self._analyse(ctx)
            except Exception as exc:  # partial completion: one failed topic must not sink the profile
                logger.exception("topic %s failed", self.topic.value)
                span.record_exception(exc)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                ctx.failures[self.topic] = f"{type(exc).__name__}: {exc}"
                ctx.categories[self.topic] = CategorySummary(
                    category=self.topic,
                    career_summary=f"The {self.topic.value} summary could not be generated.",
                    passage_count=ctx.plan.total(self.topic),
                    gaps=[f"Analysis failed ({type(exc).__name__}); retry later."],
                )
            summary = ctx.categories[self.topic]
            span.set_attributes(
                {
                    "mp_profile.topic": self.topic.value,
                    "mp_profile.passage_count": summary.passage_count,
                    "mp_profile.claims": sum(len(p.claims) for p in summary.periods),
                    "mp_profile.confidence": summary.confidence,
                }
            )

    async def _analyse(self, ctx: ProfileContext) -> CategorySummary:
        assert ctx.plan is not None
        total = ctx.plan.total(self.topic)
        if total == 0:
            return CategorySummary(
                category=self.topic,
                career_summary=f"No recorded statements on {self.topic.value} in the ingested Hansard.",
                gaps=[f"No {self.topic.value} passages found for this member in the ingested Hansard."],
            )
        periods = [await self._period(ctx, p) for p in ctx.plan.periods_for(self.topic)]
        career = await ctx.summarizer.summarize_career(ctx.person, self.topic, periods)
        attribution = [c.ref for p in periods for c in p.claims]
        gaps = []
        silent = [p.label for p in ctx.plan.periods if p.topic_counts.get(self.topic, 0) == 0]
        if silent:
            gaps.append(f"No {self.topic.value} passages in: {', '.join(silent)}.")
        truncated = [
            p.label
            for p in ctx.plan.periods_for(self.topic)
            if p.topic_counts.get(self.topic, 0) > ctx.settings.passages_per_period
        ]
        if not any(p.claims for p in periods):
            gaps.append(f"Only brief or procedural {self.topic.value} mentions; no substantive statements to quote.")
        if truncated:
            gaps.append(
                f"Summaries for {', '.join(truncated)} use the {ctx.settings.passages_per_period} "
                "most relevant passages per period, not every passage."
            )
        mean_attr = 1.0
        if attribution:
            scores = []
            for ref in attribution:
                passage = ctx.store.get_passage(ref.passage_id)
                scores.append(passage.attribution_confidence if passage else 0.0)
            mean_attr = sum(scores) / len(scores)
        return CategorySummary(
            category=self.topic,
            career_summary=career.career_summary,
            evolution=career.evolution,
            periods=periods,
            passage_count=total,
            confidence=_confidence(total, len(periods), mean_attr),
            gaps=gaps,
        )


class VerifyNode(_Node):
    node_name = "verify"

    async def run(self, ctx: ProfileContext) -> None:
        assert ctx.plan is not None
        with tracer.start_as_current_span("mp_profile.verify") as span:
            categories = []
            for topic in (Topic.HOUSING, Topic.HEALTHCARE, Topic.ECONOMY, Topic.EDUCATION):
                summary = ctx.categories.get(topic) or CategorySummary(
                    category=topic, career_summary="Not generated.", gaps=["Topic analyst did not run."]
                )
                verified, issues = verify_category(ctx.store, ctx.person, summary)
                ctx.issues.extend(issues)
                categories.append(verified)
            ctx.profile = MPProfile(
                status="partial" if ctx.failures else "ok",
                person=ctx.person,
                categories=categories,
                coverage_notes=ctx.plan.coverage_notes,
                verification_issues=ctx.issues,
                generated_with={"summarizer": ctx.summarizer.name},
            )
            span.set_attributes(
                {"mp_profile.verification_issues": len(ctx.issues), "mp_profile.status": ctx.profile.status}
            )
