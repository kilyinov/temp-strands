"""Offline eval experiment on the golden set using strands-agents-evals.

Deterministic evaluators always run (they gate CI). With --judge, LLM-as-judge evaluators score
neutrality and faithfulness of the generated summaries (requires Bedrock credentials).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator, OutputEvaluator
from strands_evals.types import EvaluationData, EvaluationOutput

from ..config import Settings
from ..models import Disambiguation, MPProfile, ProfileResponse
from ..pipeline.summarizers import Summarizer
from ..service import run_profile
from ..store import SpeechStore
from ..telemetry import domain_metrics
from .checks import CheckResult, GoldenCase, golden_checks, trajectory_check
from .datasets import golden_cases

NEUTRALITY_RUBRIC = """You are auditing an automatically generated profile of what a parliamentarian said.
Score 1.0 if every summary and paraphrase neutrally reports what the member said, without praising,
criticising, speculating about motives, or presenting inferred positions as stated ones. Score 0.5 if there
is mild evaluative language. Score 0.0 if it editorialises or attributes positions not supported by quotes."""

FAITHFULNESS_RUBRIC = """You are checking a generated parliamentary profile. Each claim has a paraphrase and a
verbatim quote. Score 1.0 if every paraphrase is fully supported by its own quote and every career summary is
supported by the claims and counts shown; 0.5 if minor overstatement; 0.0 if any paraphrase or summary asserts
something its evidence does not support. Thin records described as thin are correct, not failures."""


def _parse(output: object) -> ProfileResponse:
    data = json.loads(output) if isinstance(output, str) else output
    if not isinstance(data, dict):
        raise TypeError("expected a profile dict")
    return (
        MPProfile.model_validate(data)
        if data.get("status") in ("ok", "partial")
        else Disambiguation.model_validate(data)
    )


class GoldenChecksEvaluator(Evaluator[str, str]):
    """Runs the deterministic golden checks; one EvaluationOutput per check."""

    def __init__(self, store: SpeechStore) -> None:
        super().__init__(name="golden_checks")
        self.store = store

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        metadata = evaluation_case.metadata or {}
        case = GoldenCase.model_validate(metadata["golden"])
        response = _parse(evaluation_case.actual_output)
        results = golden_checks(self.store, response, case)
        trajectory = evaluation_case.actual_trajectory
        if isinstance(trajectory, list):
            results.append(trajectory_check(response, [str(s) for s in trajectory]))
        return [_to_output(r) for r in results]


def _to_output(result: CheckResult) -> EvaluationOutput:
    domain_metrics().eval_scores.record(result.score, {"check": result.name.split(":")[0], "mode": "offline"})
    return EvaluationOutput(score=result.score, test_pass=result.passed, reason=f"{result.name}: {result.detail}")


def build_experiment(store: SpeechStore, settings: Settings, judge: bool) -> Experiment[str, str]:
    cases = [
        Case[str, str](name=c.name, input=c.query, metadata={"golden": c.model_dump(mode="json")})
        for c in golden_cases()
    ]
    evaluators: list[Evaluator[str, str]] = [GoldenChecksEvaluator(store)]
    if judge:
        evaluators += [
            OutputEvaluator(rubric=NEUTRALITY_RUBRIC, model=settings.judge_model_id),
            OutputEvaluator(rubric=FAITHFULNESS_RUBRIC, model=settings.judge_model_id),
        ]
    return Experiment[str, str](cases=cases, evaluators=evaluators)


def make_task(
    store: SpeechStore, settings: Settings, summarizer: Summarizer | None = None
) -> Callable[[Case[str, str]], Awaitable[dict[str, object]]]:
    async def task(case: Case[str, str]) -> dict[str, object]:
        response, trajectory = await run_profile(store, case.input, settings, summarizer)
        return {"output": response.model_dump_json(), "trajectory": trajectory}

    return task
