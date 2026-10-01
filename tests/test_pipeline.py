import json
from pathlib import Path

import pytest

from mp_profile.config import Settings
from mp_profile.models import TOPICS, Claim, MPProfile, Passage, Period, Person, Topic
from mp_profile.pipeline.planner import PlannedPeriod, plan_coverage
from mp_profile.pipeline.summarizers import CareerDraft, ExtractiveSummarizer, PeriodDraft
from mp_profile.pipeline.verify import verify_claim
from mp_profile.service import build_profile, run_profile
from mp_profile.store import SpeechStore

from .conftest import ALEX, PAT, SAM


async def _profile(store: SpeechStore, settings: Settings, query: str) -> MPProfile:
    result = await build_profile(store, query, settings)
    assert isinstance(result, MPProfile)
    return result


def test_plan_covers_whole_career_in_both_parliaments(store: SpeechStore, settings: Settings) -> None:
    person = store.get_person(ALEX)
    assert person is not None
    plan = plan_coverage(store, person, settings.period_years)
    assert {p.parliament for p in plan.periods} == {"commonwealth", "victoria"}
    commonwealth = [p for p in plan.periods if p.parliament == "commonwealth"]
    assert commonwealth[0].start.isoformat() == "2007-11-24"
    assert commonwealth[-1].end.isoformat() == "2013-09-07"
    assert any("2007-11-24" in note for note in plan.coverage_notes)


async def test_profile_has_four_categories_with_verified_cited_claims(store: SpeechStore, settings: Settings) -> None:
    profile = await _profile(store, settings, "Alex Morgan")
    assert [c.category for c in profile.categories] == list(TOPICS)
    claims = [cl for c in profile.categories for p in c.periods for cl in p.claims]
    assert claims
    for claim in claims:
        passage = store.get_passage(claim.ref.passage_id)
        assert passage is not None
        assert claim.quote in passage.text
        assert claim.ref.url
    assert {c.ref.parliament for c in claims} == {"commonwealth", "victoria"}
    assert profile.verification_issues == []


async def test_quoted_material_never_attributed(store: SpeechStore, settings: Settings) -> None:
    for query in ("Alex Morgan", "Sam Taylor"):
        profile = await _profile(store, settings, query)
        for c in profile.categories:
            for p in c.periods:
                for claim in p.claims:
                    assert "myth" not in claim.quote
                    assert "Every dollar we spend" not in claim.quote


async def test_capacity_labels(store: SpeechStore, settings: Settings) -> None:
    profile = await _profile(store, settings, "Alex Morgan")
    caps = {cl.ref.date.year: cl.capacity for c in profile.categories for p in c.periods for cl in p.claims}
    assert caps[2012] == "ministerial"
    sam = await _profile(store, settings, "Sam Taylor")
    assert {cl.capacity for c in sam.categories for p in c.periods for cl in p.claims} <= {"shadow", "personal"}


async def test_sparse_member_gets_gaps_not_claims(store: SpeechStore, settings: Settings) -> None:
    profile = await _profile(store, settings, PAT)
    assert all(not p.claims for c in profile.categories for p in c.periods)
    assert all(c.gaps for c in profile.categories)
    assert all(c.confidence == "low" for c in profile.categories)


async def test_trajectory_and_run_log(store: SpeechStore, settings: Settings) -> None:
    response, trajectory = await run_profile(store, "Alex Morgan", settings)
    assert trajectory[:2] == ["resolve", "plan"] and trajectory[-1] == "verify"
    assert set(trajectory[2:-1]) == {t.value for t in TOPICS}
    assert settings.run_log_path is not None
    record = json.loads(Path(settings.run_log_path).read_text().splitlines()[-1])
    assert record["query"] == "Alex Morgan" and record["response"]["status"] == response.status


async def test_disambiguation_skips_graph(store: SpeechStore, settings: Settings) -> None:
    response, trajectory = await run_profile(store, "Taylor", settings)
    assert response.status == "needs_disambiguation" and trajectory == ["resolve"]


class CountingSummarizer(ExtractiveSummarizer):
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    async def summarize_period(
        self, person: Person, topic: Topic, period: PlannedPeriod, passages: list[Passage], max_claims: int
    ) -> PeriodDraft:
        self.calls += 1
        return await super().summarize_period(person, topic, period, passages, max_claims)


async def test_period_summaries_are_cached(store: SpeechStore, settings: Settings) -> None:
    summarizer = CountingSummarizer()
    await build_profile(store, SAM, settings, summarizer)
    first = summarizer.calls
    assert first > 0
    await build_profile(store, SAM, settings, summarizer)
    assert summarizer.calls == first


class FabricatingSummarizer(ExtractiveSummarizer):
    name = "fabricating"

    async def summarize_period(
        self, person: Person, topic: Topic, period: PlannedPeriod, passages: list[Passage], max_claims: int
    ) -> PeriodDraft:
        draft = await super().summarize_period(person, topic, period, passages, max_claims)
        for claim in draft.claims:
            claim.quote = claim.quote + " and I promise free houses for everyone"
        return draft


async def test_verifier_removes_fabricated_quotes(store: SpeechStore, settings: Settings) -> None:
    result = await build_profile(store, ALEX, settings, FabricatingSummarizer())
    assert isinstance(result, MPProfile)
    assert all(not p.claims for c in result.categories for p in c.periods)
    assert result.verification_issues
    assert {i.check for i in result.verification_issues} == {"quote_fidelity"}


class BrokenSummarizer(ExtractiveSummarizer):
    name = "broken"

    async def summarize_career(self, person: Person, topic: Topic, periods: list[Period]) -> CareerDraft:
        if topic is Topic.ECONOMY:
            raise RuntimeError("model unavailable")
        return await super().summarize_career(person, topic, periods)


async def test_one_failed_analyst_gives_partial_profile(store: SpeechStore, settings: Settings) -> None:
    result = await build_profile(store, ALEX, settings, BrokenSummarizer())
    assert isinstance(result, MPProfile)
    assert result.status == "partial"
    economy = next(c for c in result.categories if c.category is Topic.ECONOMY)
    assert economy.confidence == "low" and any("failed" in g.lower() for g in economy.gaps)
    housing = next(c for c in result.categories if c.category is Topic.HOUSING)
    assert any(p.claims for p in housing.periods)


async def test_verify_claim_rejects_wrong_speaker_and_quoted_lines(store: SpeechStore, settings: Settings) -> None:
    profile = await _profile(store, settings, "Alex Morgan")
    housing = profile.categories[0]
    claim = next(cl for p in housing.periods for cl in p.claims if cl.ref.parliament == "victoria")
    sam = store.get_person(SAM)
    assert sam is not None
    assert verify_claim(store, sam, housing, claim)[1].check == "speaker_attribution"  # type: ignore[union-attr]
    alex = profile.person
    quoted = claim.model_copy(update={"quote": "The housing crisis is a myth invented by renters' lobby groups."})
    assert verify_claim(store, alex, housing, quoted)[1].check == "quote_fidelity"  # type: ignore[union-attr]
    ok_claim, issue = verify_claim(store, alex, housing, claim)
    assert isinstance(ok_claim, Claim) and issue is None


@pytest.mark.parametrize("query", ["Nobody Real", ""])
async def test_not_found(store: SpeechStore, settings: Settings, query: str) -> None:
    assert (await build_profile(store, query, settings)).status == "not_found"
