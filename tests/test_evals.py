import json
from pathlib import Path

from mp_profile.config import Settings
from mp_profile.evals.checks import evidence_checks, golden_checks, structure_checks
from mp_profile.evals.datasets import golden_cases
from mp_profile.evals.experiment import build_experiment, make_task
from mp_profile.evals.online import score_run_log
from mp_profile.models import MPProfile
from mp_profile.service import build_profile, run_profile
from mp_profile.store import SpeechStore


async def test_golden_set_passes(store: SpeechStore, settings: Settings) -> None:
    for case in golden_cases():
        response, _ = await run_profile(store, case.query, settings)
        failed = [r for r in golden_checks(store, response, case) if not r.passed]
        assert not failed, (case.name, failed)


async def test_checks_detect_tampering(store: SpeechStore, settings: Settings) -> None:
    profile = await build_profile(store, "Alex Morgan", settings)
    assert isinstance(profile, MPProfile)
    claim = profile.categories[0].periods[-1].claims[0]
    claim.quote = "I will abolish all rent."
    claim.paraphrase = "Gave a brilliant, heroic speech."
    results = {r.name: r for r in evidence_checks(store, profile) + structure_checks(profile)}
    assert not results["quote_fidelity"].passed
    assert not results["neutrality_lexicon"].passed
    assert "brilliant" in results["neutrality_lexicon"].detail


async def test_strands_evals_experiment(store: SpeechStore, settings: Settings) -> None:
    experiment = build_experiment(store, settings, judge=False)
    report = await experiment.run_evaluations_async(make_task(store, settings))
    assert len(report.cases) == len(golden_cases())
    assert all(report.test_passes)
    assert all(out.test_pass for outputs in report.detailed_results for out in outputs)


async def test_online_scoring_reads_run_log(store: SpeechStore, settings: Settings) -> None:
    await build_profile(store, "Alex Morgan", settings)
    await build_profile(store, "Taylor", settings)
    assert settings.run_log_path is not None
    log = Path(settings.run_log_path)
    record = json.loads(log.read_text().splitlines()[0])
    record["response"]["categories"][0]["periods"][-1]["claims"][0]["quote"] = "fabricated words"
    record["trace_id"] = "tampered"
    log.write_text(log.read_text() + json.dumps(record) + "\n")
    scores, summary = score_run_log(store, log)
    assert len(scores) == 2
    assert scores[0].failed == []
    assert scores[1].trace_id == "tampered" and "quote_fidelity" in scores[1].failed
    assert summary["quote_fidelity"] < 1.0
