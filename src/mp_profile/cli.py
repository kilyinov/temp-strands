"""Command line: ingestion, profiles, MCP server, chat and evals."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from datetime import date
from pathlib import Path

from . import crosswalk
from .chat import create_chat_agent, parliament_mcp_client
from .config import Settings
from .evals.datasets import topic_labels
from .evals.experiment import build_experiment, make_task
from .evals.online import score_run_log
from .evals.topic_eval import evaluate_tagger
from .ingest import (
    ingest_federal_hansard,
    ingest_federal_people,
    ingest_victorian_hansard,
    ingest_victorian_people,
    link_handbook,
)
from .mcp_server import main as mcp_main
from .models import Disambiguation, MPProfile
from .sample import load_sample
from .service import build_profile
from .sources.http import HttpClient
from .store import SpeechStore
from .telemetry import setup_telemetry


def _date(value: str) -> date:
    return date.fromisoformat(value)


def _client(settings: Settings) -> HttpClient:
    return HttpClient(settings.user_agent, cache_dir=settings.raw_cache_dir)


def render_markdown(profile: MPProfile) -> str:
    lines = [f"# {profile.person.name}", ""]
    for m in profile.person.memberships:
        span = f"{m.start or '?'} – {m.end or 'present'}"
        lines.append(f"- {m.parliament} / {m.chamber}: {m.seat} ({m.party}) {span}")
    for note in profile.coverage_notes:
        lines.append(f"> Coverage: {note}")
    for cat in profile.categories:
        lines += [
            "",
            f"## {cat.category.value.title()} (confidence: {cat.confidence}, passages: {cat.passage_count})",
            "",
            cat.career_summary,
        ]
        if cat.evolution:
            lines += ["", f"*Over time:* {cat.evolution}"]
        for period in cat.periods:
            if not period.claims:
                continue
            lines += ["", f"### {period.label}", period.summary]
            for c in period.claims:
                lines.append(
                    f'- {c.paraphrase} "{c.quote}" — {c.ref.date}, {c.ref.debate_title} [{c.capacity}] {c.ref.url}'
                )
        for gap in cat.gaps:
            lines.append(f"- _Gap:_ {gap}")
    if profile.verification_issues:
        lines += ["", f"_{len(profile.verification_issues)} claim(s) removed by verification._"]
    return "\n".join(lines)


def cmd_ingest(args: argparse.Namespace, settings: Settings, store: SpeechStore) -> int:
    client = _client(settings)
    try:
        if args.what == "people":
            print("federal people:", ingest_federal_people(store, client))
            print("victorian people:", ingest_victorian_people(store, client))
            if args.handbook:
                print("handbook PHIDs linked:", link_handbook(store, client))
        elif args.what == "federal":
            print("passages:", ingest_federal_hansard(store, client, args.start, args.end, force=args.force))
        elif args.what == "victoria":
            print("passages:", ingest_victorian_hansard(store, client, args.start, args.end, force=args.force))
    finally:
        client.close()
    return 0


def cmd_profile(args: argparse.Namespace, settings: Settings, store: SpeechStore) -> int:
    response = asyncio.run(build_profile(store, args.name, settings, parliament=args.parliament))
    if args.json:
        print(response.model_dump_json(indent=2))
    elif isinstance(response, Disambiguation):
        print(response.message)
        for c in response.candidates:
            print(f"  {c.person_id}\t{c.name}\t{c.summary}")
    else:
        print(render_markdown(response))
    return 0 if response.status in ("ok", "partial") else 2


def cmd_crosswalk(args: argparse.Namespace, settings: Settings, store: SpeechStore) -> int:
    if args.action == "apply":
        print("links applied:", crosswalk.apply(store, args.file))
        return 0
    writer = csv.writer(sys.stdout)
    writer.writerow(["federal_id", "federal_name", "victorian_id", "victorian_name", "reason"])
    for s in crosswalk.suggest(store):
        writer.writerow([s.federal_id, s.federal_name, s.victorian_id, s.victorian_name, s.reason])
    return 0


def cmd_eval(args: argparse.Namespace, settings: Settings, store: SpeechStore) -> int:
    if args.mode == "online":
        scores, summary = score_run_log(store, args.log)
        print(
            json.dumps(
                {"runs": len(scores), "mean": summary, "failing_traces": [s.trace_id for s in scores if s.failed]},
                indent=2,
            )
        )
        return 0
    if args.mode == "topics":
        per_topic, macro = evaluate_tagger(topic_labels())
        for s in per_topic:
            print(f"{s.topic.value:<11} P={s.precision:.2f} R={s.recall:.2f} F1={s.f1:.2f}")
        print(f"macro F1={macro:.3f}")
        return 0 if macro >= args.min_f1 else 1
    experiment = build_experiment(store, settings, judge=args.judge)
    report = asyncio.run(experiment.run_evaluations_async(make_task(store, settings)))
    failures = [
        f"{case.get('name')}: {out.reason}"
        for case, outputs in zip(report.cases, report.detailed_results, strict=True)
        for out in outputs
        if not out.test_pass
    ]
    print(f"overall score {report.overall_score:.3f}; {len(report.cases)} case(s); {len(failures)} failing check(s)")
    for line in failures:
        print("  FAIL", line)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    return 1 if failures or not all(report.test_passes) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mp-profile", description=__doc__)
    parser.add_argument("--db", type=Path, help="speech store path (default $MP_PROFILE_DB or data/speeches.sqlite)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="fetch registries or Hansard into the store")
    ingest.add_argument("what", choices=["people", "federal", "victoria"])
    ingest.add_argument("--start", type=_date, default=date(2018, 12, 1))
    ingest.add_argument("--end", type=_date, default=date.today())
    ingest.add_argument("--handbook", action="store_true", help="also link APH Parliamentary Handbook PHIDs")
    ingest.add_argument("--force", action="store_true", help="re-ingest sittings already in the store")

    sub.add_parser("load-sample", help="load the bundled fictional sample dataset")
    sub.add_parser("coverage", help="show ingested sources and date ranges")

    profile = sub.add_parser("profile", help="summarise what a member said on the four topics")
    profile.add_argument("name")
    profile.add_argument("--parliament", choices=["commonwealth", "victoria"])
    profile.add_argument("--json", action="store_true")

    cw = sub.add_parser("crosswalk", help="federal <-> Victorian person links")
    cw.add_argument("action", choices=["suggest", "apply"])
    cw.add_argument("file", type=Path, nargs="?")

    sub.add_parser("mcp", help="run the Parliament MCP server on stdio")
    sub.add_parser("chat", help="interactive agent over the MCP server (needs a model)")

    ev = sub.add_parser("eval", help="offline golden-set, topic-tagger or online run-log evals")
    ev.add_argument("mode", choices=["offline", "topics", "online"])
    ev.add_argument("--judge", action="store_true", help="add LLM-as-judge evaluators (needs Bedrock)")
    ev.add_argument("--log", type=Path, help="run log JSONL for online mode")
    ev.add_argument("--output", type=Path, help="write offline reports as JSON")
    ev.add_argument("--min-f1", type=float, default=0.85)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    settings = Settings()
    if args.db:
        settings = Settings(db_path=args.db)
    if args.command == "mcp":
        mcp_main()
        return 0
    setup_telemetry()
    store = SpeechStore(settings.db_path)
    if args.command == "load-sample":
        load_sample(store)
        print(json.dumps(store.coverage(), indent=2))
        return 0
    if args.command == "coverage":
        print(json.dumps(store.coverage(), indent=2))
        return 0
    if args.command == "ingest":
        return cmd_ingest(args, settings, store)
    if args.command == "profile":
        return cmd_profile(args, settings, store)
    if args.command == "crosswalk":
        if args.action == "apply" and not args.file:
            raise SystemExit("crosswalk apply needs a CSV file")
        return cmd_crosswalk(args, settings, store)
    if args.command == "eval":
        if args.mode == "online" and not args.log:
            raise SystemExit("eval online needs --log")
        return cmd_eval(args, settings, store)
    if args.command == "chat":
        return run_chat(settings)
    return 1


def run_chat(settings: Settings) -> int:
    if not settings.uses_llm:
        raise SystemExit("chat needs a model: set MP_PROFILE_MODEL to a Bedrock model id")
    client = parliament_mcp_client()
    with client:
        agent = create_chat_agent(settings.model_id, client)
        while True:
            try:
                question = input("> ").strip()
            except EOFError:
                return 0
            if question:
                agent(question)
                print()


if __name__ == "__main__":
    raise SystemExit(main())
