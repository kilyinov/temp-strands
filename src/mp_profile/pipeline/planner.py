"""Deterministic coverage planner: split a whole career into periods and count evidence per topic."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from ..models import TOPICS, Parliament, Person, Topic
from ..store import SpeechStore

PARLIAMENT_LABELS: dict[str, str] = {"commonwealth": "Commonwealth", "victoria": "Victoria"}
SOURCE_PARLIAMENT: dict[str, Parliament] = {"openaustralia": "commonwealth", "vic_hansard": "victoria"}


@dataclass
class PlannedPeriod:
    label: str
    parliament: Parliament
    start: date
    end: date
    topic_counts: dict[Topic, int] = field(default_factory=dict)


@dataclass
class CoveragePlan:
    person: Person
    periods: list[PlannedPeriod]
    coverage_notes: list[str]

    def periods_for(self, topic: Topic) -> list[PlannedPeriod]:
        return [p for p in self.periods if p.topic_counts.get(topic, 0) > 0]

    def total(self, topic: Topic) -> int:
        return sum(p.topic_counts.get(topic, 0) for p in self.periods)


def _merge(intervals: list[tuple[date, date]]) -> list[tuple[date, date]]:
    merged: list[tuple[date, date]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + timedelta(days=1):
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def service_intervals(store: SpeechStore, person: Person, today: date) -> dict[Parliament, list[tuple[date, date]]]:
    speeches = store.speech_dates(person.person_id)
    out: dict[Parliament, list[tuple[date, date]]] = {}
    for parliament in ("commonwealth", "victoria"):
        mships = [m for m in person.memberships if m.parliament == parliament]
        if not mships:
            continue
        dated = [(m.start, m.end or today) for m in mships if m.start]
        spoken = [d for d, p in speeches if p == parliament]
        if not dated and spoken:
            dated = [(spoken[0], spoken[-1])]
        if dated:
            out[parliament] = _merge(dated)
    return out


def _windows(start: date, end: date, years: int) -> list[tuple[date, date]]:
    out = []
    cursor = start
    while cursor <= end:
        nxt = date(cursor.year + years, cursor.month, min(cursor.day, 28))
        out.append((cursor, min(end, nxt - timedelta(days=1))))
        cursor = nxt
    return out


def coverage_notes(
    store: SpeechStore, person: Person, intervals: dict[Parliament, list[tuple[date, date]]]
) -> list[str]:
    notes = []
    ingested: dict[Parliament, tuple[date, date]] = {}
    for row in store.coverage():
        parliament = SOURCE_PARLIAMENT.get(row["source"])
        if parliament is None:
            continue
        first, last = date.fromisoformat(row["first"]), date.fromisoformat(row["last"])
        prev = ingested.get(parliament)
        ingested[parliament] = (min(first, prev[0]), max(last, prev[1])) if prev else (first, last)
    for parliament, spans in intervals.items():
        label = PARLIAMENT_LABELS[parliament]
        served = (spans[0][0], spans[-1][1])
        window = ingested.get(parliament)
        if window is None:
            notes.append(f"No {label} Hansard has been ingested; {label} service is not summarised.")
            continue
        if served[0] < window[0]:
            notes.append(
                f"{label} service began {served[0].isoformat()} but ingested Hansard starts {window[0].isoformat()}; "
                "earlier speeches are not covered."
            )
        if served[1] > window[1] + timedelta(days=60):
            notes.append(f"{label} Hansard ingested only up to {window[1].isoformat()}.")
    if any(m.parliament == "victoria" and m.start is None for m in person.memberships):
        notes.append("Victorian membership dates are not published by the member search; periods use speech dates.")
    return notes


def plan_coverage(store: SpeechStore, person: Person, period_years: int, today: date | None = None) -> CoveragePlan:
    today = today or date.today()
    intervals = service_intervals(store, person, today)
    periods = []
    for parliament, spans in intervals.items():
        for span_start, span_end in spans:
            for start, end in _windows(span_start, span_end, period_years):
                counts = store.topic_counts(person.person_id, start, end)
                periods.append(
                    PlannedPeriod(
                        label=f"{PARLIAMENT_LABELS[parliament]} {start.year}–{end.year}",
                        parliament=parliament,
                        start=start,
                        end=end,
                        topic_counts={t: counts.get(t, 0) for t in TOPICS},
                    )
                )
    periods.sort(key=lambda p: p.start)
    return CoveragePlan(person=person, periods=periods, coverage_notes=coverage_notes(store, person, intervals))
