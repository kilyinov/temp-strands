"""Domain and output models shared by ingestion, the MCP server, the graph and evals."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

Parliament = Literal["commonwealth", "victoria"]
Chamber = Literal["house_of_reps", "senate", "legislative_assembly", "legislative_council"]
Capacity = Literal["personal", "ministerial", "shadow", "committee", "presiding"]
Confidence = Literal["low", "medium", "high"]
TalkType = Literal["speech", "continuation", "question", "answer", "procedural", "petition"]


class Topic(str, Enum):
    HOUSING = "housing"
    HEALTHCARE = "healthcare"
    ECONOMY = "economy"
    EDUCATION = "education"


TOPICS: tuple[Topic, ...] = (Topic.HOUSING, Topic.HEALTHCARE, Topic.ECONOMY, Topic.EDUCATION)


class Membership(BaseModel):
    """One continuous period of service in one chamber for one seat."""

    source_id: str
    parliament: Parliament
    chamber: Chamber
    seat: str
    party: str
    start: date | None = None
    end: date | None = None


class Role(BaseModel):
    """An office held during a period, e.g. a ministry or shadow ministry."""

    title: str
    kind: Literal["ministerial", "shadow", "committee", "presiding", "other"]
    start: date | None = None
    end: date | None = None


class Person(BaseModel):
    person_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    memberships: list[Membership] = Field(default_factory=list)
    roles: list[Role] = Field(default_factory=list)


class Passage(BaseModel):
    """One attributed speech segment from Hansard."""

    passage_id: str
    source_speaker_id: str
    speaker_name: str
    parliament: Parliament
    chamber: Chamber
    date: date
    debate_title: str
    text: str
    url: str
    talktype: TalkType = "speech"
    stated_role: str | None = None
    is_proof: bool = False
    attribution_confidence: float = 1.0
    parse_confidence: float = 1.0


class TopicTag(BaseModel):
    topic: Topic
    score: float
    matched_terms: list[str]


class PersonCandidate(BaseModel):
    person_id: str
    name: str
    score: float
    summary: str


class HansardRef(BaseModel):
    passage_id: str
    parliament: Parliament
    chamber: Chamber
    date: date
    debate_title: str
    url: str
    speaker_role: str | None = None
    is_proof: bool = False
    parse_confidence: float = 1.0


class Claim(BaseModel):
    paraphrase: str
    quote: str
    ref: HansardRef
    capacity: Capacity


class Period(BaseModel):
    label: str
    start: date
    end: date
    passage_count: int
    summary: str
    claims: list[Claim] = Field(default_factory=list)


class CategorySummary(BaseModel):
    category: Topic
    career_summary: str
    evolution: str | None = None
    periods: list[Period] = Field(default_factory=list)
    passage_count: int = 0
    confidence: Confidence = "low"
    gaps: list[str] = Field(default_factory=list)


class VerificationIssue(BaseModel):
    category: Topic
    passage_id: str
    check: Literal["quote_fidelity", "speaker_attribution", "membership_date", "topic_relevance", "unknown_passage"]
    detail: str


class MPProfile(BaseModel):
    status: Literal["ok", "partial"] = "ok"
    person: Person
    categories: list[CategorySummary]
    coverage_notes: list[str] = Field(default_factory=list)
    verification_issues: list[VerificationIssue] = Field(default_factory=list)
    generated_with: dict[str, str] = Field(default_factory=dict)


class Disambiguation(BaseModel):
    status: Literal["needs_disambiguation", "not_found"]
    query: str
    candidates: list[PersonCandidate] = Field(default_factory=list)
    message: str


ProfileResponse = MPProfile | Disambiguation
