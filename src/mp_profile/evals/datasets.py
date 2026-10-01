"""Eval datasets bundled with the package."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from ..models import Topic
from .checks import GoldenCase

EVALS_DIR = Path(__file__).parent
SAMPLE_DIR = EVALS_DIR.parent / "sample_data"


class LabelledPassage(BaseModel):
    text: str
    topics: list[Topic]


def golden_cases(path: Path = EVALS_DIR / "golden_cases.json") -> list[GoldenCase]:
    return TypeAdapter(list[GoldenCase]).validate_json(path.read_text())


def topic_labels(path: Path = SAMPLE_DIR / "topic_labels.jsonl") -> list[LabelledPassage]:
    return [LabelledPassage.model_validate(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
