"""Topic-tagger precision / recall / F1 against labelled passages."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import TOPICS, Topic
from ..topics import tag_passage
from .datasets import LabelledPassage


@dataclass(frozen=True)
class TopicScore:
    topic: Topic
    precision: float
    recall: float
    f1: float


def evaluate_tagger(labelled: list[LabelledPassage]) -> tuple[list[TopicScore], float]:
    scores = []
    for topic in TOPICS:
        tp = fp = fn = 0
        for item in labelled:
            predicted = topic in {t.topic for t in tag_passage(item.text)}
            actual = topic in item.topics
            tp += predicted and actual
            fp += predicted and not actual
            fn += actual and not predicted
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(TopicScore(topic, round(precision, 3), round(recall, 3), round(f1, 3)))
    macro = sum(s.f1 for s in scores) / len(scores)
    return scores, round(macro, 3)
