from mp_profile.evals.datasets import topic_labels
from mp_profile.evals.topic_eval import evaluate_tagger
from mp_profile.models import Topic
from mp_profile.topics import best_sentences, tag_passage


def test_tags_multiple_topics() -> None:
    tags = {t.topic for t in tag_passage("Mortgage stress and inflation are hurting first home buyers and wages.")}
    assert tags == {Topic.HOUSING, Topic.ECONOMY}


def test_procedural_text_untagged() -> None:
    assert tag_passage("Order! The member will resume his seat.") == []


def test_tagger_quality_gate() -> None:
    per_topic, macro = evaluate_tagger(topic_labels())
    assert macro >= 0.85
    assert all(s.precision >= 0.8 for s in per_topic)


def test_best_sentences_skips_reported_speech() -> None:
    reported = "Of course, the member for Griffith has said that there isn't a housing supply problem in this country."
    own = "Our position is that the key to addressing the housing shortage is more supply."
    assert best_sentences("", Topic.HOUSING, [reported, own], limit=2) == [own]
