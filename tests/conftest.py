from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from mp_profile.config import Settings
from mp_profile.sample import load_sample
from mp_profile.store import SpeechStore

ALEX = "oa:person/20001"
ALEX_VIC = "vic:member/9001"
SAM = "oa:person/20002"
PAT = "vic:member/9003"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SpeechStore]:
    s = SpeechStore(tmp_path / "speeches.sqlite")
    load_sample(s)
    yield s
    s.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "speeches.sqlite", model_id="none", run_log_path=tmp_path / "runs.jsonl")
