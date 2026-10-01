"""Runtime configuration read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(default_factory=lambda: Path(os.environ.get("MP_PROFILE_DB", "data/speeches.sqlite")))
    raw_cache_dir: Path = field(default_factory=lambda: Path(os.environ.get("MP_PROFILE_RAW_CACHE", "data/raw")))
    model_id: str = field(default_factory=lambda: os.environ.get("MP_PROFILE_MODEL", "none"))
    judge_model_id: str = field(
        default_factory=lambda: os.environ.get("MP_PROFILE_JUDGE_MODEL", "global.anthropic.claude-sonnet-4-6")
    )
    period_years: int = field(default_factory=lambda: _env_int("MP_PROFILE_PERIOD_YEARS", 3))
    passages_per_period: int = field(default_factory=lambda: _env_int("MP_PROFILE_PASSAGES_PER_PERIOD", 12))
    claims_per_period: int = field(default_factory=lambda: _env_int("MP_PROFILE_CLAIMS_PER_PERIOD", 3))
    run_log_path: Path | None = field(
        default_factory=lambda: Path(p) if (p := os.environ.get("MP_PROFILE_RUN_LOG")) else None
    )
    user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "MP_PROFILE_USER_AGENT", "mp-profile/0.1 (+https://github.com/kilyinov/temp-strands)"
        )
    )

    @property
    def uses_llm(self) -> bool:
        return self.model_id.lower() not in {"", "none", "extractive"}
