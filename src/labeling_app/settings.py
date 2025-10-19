"""Configuration helpers for the labeling platform."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppPaths:
    """Resolved filesystem locations for labeling artifacts."""

    root: Path
    seeds: Path
    reviews: Path
    evals: Path

    @classmethod
    def from_root(cls, root: Path) -> AppPaths:
        return cls(
            root=root,
            seeds=root / "seeds",
            reviews=root / "reviews",
            evals=root / "evals",
        )


@dataclass(frozen=True)
class AppSettings:
    """Top-level application configuration derived from environment variables."""

    database_url: str | None
    auth_token: str | None
    paths: AppPaths
    runs_root: Path

    def require_database_url(self) -> str:
        if self.database_url is None or self.database_url.strip() == "":
            raise RuntimeError(
                "TURSO_DATABASE_URL is not configured. "
                "Set it in the environment or .env before running the labeling platform."
            )
        return self.database_url

    def find_latest_run_file(self) -> Path | None:
        """Return the newest `run.jsonl` file under outputs/runs if available."""
        runs_root = self.runs_root
        if not runs_root.exists():
            return None

        candidates: list[tuple[float, Path]] = []
        for entry in runs_root.iterdir():
            if not entry.is_dir():
                continue
            run_file = entry / "run.jsonl"
            if run_file.exists():
                try:
                    candidates.append((run_file.stat().st_mtime, run_file))
                except OSError:
                    continue

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]


def _read_env(key: str, default: str | None = None) -> str | None:
    """Read an environment value with optional default and whitespace trimming."""
    value = os.getenv(key, default)
    if value is None:
        return None
    return value.strip() or default


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load application settings once and cache the result for reuse."""
    data_root_env = _read_env("LABEL_DATA_ROOT")
    data_root = (
        Path(data_root_env).resolve() if data_root_env else Path("data/humanLabel").resolve()
    )
    paths = AppPaths.from_root(data_root)

    return AppSettings(
        database_url=_read_env("TURSO_DATABASE_URL", None),
        auth_token=_read_env("TURSO_AUTH_TOKEN"),
        paths=paths,
        runs_root=_resolve_runs_root(),
    )


def _resolve_runs_root() -> Path:
    runs_root_env = _read_env("STREAMLIT_RUNS_ROOT")
    if runs_root_env:
        return Path(runs_root_env).resolve()
    return Path("outputs/runs").resolve()


__all__ = ["AppSettings", "AppPaths", "get_settings"]
