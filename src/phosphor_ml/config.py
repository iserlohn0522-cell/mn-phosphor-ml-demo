from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    """Runtime configuration loaded from environment variables."""

    openalex_email: str
    openalex_api_key: str
    semantic_scholar_api_key: str
    crossref_mailto: str
    data_dir: Path
    log_level: str

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "Config":
        root = Path(project_root).resolve() if project_root else PROJECT_ROOT
        env_path = Path(env_file).resolve() if env_file else root / ".env"
        _load_env(env_path)

        data_dir_value = os.getenv("DATA_DIR", "data").strip() or "data"
        data_dir = Path(data_dir_value)
        if not data_dir.is_absolute():
            data_dir = root / data_dir

        return cls(
            openalex_email=os.getenv("OPENALEX_EMAIL", "").strip(),
            openalex_api_key=os.getenv("OPENALEX_API_KEY", "").strip(),
            semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip(),
            crossref_mailto=os.getenv("CROSSREF_MAILTO", "").strip(),
            data_dir=data_dir,
            log_level=(os.getenv("LOG_LEVEL", "INFO").strip() or "INFO").upper(),
        )


def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_fallback(env_path)
        return

    load_dotenv(env_path, override=False)


def _load_env_fallback(env_path: Path) -> None:
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
