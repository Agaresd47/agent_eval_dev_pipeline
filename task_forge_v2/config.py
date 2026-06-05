"""Configuration helpers for the standalone task-forge prototype."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

ENV_PREFIX: Final[str] = "TASK_FORGE_"


def get_repo_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[1]


def resolve_path(*parts: str | os.PathLike[str]) -> Path:
    """Resolve a path relative to the repository root."""

    return get_repo_root().joinpath(*(Path(part) for part in parts))


@dataclass(frozen=True, slots=True)
class ModelDefaults:
    """Model settings used by the planner, retriever, and worker roles."""

    provider: str = "openai_compatible"
    chat_model: str = "gpt-4.1-mini"
    reasoning_model: str = "gpt-4.1"
    embedding_model: str = "text-embedding-3-small"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.2
    max_output_tokens: int = 12000

    @classmethod
    def from_env(cls) -> "ModelDefaults":
        """Build defaults from `TASK_FORGE_*` environment variables."""

        defaults = cls()
        return cls(
            provider=os.getenv(f"{ENV_PREFIX}MODEL_PROVIDER", defaults.provider),
            chat_model=os.getenv(f"{ENV_PREFIX}CHAT_MODEL", defaults.chat_model),
            reasoning_model=os.getenv(f"{ENV_PREFIX}REASONING_MODEL", defaults.reasoning_model),
            embedding_model=os.getenv(f"{ENV_PREFIX}EMBEDDING_MODEL", defaults.embedding_model),
            base_url=os.getenv(f"{ENV_PREFIX}BASE_URL", defaults.base_url),
            temperature=float(os.getenv(f"{ENV_PREFIX}TEMPERATURE", str(defaults.temperature))),
            max_output_tokens=int(os.getenv(f"{ENV_PREFIX}MAX_OUTPUT_TOKENS", str(defaults.max_output_tokens))),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a plain dictionary for graph or client configuration."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Common repository paths used by the prototype."""

    root: Path
    package: Path
    knowledge: Path
    artifacts: Path
    cache: Path
    examples: Path
    source_root: Path

    @classmethod
    def from_repo_root(cls) -> "ProjectPaths":
        root = get_repo_root()
        source_root_env = os.getenv(f"{ENV_PREFIX}SOURCE_ROOT", "").strip()
        if source_root_env:
            source_root = Path(source_root_env).expanduser().resolve()
        else:
            source_root = (root / "examples" / "source").resolve()
        return cls(
            root=root,
            package=root / "task_forge_v2",
            knowledge=root / "knowledge",
            artifacts=root / "artifacts",
            cache=root / ".cache",
            examples=root / "examples",
            source_root=source_root,
        )


def get_project_paths() -> ProjectPaths:
    """Return the current repository path map."""

    return ProjectPaths.from_repo_root()


def get_model_defaults() -> ModelDefaults:
    """Return model defaults with environment overrides applied."""

    return ModelDefaults.from_env()


def ensure_within_repo(path: str | Path) -> Path:
    """Guard against accidental path escapes outside the repository root."""

    candidate = Path(path).expanduser().resolve()
    root = get_repo_root().resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path {candidate} is outside repo root {root}")
    return candidate


def load_dotenv(path: Path | None = None) -> None:
    """Minimal in-process .env loader.

    Kept hand-rolled on purpose to avoid pulling python-dotenv as a hard dependency
    for what is essentially a single ``KEY=value`` loop. Uses ``setdefault`` so an
    already-set process env wins over the file.
    """

    env_path = path or (get_repo_root() / ".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def resolve_source_path(value: str | Path) -> Path:
    """Resolve a source file path from absolute, source-root-relative, or repo-relative input."""

    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    paths = get_project_paths()
    source_candidate = (paths.source_root / raw).resolve()
    if source_candidate.exists():
        return source_candidate
    return (paths.root / raw).resolve()


DEFAULT_MODEL_DEFAULTS: Final[ModelDefaults] = ModelDefaults()
DEFAULT_PROJECT_PATHS: Final[ProjectPaths] = ProjectPaths.from_repo_root()
