"""Task Forge v2 prototype package."""

from .config import (
    DEFAULT_MODEL_DEFAULTS,
    DEFAULT_PROJECT_PATHS,
    ModelDefaults,
    ProjectPaths,
    ensure_within_repo,
    load_dotenv,
    get_model_defaults,
    get_project_paths,
    get_repo_root,
    resolve_path,
    resolve_source_path,
)

__all__ = [
    "DEFAULT_MODEL_DEFAULTS",
    "DEFAULT_PROJECT_PATHS",
    "ModelDefaults",
    "ProjectPaths",
    "ensure_within_repo",
    "load_dotenv",
    "get_model_defaults",
    "get_project_paths",
    "get_repo_root",
    "resolve_path",
    "resolve_source_path",
]

__version__ = "0.2.0"
