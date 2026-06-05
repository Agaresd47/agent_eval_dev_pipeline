"""Source mining helpers for task-forge v2."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

from .config import get_project_paths, get_repo_root
from .schemas import SourceSummary

PATH_LITERAL_RE = re.compile(r"(/data/[^\"'\s]+|[A-Za-z]:\\[^\"'\s]+)")


def iter_source_python_files(source_root: Path | None = None) -> Iterable[Path]:
    root = (source_root or get_project_paths().source_root).resolve()
    if not root.exists():
        return
    yield from sorted(root.rglob("*.py"))


def iter_dirty_python_files() -> Iterable[Path]:
    """Backward-compatible alias for older tests and scripts."""

    yield from iter_source_python_files()


def mine_source_file(path: Path) -> SourceSummary:
    path = path.resolve()
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = _safe_parse(text, path)
    import_names = _extract_imports(tree)
    function_names = _extract_functions(tree)
    path_literals = _extract_paths(text)
    risks = _extract_risks(text)
    recoverable = _recoverable_facts(path.name, risks)
    user_policy = _user_policy_hints(path.name)
    key_facts = [
        f"imports: {', '.join(import_names[:5])}" if import_names else "imports: none",
        f"functions: {', '.join(function_names[:5])}" if function_names else "functions: none",
    ]
    relative_source = _display_source_path(path)
    return SourceSummary(
        artifact_id=f"source::{path.stem}",
        source_ids=[relative_source],
        tags=risks,
        source_path=relative_source,
        source_kind="python_script",
        title=path.name,
        summary=_guess_summary(path.name, risks),
        key_facts=key_facts,
        recoverable_facts=recoverable,
        user_only_policies=user_policy,
        risks=risks,
        open_questions=[],
        linked_paths=path_literals[:8],
    )


def _display_source_path(path: Path) -> str:
    root = get_repo_root().resolve()
    source_root = get_project_paths().source_root
    for base in (source_root, root):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def _safe_parse(text: str, path: Path) -> ast.AST | None:
    try:
        return ast.parse(text, filename=str(path))
    except SyntaxError:
        return None


def _extract_imports(tree: ast.AST | None) -> list[str]:
    if tree is None:
        return []
    items: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            items.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                items.append(node.module)
    return sorted(set(items))


def _extract_functions(tree: ast.AST | None) -> list[str]:
    if tree is None:
        return []
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


def _extract_paths(text: str) -> list[str]:
    values = PATH_LITERAL_RE.findall(text)
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _extract_risks(text: str) -> list[str]:
    lower = text.lower()
    tags: list[str] = []
    if "shutil.move" in lower or "os.rename" in lower:
        tags.append("mutation_move")
    if "tarfile" in lower or "os.remove(" in lower or ".part_" in lower or "split_file(" in lower:
        tags.append("archive_cleanup")
    if "symlink" in lower:
        tags.append("symlink_layout")
    if "dataset.json" in lower or "json.dump" in lower:
        tags.append("schema_generation")
    if "pool(" in lower or "multiprocessing" in lower:
        tags.append("parallel_ops")
    if "mask" in lower or "nii.gz" in lower or "totalseg" in lower:
        tags.append("medical_imaging")
    return tags


def _recoverable_facts(name: str, risks: list[str]) -> list[str]:
    facts = []
    if "archive_cleanup" in risks:
        facts.extend(
            [
                "Archive output names and split-part naming rules are recoverable from code and emitted filenames.",
                "Whether the original archive still exists after splitting is recoverable by inspection.",
            ]
        )
    if "schema_generation" in risks:
        facts.extend(
            [
                "Directory names and file suffix conventions can be inspected from the workspace.",
                "Case-to-label pairing rules are recoverable from filenames and code.",
            ]
        )
    if "mutation_move" in risks:
        facts.extend(
            [
                "Current source inventory and destination conflicts are recoverable by inspection.",
                "Naming rules for pair matching can be recovered from filenames and code constants.",
            ]
        )
    if "symlink_layout" in risks:
        facts.extend(
            [
                "Split membership files and source path structure are inspectable.",
                "Broken-link risk can be checked from source existence and target layout.",
            ]
        )
    if not facts:
        facts.append("Core workflow facts should be recoverable from local code and file inspection.")
    return facts


def _user_policy_hints(name: str) -> list[str]:
    lower_name = name.lower()
    if "compress" in lower_name or "backup" in lower_name or "split" in lower_name:
        return [
            "Whether the original archive may be deleted before every expected split part is verified.",
            "Whether an incomplete split should block cleanup or leave the archive in place.",
        ]
    if "finished" in lower_name:
        return [
            "Whether missing or partial pairs should be skipped or treated as hard failures.",
            "Whether execution is approved after a dry-run ledger.",
        ]
    if "totalseg" in lower_name:
        return [
            "Whether the keep-list is fixed policy or should be questioned before mutation.",
            "How to handle duplicates or conflicting targets.",
        ]
    if "link" in lower_name:
        return [
            "Whether split assignments are fixed inputs or need recovery from files.",
            "How missing source files should be reported versus blocked.",
        ]
    if "json" in lower_name or "unet" in lower_name:
        return [
            "Whether defaults like modality or labels are fixed or require clarification.",
            "How strictly missing directories should fail the run.",
        ]
    return ["Any non-recoverable operational policy should remain explicitly user-provided."]


def _guess_summary(name: str, risks: list[str]) -> str:
    lower = name.lower()
    if "archive_cleanup" in risks or "compress" in lower:
        return "Compress a source folder, split the resulting archive, and only clean up the original archive when the split output is verified."
    if "finished" in lower:
        return "Promote only complete image-mask pairs into a finished area while avoiding partial cases."
    if "totalseg" in lower:
        return "Move segmentation artifacts while preserving an exact keep-list and surfacing file conflicts."
    if "link" in lower:
        return "Build a dataset symlink layout while preserving split membership and missing-file reporting."
    if "json" in lower or "unet" in lower:
        return "Generate dataset metadata without drifting away from the intended schema."
    return f"Mine risky workflow behavior from {name} with tags: {', '.join(risks) or 'none'}."
