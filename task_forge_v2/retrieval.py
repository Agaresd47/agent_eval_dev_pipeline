"""LlamaIndex-backed retrieval memory for task-forge v2."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter

from .config import get_project_paths, get_repo_root
from .runtime_metrics import get_runtime_telemetry
from .schemas import RetrievalHit, SourceSummary

if TYPE_CHECKING:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

_EMBEDDING: Any | None = None
_INDEX: VectorStoreIndex | None = None
_INDEX_SIGNATURE: str | None = None
_FROZEN_SIGNATURE: str | None = None


class BenchmarkMemory:
    def __init__(self) -> None:
        self.root = get_repo_root()
        self.splitter = SentenceSplitter(chunk_size=700, chunk_overlap=120)
        self.embedding = _get_embedding()
        self.index = _get_index(self.splitter, self.embedding)

    def retrieve(self, query: str, top_k: int = 6) -> list[RetrievalHit]:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
        results: list[RetrievalHit] = []
        for idx, node in enumerate(nodes, start=1):
            metadata = getattr(node.node, "metadata", {}) or {}
            results.append(
                RetrievalHit(
                    query=query,
                    source_id=str(metadata.get("source_id", metadata.get("file_path", f"chunk-{idx}"))),
                    source_path=str(metadata.get("file_path", "unknown")),
                    chunk_id=str(metadata.get("chunk_id", idx)),
                    rank=idx,
                    score=float(getattr(node, "score", 0.0) or 0.0),
                    excerpt=node.text[:500],
                    match_reasons=[str(metadata.get("memory_type", "benchmark_rule"))],
                    highlights=[
                        f"family:{metadata.get('task_family', 'generic')}",
                        f"shape:{metadata.get('task_shape', 'unknown')}",
                    ],
                )
            )
        return results


def build_retrieval_query(summary: SourceSummary) -> str:
    return (
        f"Need benchmark authoring rules for: {summary.summary}\n"
        f"Risks: {', '.join(summary.risks)}\n"
        f"Recoverable facts: {', '.join(summary.recoverable_facts[:2])}"
    )


def retrieve_context_pack(summary: SourceSummary, top_k: int = 8) -> list[RetrievalHit]:
    memory = BenchmarkMemory()
    query_routes = [
        (
            f"Find benchmark authoring rules for a task with summary: {summary.summary}. "
            f"Focus on single boundary, recoverable facts, and user-only policy. Risks: {', '.join(summary.risks)}",
            "authoring_rule",
        ),
        (
            f"Find similar workflow tasks for: {summary.summary}. "
            f"Source kind: {summary.source_kind}. Risks: {', '.join(summary.risks)}",
            "workflow_analog",
        ),
        (
            f"Find benchmark failure modes for: {summary.summary}. "
            "Focus on leakage, scope creep, and harness overreach.",
            "failure_mode",
        ),
    ]
    if "archive_cleanup" in summary.risks:
        query_routes.insert(
            1,
            (
                "Find execution-grounded workflow tasks about compressing data, splitting an archive into parts, "
                "and deciding when the original archive may be safely deleted. Prefer mutation or dry-run ledger tasks, "
                "not static code review tasks.",
                "workflow_analog",
            ),
        )
    merged: dict[tuple[str, str], RetrievalHit] = {}
    for query, reason in query_routes:
        for hit in memory.retrieve(query, top_k=top_k):
            if not _keep_hit(summary, hit, reason):
                continue
            key = (hit.source_path, hit.chunk_id or str(hit.rank))
            existing = merged.get(key)
            if existing is None or hit.score > existing.score:
                merged[key] = hit.model_copy(update={"match_reasons": [reason]})
            elif reason not in existing.match_reasons:
                existing.match_reasons.append(reason)
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:top_k]


def _keep_hit(summary: SourceSummary, hit: RetrievalHit, reason: str) -> bool:
    excerpt = (hit.excerpt or "").lower()
    tags = set(hit.highlights or [])
    family = _tag_value(tags, "family:")
    shape = _tag_value(tags, "shape:")
    if reason == "workflow_analog" and '"reject_seed"' in excerpt:
        return False
    if reason == "workflow_analog" and family == "reject_seed":
        return False
    if "archive_cleanup" in summary.risks and reason == "workflow_analog":
        if shape == "static_review" or '"static_code_review_shape"' in excerpt:
            return False
        if family not in {"archive_cleanup", "generic", "benchmark_rule", "unknown"}:
            return False
        if any(marker in excerpt for marker in ("unsafe file deletion", "error handling gap", "code quality issue")):
            return False
    if "mutation_move" in summary.risks and reason == "workflow_analog":
        if shape == "static_review":
            return False
        if family not in {"mutation_move", "generic", "benchmark_rule", "unknown"}:
            return False
    return True


def _get_embedding():
    global _EMBEDDING
    if _EMBEDDING is None:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            _EMBEDDING = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDING


def _get_index(splitter: SentenceSplitter, embedding) -> VectorStoreIndex:
    global _INDEX, _INDEX_SIGNATURE
    signature = _effective_corpus_signature()
    if _INDEX is not None and _INDEX_SIGNATURE == signature:
        telemetry = get_runtime_telemetry()
        if telemetry is not None:
            telemetry.record_retrieval_cache(source="memory", hit=True)
        return _INDEX

    cache_dir = _index_cache_dir()
    manifest_path = cache_dir / "manifest.json"
    if cache_dir.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("signature") == signature:
                sink = io.StringIO()
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    storage_context = StorageContext.from_defaults(persist_dir=str(cache_dir))
                    _INDEX = load_index_from_storage(storage_context, embed_model=embedding)
                _INDEX_SIGNATURE = signature
                telemetry = get_runtime_telemetry()
                if telemetry is not None:
                    telemetry.record_retrieval_cache(source="disk", hit=True)
                return _INDEX
        except Exception:
            pass

    docs = list(_iter_documents())
    nodes = splitter.get_nodes_from_documents(docs)
    for idx, node in enumerate(nodes):
        node.metadata["chunk_id"] = f"chunk-{idx}"
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        _INDEX = VectorStoreIndex(nodes, embed_model=embedding)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _INDEX.storage_context.persist(persist_dir=str(cache_dir))
    manifest_path.write_text(json.dumps({"signature": signature}, indent=2), encoding="utf-8")
    _INDEX_SIGNATURE = signature
    telemetry = get_runtime_telemetry()
    if telemetry is not None:
        telemetry.record_retrieval_cache(source="rebuild", hit=False)
    return _INDEX


def _iter_documents() -> Iterable[Document]:
    paths = get_project_paths()
    yield from _iter_knowledge_documents(paths.knowledge)
    yield from _iter_curated_task_documents(paths.artifacts, memory_type="run_artifact")


def _iter_knowledge_documents(root: Path) -> Iterable[Document]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if _looks_mojibake(text):
            continue
        rel_path = str(path.relative_to(get_repo_root()))
        yield Document(
            text=text,
            metadata={
                "file_path": rel_path,
                "source_id": rel_path,
                "memory_type": "benchmark_rule",
                "task_family": "benchmark_rule",
                "task_shape": "benchmark_rule",
            },
        )


def _iter_curated_task_documents(root: Path, memory_type: str) -> Iterable[Document]:
    if not root.exists():
        return
    for path in sorted(root.rglob("curated_tasks.json")):
        yield from _documents_from_curated_task_file(path, memory_type)
    for path in sorted(root.rglob("run_bundle.json")):
        yield from _documents_from_run_bundle(path, memory_type)


def _documents_from_curated_task_file(path: Path, memory_type: str) -> Iterable[Document]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return
    if not isinstance(payload, list):
        return
    for idx, record in enumerate(payload):
        curated = record.get("curated_task", {})
        if not curated:
            continue
        task_family = _task_family_from_curated(curated)
        task_shape = _task_shape_from_curated(curated)
        yield Document(
            text=json.dumps(curated, ensure_ascii=False),
            metadata={
                "file_path": str(path.relative_to(get_repo_root())),
                "source_id": f"{path.name}:{idx}",
                "memory_type": memory_type,
                "task_family": task_family,
                "task_shape": task_shape,
            },
        )


def _documents_from_run_bundle(path: Path, memory_type: str) -> Iterable[Document]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return
    tasks = payload.get("curated_tasks", [])
    for idx, curated in enumerate(tasks):
        task_family = _task_family_from_curated(curated)
        task_shape = _task_shape_from_curated(curated)
        yield Document(
            text=json.dumps(curated, ensure_ascii=False),
            metadata={
                "file_path": str(path.relative_to(get_repo_root())),
                "source_id": f"{path.name}:{idx}",
                "memory_type": memory_type,
                "task_family": task_family,
                "task_shape": task_shape,
            },
        )


def _looks_mojibake(text: str) -> bool:
    return text.count("\ufffd") > 10


def _task_family_from_curated(curated: dict[str, object]) -> str:
    quality_flags = {str(item) for item in curated.get("quality_flags", []) or []}
    if "reject_seed" in quality_flags:
        return "reject_seed"
    text = " ".join(
        [
            str(curated.get("title", "")).lower(),
            str(curated.get("benchmark_goal", "")).lower(),
            str(curated.get("core_boundary", "")).lower(),
            str(curated.get("planning_unit", "")).lower(),
            str(curated.get("problem_statement", "")).lower(),
        ]
    )
    if any(word in text for word in ("archive", "chunk", "cleanup", "split output")):
        return "archive_cleanup"
    if any(word in text for word in ("image-mask", "pair", "partial pair", "finished destinations")):
        return "mutation_move"
    if any(word in text for word in ("symlink", "split-member", "link decision")):
        return "symlink_layout"
    if any(word in text for word in ("schema", "dataset-entry", "dataset metadata")):
        return "schema_generation"
    if "t2_handoff_style" in text or any(word in text for word in ("planner", "worker", "handoff")):
        return "handoff"
    return "generic"


def _task_shape_from_curated(curated: dict[str, object]) -> str:
    quality_flags = {str(item) for item in curated.get("quality_flags", []) or []}
    if "reject_seed" in quality_flags:
        return "reject_seed"
    if "static_code_review_shape" in quality_flags:
        return "static_review"
    if "workflow_grounded_shape" in quality_flags:
        return "workflow_grounded"
    return "unknown"


def _tag_value(tags: set[str], prefix: str) -> str:
    for tag in tags:
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return "unknown"


def _index_cache_dir() -> Path:
    return get_project_paths().cache / "retrieval_index"


def _corpus_signature() -> str:
    root = get_repo_root()
    paths = get_project_paths()
    digest = hashlib.sha256()
    if paths.knowledge.exists():
        for path in sorted(paths.knowledge.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            digest.update(f"{path.relative_to(root)}|{stat.st_mtime_ns}|{stat.st_size}\n".encode("utf-8"))
    if paths.artifacts.exists():
        for path in sorted(paths.artifacts.rglob("curated_tasks.json")):
            stat = path.stat()
            digest.update(f"{path.relative_to(root)}|{stat.st_mtime_ns}|{stat.st_size}\n".encode("utf-8"))
        for path in sorted(paths.artifacts.rglob("run_bundle.json")):
            stat = path.stat()
            digest.update(f"{path.relative_to(root)}|{stat.st_mtime_ns}|{stat.st_size}\n".encode("utf-8"))
    return digest.hexdigest()


def _effective_corpus_signature() -> str:
    global _FROZEN_SIGNATURE
    signature = _corpus_signature()
    freeze_flag = os.getenv("TASK_FORGE_FREEZE_RETRIEVAL_CORPUS", "").strip().lower()
    if freeze_flag not in {"1", "true", "yes", "on"}:
        _FROZEN_SIGNATURE = None
        return signature
    if _FROZEN_SIGNATURE is None:
        _FROZEN_SIGNATURE = signature
    return _FROZEN_SIGNATURE
