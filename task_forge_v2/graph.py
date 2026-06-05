"""LangGraph orchestration for the task-forge v2 prototype."""

from __future__ import annotations

import json
import logging
import operator
import warnings
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

warnings.filterwarnings("ignore", message="The default value of `allowed_objects` will change.*")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

from .assessment import assess_task
from .config import get_project_paths
from .mimo import MimoStructuredClient
from .mining import mine_source_file
from .postprocess import build_reject_curated_task, sanitize_curated_task
from .prompt_contracts import (
    BOUNDARY_CRITIC_SYSTEM,
    CONSENSUS_SYSTEM,
    CURATOR_SYSTEM,
    GENERATOR_SYSTEM,
    HARNESS_CRITIC_SYSTEM,
    LEAKAGE_CRITIC_SYSTEM,
    SCOPE_CRITIC_SYSTEM,
    TRIAGE_SYSTEM,
)
from .runtime_metrics import get_runtime_telemetry
from .retrieval import build_retrieval_query, retrieve_context_pack
from .schemas import AnchorAssessment, ConsensusReview, CriticReview, CuratedTask, DraftTask, RetrievalHit, RunArtifact, SourceSummary, SourceTriage
from .triage_rules import normalize_source_triage
from .validators import validate_curated_task, validate_draft_task

_SHARED_CLIENT: MimoStructuredClient | None = None
CRITIC_SOURCE_CHAR_LIMIT = 9000


def _get_client() -> MimoStructuredClient:
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None:
        _SHARED_CLIENT = MimoStructuredClient()
    return _SHARED_CLIENT


def _read_source_excerpt(source_path: str) -> str:
    """Read the source file the critic is reviewing, truncated to keep prompt size bounded."""
    try:
        text = Path(source_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("critic_source_read_failed path=%s error=%s", source_path, exc)
        return ""
    if len(text) <= CRITIC_SOURCE_CHAR_LIMIT:
        return text
    head = text[: CRITIC_SOURCE_CHAR_LIMIT // 2]
    tail = text[-CRITIC_SOURCE_CHAR_LIMIT // 3 :]
    return f"{head}\n\n# ... source excerpt truncated ...\n\n{tail}"


class ForgeState(TypedDict, total=False):
    run_id: str
    pipeline_variant: str
    source_path: str
    source_summary: SourceSummary
    retrieval_query: str
    retrieval_hits: list[RetrievalHit]
    source_triage: SourceTriage
    draft_task: DraftTask
    critic_reviews: Annotated[list[CriticReview], operator.add]
    consensus_review: ConsensusReview
    curated_task: CuratedTask
    anchor_assessment: AnchorAssessment
    artifact_dir: str
    validator_issues: Annotated[list[str], operator.add]


def build_graph(enable_critics: bool = True) -> StateGraph:
    graph = StateGraph(ForgeState)
    graph.add_node("mine_source", mine_source_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("triage_source", triage_source_node)
    graph.add_node("curate_reject_seed", curate_reject_seed_node)
    graph.add_node("draft_task", draft_task_node)
    if enable_critics:
        graph.add_node("boundary_critic", boundary_critic_node)
        graph.add_node("leakage_critic", leakage_critic_node)
        graph.add_node("scope_critic", scope_critic_node)
        graph.add_node("harness_critic", harness_critic_node)
    else:
        graph.add_node("synthesize_review_no_critics", synthesize_review_no_critics_node)
    graph.add_node("synthesize_review", synthesize_review_node)
    graph.add_node("curate_task", curate_task_node)
    graph.add_node("assess_task", assess_task_node)
    graph.add_node("emit_artifacts", emit_artifacts_node)

    graph.add_edge(START, "mine_source")
    graph.add_edge("mine_source", "retrieve_context")
    graph.add_edge("retrieve_context", "triage_source")
    graph.add_conditional_edges(
        "triage_source",
        _route_after_triage,
        {"draft_task": "draft_task", "curate_reject_seed": "curate_reject_seed"},
    )
    if enable_critics:
        graph.add_edge("draft_task", "boundary_critic")
        graph.add_edge("draft_task", "leakage_critic")
        graph.add_edge("draft_task", "scope_critic")
        graph.add_edge("draft_task", "harness_critic")
        graph.add_edge("boundary_critic", "synthesize_review")
        graph.add_edge("leakage_critic", "synthesize_review")
        graph.add_edge("scope_critic", "synthesize_review")
        graph.add_edge("harness_critic", "synthesize_review")
    else:
        graph.add_edge("draft_task", "synthesize_review_no_critics")
        graph.add_edge("synthesize_review_no_critics", "curate_task")
    graph.add_edge("synthesize_review", "curate_task")
    graph.add_edge("curate_reject_seed", "assess_task")
    graph.add_edge("curate_task", "assess_task")
    graph.add_edge("assess_task", "emit_artifacts")
    graph.add_edge("emit_artifacts", END)
    return graph


def compile_graph(enable_critics: bool = True):
    return build_graph(enable_critics=enable_critics).compile(checkpointer=InMemorySaver())


def mine_source_node(state: ForgeState) -> ForgeState:
    source_path = Path(state["source_path"])
    summary = mine_source_file(source_path)
    return {"source_summary": summary}


def retrieve_context_node(state: ForgeState) -> ForgeState:
    summary = state["source_summary"]
    query = build_retrieval_query(summary)
    hits = retrieve_context_pack(summary)
    return {"retrieval_query": query, "retrieval_hits": hits}


def draft_task_node(state: ForgeState) -> ForgeState:
    client = _get_client()
    summary = state["source_summary"]
    hits = state.get("retrieval_hits", [])
    triage = state["source_triage"]
    prompt = (
        f"Source summary:\n{summary.model_dump_json(indent=2)}\n\n"
        f"Source triage:\n{triage.model_dump_json(indent=2)}\n\n"
        f"Retrieved benchmark context:\n{json.dumps([hit.model_dump() for hit in hits[:4]], indent=2)}\n\n"
        "Return one candidate benchmark task as JSON."
    )
    draft = client.invoke_model(DraftTask, GENERATOR_SYSTEM, prompt, label="draft_task")
    issues = validate_draft_task(draft)
    return {"draft_task": draft, "validator_issues": issues}


def triage_source_node(state: ForgeState) -> ForgeState:
    client = _get_client()
    summary = state["source_summary"]
    hits = state.get("retrieval_hits", [])
    prompt = (
        f"Source summary:\n{summary.model_dump_json(indent=2)}\n\n"
        f"Retrieved benchmark context:\n{json.dumps([hit.model_dump() for hit in hits[:4]], indent=2)}\n\n"
        "Decide whether this source is a strong benchmark seed before drafting a task."
    )
    triage = client.invoke_model(SourceTriage, TRIAGE_SYSTEM, prompt, label="triage_source")
    triage = normalize_source_triage(summary, triage)
    return {"source_triage": triage}


def _route_after_triage(state: ForgeState) -> str:
    if state["source_triage"].verdict == "reject":
        telemetry = get_runtime_telemetry()
        if telemetry is not None:
            telemetry.fast_path_taken = True
        return "curate_reject_seed"
    return "draft_task"


def boundary_critic_node(state: ForgeState) -> ForgeState:
    return _critic_node(state, "boundary", BOUNDARY_CRITIC_SYSTEM)


def leakage_critic_node(state: ForgeState) -> ForgeState:
    return _critic_node(state, "leakage", LEAKAGE_CRITIC_SYSTEM)


def scope_critic_node(state: ForgeState) -> ForgeState:
    return _critic_node(state, "scope", SCOPE_CRITIC_SYSTEM)


def harness_critic_node(state: ForgeState) -> ForgeState:
    return _critic_node(state, "harness", HARNESS_CRITIC_SYSTEM)


def _critic_node(state: ForgeState, critic_name: str, system_prompt: str) -> ForgeState:
    client = _get_client()
    draft = state["draft_task"]
    summary = state["source_summary"]
    source_excerpt = _read_source_excerpt(state["source_path"])
    prompt = (
        f"Source summary:\n{summary.model_dump_json(indent=2)}\n\n"
        f"Draft task:\n{draft.model_dump_json(indent=2)}\n\n"
        f"Visible Python source (reference only — audit the spec first):\n"
        f"```python\n{source_excerpt}\n```\n\n"
        f"You are the {critic_name} critic. Return one critic review JSON object."
    )
    review = client.invoke_model(CriticReview, system_prompt, prompt, label=f"critic:{critic_name}")
    if review.critic != critic_name:
        logger.warning(
            "critic_label_mismatch label=%s returned=%s overriding_to=%s",
            critic_name,
            review.critic,
            critic_name,
        )
        review = review.model_copy(update={"critic": critic_name})
    return {"critic_reviews": [review]}


def synthesize_review_node(state: ForgeState) -> ForgeState:
    client = _get_client()
    draft = state["draft_task"]
    reviews = state.get("critic_reviews", [])
    prompt = (
        f"Draft task:\n{draft.model_dump_json(indent=2)}\n\n"
        f"Panel reviews:\n{json.dumps([review.model_dump() for review in reviews], indent=2)}\n\n"
        "Synthesize them into one consensus review JSON object."
    )
    consensus = client.invoke_model(ConsensusReview, CONSENSUS_SYSTEM, prompt, label="synthesize_review")
    return {"consensus_review": consensus}


def synthesize_review_no_critics_node(state: ForgeState) -> ForgeState:
    summary = state["source_summary"]
    draft = state["draft_task"]
    consensus = ConsensusReview(
        consensus_id=f"{draft.draft_id}::no_critics",
        panel_verdict="pass",
        synthesis=(
            "Critic panel skipped for ablation baseline. Preserve the draft boundary unless postprocess detects "
            "shape drift or missing observability."
        ),
        agreed_changes=[
            "Keep the task execution-grounded and aligned to the visible workflow slice.",
            "Do not broaden the boundary beyond recoverable facts and explicit user policy.",
        ],
        unresolved_disagreements=[],
        must_fix=[],
        nice_to_have=["Later compare this baseline against the four-critic path."],
        revision_brief=(
            f"No critic panel for {summary.title or summary.source_path}. "
            "Curate directly from the draft while preserving single-boundary execution shape."
        ),
    )
    return {"consensus_review": consensus}


def curate_task_node(state: ForgeState) -> ForgeState:
    client = _get_client()
    draft = state["draft_task"]
    summary = state["source_summary"]
    consensus = state["consensus_review"]
    hits = state.get("retrieval_hits", [])
    prompt = (
        f"Source summary:\n{summary.model_dump_json(indent=2)}\n\n"
        f"Draft task:\n{draft.model_dump_json(indent=2)}\n\n"
        f"Consensus review:\n{consensus.model_dump_json(indent=2)}\n\n"
        f"Retrieved benchmark context:\n{json.dumps([hit.model_dump() for hit in hits[:4]], indent=2)}\n\n"
        "Produce one curated benchmark task JSON object."
    )
    curated = client.invoke_model(CuratedTask, CURATOR_SYSTEM, prompt, label="curate_task")
    curated = sanitize_curated_task(curated, summary, consensus, hits)
    issues = validate_curated_task(curated)
    return {"curated_task": curated, "validator_issues": issues}


def curate_reject_seed_node(state: ForgeState) -> ForgeState:
    curated = build_reject_curated_task(state["source_summary"], state["source_triage"])
    issues = validate_curated_task(curated)
    return {"curated_task": curated, "validator_issues": issues}


def assess_task_node(state: ForgeState) -> ForgeState:
    assessment = assess_task(
        source_summary=state["source_summary"],
        source_triage=state["source_triage"],
        curated_task=state["curated_task"],
        validator_issues=state.get("validator_issues", []),
    )
    return {"anchor_assessment": assessment}


def emit_artifacts_node(state: ForgeState) -> ForgeState:
    artifact_dir = _ensure_artifact_dir(state["run_id"])
    telemetry = get_runtime_telemetry()
    if telemetry is not None and telemetry.finished_at is None:
        telemetry.mark_finished()
    runtime_metadata = telemetry.to_dict() if telemetry is not None else {}
    bundle = RunArtifact(
        run_id=state["run_id"],
        repo_root=str(get_project_paths().root),
        pipeline_name="task_forge_v2_graph",
        retrieval_query=state.get("retrieval_query"),
        source_summaries=[state["source_summary"]],
        retrieval_hits=state.get("retrieval_hits", []),
        draft_tasks=[state["draft_task"]] if state.get("draft_task") else [],
        critic_reviews=state.get("critic_reviews", []),
        consensus_reviews=[state["consensus_review"]] if state.get("consensus_review") else [],
        source_triage=state["source_triage"],
        anchor_assessment=state["anchor_assessment"],
        curated_tasks=[state["curated_task"]],
        metadata={
            "pipeline_variant": state.get("pipeline_variant", "full_review"),
            "validator_issues": state.get("validator_issues", []),
            "runtime": runtime_metadata,
        },
    )
    payload_path = artifact_dir / "run_bundle.json"
    payload_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    (artifact_dir / "source_summary.json").write_text(
        state["source_summary"].model_dump_json(indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "retrieval_hits.json").write_text(
        json.dumps([item.model_dump() for item in state.get("retrieval_hits", [])], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if state.get("draft_task"):
        (artifact_dir / "draft_task.json").write_text(
            state["draft_task"].model_dump_json(indent=2),
            encoding="utf-8",
        )
    (artifact_dir / "source_triage.json").write_text(
        state["source_triage"].model_dump_json(indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "critic_reviews.json").write_text(
        json.dumps([item.model_dump() for item in state.get("critic_reviews", [])], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if state.get("consensus_review"):
        (artifact_dir / "consensus_review.json").write_text(
            state["consensus_review"].model_dump_json(indent=2),
            encoding="utf-8",
        )
    (artifact_dir / "curated_task.json").write_text(
        state["curated_task"].model_dump_json(indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "anchor_assessment.json").write_text(
        state["anchor_assessment"].model_dump_json(indent=2),
        encoding="utf-8",
    )
    report_path = artifact_dir / "summary.md"
    report_path.write_text(_render_summary(bundle), encoding="utf-8")
    return {"artifact_dir": str(artifact_dir)}


def _ensure_artifact_dir(run_id: str) -> Path:
    root = get_project_paths().artifacts
    root.mkdir(parents=True, exist_ok=True)
    artifact_dir = root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _render_summary(bundle: RunArtifact) -> str:
    source = bundle.source_summaries[0] if bundle.source_summaries else None
    curated = bundle.curated_tasks[0] if bundle.curated_tasks else None
    draft = bundle.draft_tasks[0] if bundle.draft_tasks else None
    consensus = bundle.consensus_reviews[0] if bundle.consensus_reviews else None
    return "\n".join(
        [
            f"# Task Forge Run {bundle.run_id}",
            "",
            f"- Source: `{source.source_path if source else 'unknown'}`",
            f"- Pipeline variant: {bundle.metadata.get('pipeline_variant', 'full_review')}",
            f"- Draft boundary: {draft.core_boundary if draft else 'skipped after triage reject'}",
            f"- Curated boundary: {curated.core_boundary if curated else 'unknown'}",
            f"- Source triage: {bundle.source_triage.verdict if bundle.source_triage else 'unknown'}",
            f"- Final cohort: {bundle.anchor_assessment.final_bucket if bundle.anchor_assessment else 'unknown'}",
            f"- Benchmark line: {(curated.benchmark_line if curated else None) or 'unknown'}",
            f"- Planning unit: {curated.planning_unit if curated else 'unknown'}",
            f"- Consensus verdict: {consensus.panel_verdict if consensus else 'skipped after triage reject'}",
            f"- LLM calls: {bundle.metadata.get('runtime', {}).get('llm_call_count', 'unknown')}",
            f"- Fast path taken: {bundle.metadata.get('runtime', {}).get('fast_path_taken', False)}",
            f"- Validator issues: {', '.join(bundle.metadata.get('validator_issues', [])) or 'none'}",
        ]
    ) + "\n"


def default_run_id() -> str:
    return datetime.now().strftime("task_forge_%Y%m%d_%H%M%S")
