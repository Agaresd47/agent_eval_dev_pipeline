"""CLI entrypoint for the task-forge v2 graph."""

from __future__ import annotations

import argparse
import os

from .config import resolve_source_path
from .graph import compile_graph, default_run_id
from .runtime_metrics import close_runtime_telemetry, start_runtime_telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run task-forge v2 on one source Python file.")
    parser.add_argument("--source", required=True, help="Path to one source Python file.")
    parser.add_argument("--run-id", default=default_run_id(), help="Optional run identifier.")
    parser.add_argument("--thread-id", default="task-forge-thread", help="LangGraph thread identifier.")
    parser.add_argument(
        "--pipeline-variant",
        choices=["full_review", "no_critics"],
        default="full_review",
        help="Choose the default four-critic path or a no-critics ablation baseline.",
    )
    parser.add_argument(
        "--freeze-retrieval-corpus",
        action="store_true",
        help="Freeze the retrieval corpus snapshot for this process so repeated runs reuse one index signature.",
    )
    args = parser.parse_args()

    source_path = resolve_source_path(args.source)
    if args.freeze_retrieval_corpus:
        os.environ["TASK_FORGE_FREEZE_RETRIEVAL_CORPUS"] = "1"

    enable_critics = args.pipeline_variant != "no_critics"
    graph = compile_graph(enable_critics=enable_critics)
    config = {"configurable": {"thread_id": args.thread_id}}
    start_runtime_telemetry(args.run_id, args.pipeline_variant)
    try:
        final_state = graph.invoke(
            {
                "run_id": args.run_id,
                "pipeline_variant": args.pipeline_variant,
                "source_path": str(source_path),
                "critic_reviews": [],
                "validator_issues": [],
            },
            config=config,
        )
    finally:
        close_runtime_telemetry()
    print(f"run_id: {args.run_id}")
    print(f"source: {source_path}")
    print(f"artifact_dir: {final_state.get('artifact_dir')}")
    print(f"pipeline_variant: {args.pipeline_variant}")
    print(f"source_triage: {final_state['source_triage'].verdict}")
    draft = final_state.get("draft_task")
    print(f"draft_boundary: {draft.core_boundary if draft else 'skipped after triage reject'}")
    print(f"curated_boundary: {final_state['curated_task'].core_boundary}")
    print(f"benchmark_line: {final_state['curated_task'].benchmark_line}")
    print(f"planning_unit: {final_state['curated_task'].planning_unit}")
    assessment = final_state.get("anchor_assessment")
    if assessment:
        print(f"final_cohort: {assessment.final_bucket}")
    issues = final_state.get("validator_issues", [])
    print(f"validator_issues: {', '.join(issues) if issues else 'none'}")


if __name__ == "__main__":
    main()
