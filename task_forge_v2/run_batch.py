"""Batch entrypoint for running task-forge v2 across multiple source files."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from .assessment import assess_task
from .config import get_project_paths
from .graph import compile_graph, default_run_id
from .mining import iter_source_python_files
from .runtime_metrics import close_runtime_telemetry, start_runtime_telemetry

NOISY_NAME_RE = re.compile(r"^\d+\.py$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run task-forge v2 on multiple source files.")
    parser.add_argument("--limit", type=int, default=4, help="How many Python files to process.")
    parser.add_argument("--run-prefix", default=default_run_id(), help="Prefix for generated run ids.")
    parser.add_argument("--thread-prefix", default="task-forge-batch", help="Prefix for LangGraph thread ids.")
    parser.add_argument("--source-root", default=None, help="Root directory to scan for Python files.")
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
    if args.freeze_retrieval_corpus:
        os.environ["TASK_FORGE_FREEZE_RETRIEVAL_CORPUS"] = "1"

    enable_critics = args.pipeline_variant != "no_critics"
    graph = compile_graph(enable_critics=enable_critics)
    repo_root = get_project_paths().root
    source_root = Path(args.source_root).expanduser().resolve() if args.source_root else get_project_paths().source_root
    files = [path for path in iter_source_python_files(source_root) if _is_candidate(path)][: args.limit]
    if not files:
        raise SystemExit(f"No Python files found under {source_root}.")

    print(f"batch_size: {len(files)}")
    print(f"source_root: {source_root}")
    rows: list[dict[str, object]] = []
    for idx, path in enumerate(files, start=1):
        run_id = f"{args.run_prefix}_{idx:02d}"
        thread_id = f"{args.thread_prefix}-{idx:02d}"
        start_runtime_telemetry(run_id, args.pipeline_variant)
        try:
            final_state = graph.invoke(
                {
                    "run_id": run_id,
                    "pipeline_variant": args.pipeline_variant,
                    "source_path": str(path.resolve()),
                    "critic_reviews": [],
                    "validator_issues": [],
                },
                config={"configurable": {"thread_id": thread_id}},
            )
        finally:
            close_runtime_telemetry()
        resolved_path = path.resolve()
        try:
            rel_source = resolved_path.relative_to(source_root)
        except ValueError:
            try:
                rel_source = resolved_path.relative_to(repo_root.resolve())
            except ValueError:
                rel_source = resolved_path
        curated = final_state["curated_task"]
        triage = final_state["source_triage"]
        issues = final_state.get("validator_issues", [])
        assessment = final_state.get("anchor_assessment") or assess_task(
            source_summary=final_state["source_summary"],
            source_triage=triage,
            curated_task=curated,
            validator_issues=issues,
        )
        cohort_bucket = assessment.final_bucket
        print(f"[{idx}/{len(files)}] {rel_source}")
        print(f"  artifact_dir: {final_state.get('artifact_dir')}")
        print(f"  pipeline_variant: {args.pipeline_variant}")
        print(f"  source_triage: {triage.verdict}")
        print(f"  benchmark_line: {curated.benchmark_line}")
        print(f"  planning_unit: {curated.planning_unit}")
        print(f"  cohort_bucket: {cohort_bucket}")
        print(f"  anchor_reason_codes: {', '.join(assessment.reason_codes) if assessment.reason_codes else 'none'}")
        print(f"  issues: {', '.join(issues) if issues else 'none'}")
        rows.append(
            {
                "source_path": str(rel_source),
                "artifact_dir": final_state.get("artifact_dir"),
                "pipeline_variant": args.pipeline_variant,
                "source_triage": triage.model_dump(),
                "anchor_assessment": assessment.model_dump(),
                "benchmark_line": curated.benchmark_line,
                "planning_unit": curated.planning_unit,
                "cohort_bucket": cohort_bucket,
                "validator_issues": issues,
            }
        )
    _write_batch_report(args.run_prefix, rows)


def _is_candidate(path: Path) -> bool:
    if NOISY_NAME_RE.match(path.name):
        return False
    try:
        return path.stat().st_size >= 400
    except OSError:
        return False


def _write_batch_report(run_prefix: str, rows: list[dict[str, object]]) -> None:
    artifact_root = get_project_paths().artifacts
    report_dir = artifact_root / f"{run_prefix}_batch_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "batch_report.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [f"# Batch Report {run_prefix}", ""]
    for row in rows:
        triage = row["source_triage"]
        lines.append(f"- `{row['source_path']}`: {row['cohort_bucket']} ({triage['verdict']}, {triage['confidence']})")
    (report_dir / "batch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
