"""Self-consistency dry-run: testers read curated_task.json, judge scores their plans.

Four tester models (Kimi K2.5, GLM 4.7 Flash, DeepSeek V4 Flash, MiMo V2.5 Pro) read
each curated task plus the visible source, and produce an execution plan as strict
JSON. An independent judge (`gpt-5.4-mini`) scores each candidate against the same
rubric and labels it `pass / partial / fail`.

This step is *not* part of the production pipeline. It exists to sanity-check that
generated tasks are self-consistent — i.e. a downstream agent that only sees the
plan JSON plus the source can stay inside the boundary, stay grounded, and produce
an execution-ready answer. MiMo is the same family as the drafting model, so its
column is kept as a reference and excluded from the cross-vendor roll-up.

All calls go through the OpenAI-compatible endpoint configured by
`TASK_FORGE_BASE_URL` / `TASK_FORGE_API_KEY` (same wiring as the main pipeline).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task_forge_v2.config import get_project_paths, load_dotenv  # noqa: E402
from task_forge_v2.mimo import MimoStructuredClient  # noqa: E402

TESTER_MODELS: dict[str, str] = {
    "kimi_k2_5": "moonshotai.kimi-k2.5",
    "glm_4_7_flash": "zai.glm-4.7-flash",
    "deepseek_v4_flash": "deepseek_v4_flash",
    "mimo_v2_5_pro": "mimo_v2_5_pro",
}
DRAFTER_FAMILY_LABELS: set[str] = {"mimo_v2_5_pro"}
CROSS_VENDOR_LABELS: list[str] = [
    label for label in TESTER_MODELS if label not in DRAFTER_FAMILY_LABELS
]
JUDGE_MODEL = "gpt-5.4-mini"
SOURCE_CHAR_LIMIT = 9000
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")

RUNNER_SYSTEM = """
You are a benchmark agent answering a generated Task Forge task.

Rules:
- Stay inside the exact task boundary.
- Use only the visible source evidence, listed recoverable facts, and explicitly named user-only policy.
- Do not write code.
- Do not invent filesystem state, hidden policy, or execution results.
- If a policy is user-owned and unresolved, keep it explicit instead of silently choosing one.
- Return valid JSON and nothing else.

Return this schema:
{
  "boundary_restatement": "string",
  "recoverable_facts_used": ["string"],
  "user_policy_handling": ["string"],
  "execution_plan": ["string"],
  "observable_checks": ["string"],
  "final_answer": "string",
  "unsupported_assumptions": ["string"]
}
""".strip()

JUDGE_SYSTEM = """
You are a strict evaluator for generated benchmark-task answers.
Evaluate the candidate answer, not the task design itself.

Scoring dimensions:
- boundary_discipline_score: 0-3
- grounding_score: 0-3
- execution_readiness_score: 0-2
- policy_handling_score: 0-2
- unsupported_assumption_penalty: 0-2

Interpretation:
- pass: the answer stays in boundary, stays grounded, preserves user-owned policy correctly, and gives an execution-ready or decision-ready response.
- partial: the answer is directionally useful but misses a material boundary, grounding, observability, or policy detail.
- fail: the answer drifts, invents policy, collapses the boundary, or cannot support the task.

Return valid JSON and nothing else:
{
  "boundary_discipline_score": 0,
  "grounding_score": 0,
  "execution_readiness_score": 0,
  "policy_handling_score": 0,
  "unsupported_assumption_penalty": 0,
  "total_score": 0,
  "outcome": "pass",
  "pass_reasons": ["string"],
  "misses": ["string"],
  "concise_rationale": "string"
}
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-consistency dry-run for generated Task Forge tasks."
    )
    parser.add_argument(
        "--artifacts",
        nargs="+",
        required=True,
        help="Artifact ids under artifacts/ (each must contain run_bundle.json).",
    )
    parser.add_argument(
        "--run-prefix",
        required=True,
        help="Prefix for report_data outputs.",
    )
    args = parser.parse_args()

    load_dotenv()
    paths = get_project_paths()
    report_data_dir = paths.root / "report_data"
    report_data_dir.mkdir(parents=True, exist_ok=True)
    rows_jsonl = report_data_dir / f"{args.run_prefix}_dryrun_plan_check_rows.jsonl"

    rows: list[dict[str, Any]] = _load_existing_rows(rows_jsonl)
    completed_pairs = {(str(row["artifact_id"]), str(row["model_label"])) for row in rows}

    tester_clients = {label: MimoStructuredClient(model=model_id) for label, model_id in TESTER_MODELS.items()}
    judge_client = MimoStructuredClient(model=JUDGE_MODEL)

    for artifact_id in args.artifacts:
        bundle = _load_artifact_bundle(paths.artifacts, artifact_id)
        source_excerpt = _make_source_excerpt(_read_source_text(paths.root, bundle))
        for label, model_id in TESTER_MODELS.items():
            if (artifact_id, label) in completed_pairs:
                print(f"[skip] {artifact_id} / {label} already present in {rows_jsonl.name}")
                continue
            runner_prompt = _build_runner_prompt(bundle, source_excerpt)
            candidate_text = tester_clients[label].invoke_text(
                RUNNER_SYSTEM, runner_prompt, label=f"dryrun_runner:{label}"
            )
            candidate = _extract_json_payload(candidate_text)
            judge_prompt = _build_judge_prompt(bundle, source_excerpt, label, candidate)
            judge_text = judge_client.invoke_text(
                JUDGE_SYSTEM, judge_prompt, label=f"dryrun_judge:{label}"
            )
            judged = _extract_json_payload(judge_text)
            row = _build_row(
                artifact_id=artifact_id,
                bundle=bundle,
                model_label=label,
                model_id=model_id,
                candidate=candidate,
                judged=judged,
            )
            rows.append(row)
            _append_jsonl(rows_jsonl, row)
            print(
                f"[{artifact_id}] {label}: score={row['total_score']} outcome={row['outcome']}"
            )

    payload = _aggregate_payload(args.run_prefix, rows)
    _write_outputs(report_data_dir, args.run_prefix, payload)


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_artifact_bundle(artifacts_root: Path, artifact_id: str) -> dict[str, Any]:
    bundle_path = artifacts_root / artifact_id / "run_bundle.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing run_bundle.json for artifact {artifact_id} at {bundle_path}")
    return json.loads(bundle_path.read_text(encoding="utf-8"))


def _read_source_text(repo_root: Path, bundle: dict[str, Any]) -> str:
    source_path = bundle["source_summaries"][0]["source_path"]
    candidate = Path(source_path)
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    return candidate.read_text(encoding="utf-8", errors="replace")


def _make_source_excerpt(source_text: str) -> str:
    if len(source_text) <= SOURCE_CHAR_LIMIT:
        return source_text
    head = source_text[: SOURCE_CHAR_LIMIT // 2]
    tail = source_text[-SOURCE_CHAR_LIMIT // 3 :]
    return f"{head}\n\n# ... source excerpt truncated ...\n\n{tail}"


def _extract_json_payload(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        cleaned = TRAILING_COMMA_RE.sub(r"\1", stripped)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(cleaned[start : end + 1])
            raise


def _build_runner_prompt(bundle: dict[str, Any], source_text: str) -> str:
    source_summary = bundle["source_summaries"][0]
    curated = bundle["curated_tasks"][0]
    assessment = bundle.get("anchor_assessment", {})
    return (
        f"Source summary:\n{json.dumps(source_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Curated task:\n{json.dumps(curated, ensure_ascii=False, indent=2)}\n\n"
        f"Anchor assessment context:\n{json.dumps(assessment, ensure_ascii=False, indent=2)}\n\n"
        f"Visible Python source:\n```python\n{source_text}\n```\n\n"
        "Solve the task as the benchmark agent. "
        "Do not restate the full source; produce the best bounded answer artifact."
    )


def _build_judge_prompt(
    bundle: dict[str, Any],
    source_text: str,
    model_label: str,
    candidate: dict[str, Any],
) -> str:
    source_summary = bundle["source_summaries"][0]
    curated = bundle["curated_tasks"][0]
    assessment = bundle.get("anchor_assessment", {})
    return (
        f"Source summary:\n{json.dumps(source_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Curated task:\n{json.dumps(curated, ensure_ascii=False, indent=2)}\n\n"
        f"Anchor assessment context:\n{json.dumps(assessment, ensure_ascii=False, indent=2)}\n\n"
        f"Visible Python source:\n```python\n{source_text}\n```\n\n"
        f"Candidate model label: {model_label}\n"
        f"Candidate response:\n{json.dumps(candidate, ensure_ascii=False, indent=2)}\n\n"
        "Compute total_score as boundary_discipline + grounding + execution_readiness + policy_handling - unsupported_assumption_penalty. "
        "Clamp total_score to [0, 10]. "
        "Use outcome in {pass, partial, fail}."
    )


def _build_row(
    *,
    artifact_id: str,
    bundle: dict[str, Any],
    model_label: str,
    model_id: str,
    candidate: dict[str, Any],
    judged: dict[str, Any],
) -> dict[str, Any]:
    curated = bundle["curated_tasks"][0]
    source_summary = bundle["source_summaries"][0]
    assessment = bundle.get("anchor_assessment", {})
    return {
        "artifact_id": artifact_id,
        "source_title": source_summary.get("title"),
        "source_path": source_summary.get("source_path"),
        "task_title": curated.get("title"),
        "task_boundary": curated.get("core_boundary"),
        "task_planning_unit": curated.get("planning_unit"),
        "task_bucket": assessment.get("final_bucket"),
        "model_label": model_label,
        "model_id": model_id,
        "is_drafter_family": model_label in DRAFTER_FAMILY_LABELS,
        "boundary_discipline_score": int(judged["boundary_discipline_score"]),
        "grounding_score": int(judged["grounding_score"]),
        "execution_readiness_score": int(judged["execution_readiness_score"]),
        "policy_handling_score": int(judged["policy_handling_score"]),
        "unsupported_assumption_penalty": int(judged["unsupported_assumption_penalty"]),
        "total_score": int(judged["total_score"]),
        "outcome": str(judged["outcome"]),
        "pass_reasons": list(judged.get("pass_reasons", [])),
        "misses": list(judged.get("misses", [])),
        "concise_rationale": str(judged.get("concise_rationale", "")),
        "candidate_response": candidate,
    }


def _aggregate_payload(run_prefix: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bucket_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_groups[str(row["artifact_id"])].append(row)
        bucket_groups[str(row["task_bucket"])].append(row)
        model_groups[str(row["model_label"])].append(row)

    task_summaries = []
    for artifact_id, items in sorted(task_groups.items()):
        scores = {str(item["model_label"]): int(item["total_score"]) for item in items}
        outcomes = Counter(str(item["outcome"]) for item in items)
        cross_vendor_scores = {label: scores[label] for label in CROSS_VENDOR_LABELS if label in scores}
        cross_vendor_outcomes = Counter(
            str(item["outcome"]) for item in items if item["model_label"] in CROSS_VENDOR_LABELS
        )
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        task_summaries.append(
            {
                "artifact_id": artifact_id,
                "task_bucket": items[0]["task_bucket"],
                "task_title": items[0]["task_title"],
                "source_title": items[0]["source_title"],
                "scores": scores,
                "score_spread": max(scores.values()) - min(scores.values()) if scores else 0,
                "winner": sorted_scores[0][0] if sorted_scores else None,
                "outcome_votes": dict(outcomes),
                "pass_count": outcomes.get("pass", 0),
                "cross_vendor_scores": cross_vendor_scores,
                "cross_vendor_score_spread": (
                    max(cross_vendor_scores.values()) - min(cross_vendor_scores.values())
                    if cross_vendor_scores
                    else 0
                ),
                "cross_vendor_pass_count": cross_vendor_outcomes.get("pass", 0),
            }
        )

    bucket_summaries = {}
    for bucket, items in sorted(bucket_groups.items()):
        bucket_tasks = [task for task in task_summaries if task["task_bucket"] == bucket]
        pass_rates_by_model: dict[str, float] = {}
        avg_scores_by_model: dict[str, float] = {}
        for model_label in TESTER_MODELS:
            model_items = [item for item in items if item["model_label"] == model_label]
            if not model_items:
                continue
            passes = sum(1 for item in model_items if item["outcome"] == "pass")
            pass_rates_by_model[model_label] = round(passes / len(model_items), 3)
            avg_scores_by_model[model_label] = round(
                statistics.mean(int(item["total_score"]) for item in model_items),
                3,
            )
        cross_vendor_items = [item for item in items if item["model_label"] in CROSS_VENDOR_LABELS]
        cross_vendor_pass_rate = (
            round(sum(1 for item in cross_vendor_items if item["outcome"] == "pass") / len(cross_vendor_items), 3)
            if cross_vendor_items
            else None
        )
        bucket_summaries[bucket] = {
            "task_count": len({item["artifact_id"] for item in items}),
            "avg_score_spread": (
                round(statistics.mean(task["score_spread"] for task in bucket_tasks), 3)
                if bucket_tasks
                else None
            ),
            "avg_pass_count_per_task": (
                round(statistics.mean(task["pass_count"] for task in bucket_tasks), 3)
                if bucket_tasks
                else None
            ),
            "avg_cross_vendor_score_spread": (
                round(statistics.mean(task["cross_vendor_score_spread"] for task in bucket_tasks), 3)
                if bucket_tasks
                else None
            ),
            "avg_cross_vendor_pass_count_per_task": (
                round(statistics.mean(task["cross_vendor_pass_count"] for task in bucket_tasks), 3)
                if bucket_tasks
                else None
            ),
            "avg_total_score_by_model": avg_scores_by_model,
            "pass_rate_by_model": pass_rates_by_model,
            "cross_vendor_pass_rate": cross_vendor_pass_rate,
            "outcome_votes": dict(Counter(str(item["outcome"]) for item in items)),
        }

    model_summaries = {}
    for model_label, items in sorted(model_groups.items()):
        model_summaries[model_label] = {
            "is_drafter_family": model_label in DRAFTER_FAMILY_LABELS,
            "avg_total_score": round(statistics.mean(int(item["total_score"]) for item in items), 3),
            "bucket_breakdown": dict(Counter(str(item["task_bucket"]) for item in items)),
            "outcome_breakdown": dict(Counter(str(item["outcome"]) for item in items)),
        }

    return {
        "run_prefix": run_prefix,
        "judge_model": JUDGE_MODEL,
        "tester_models": TESTER_MODELS,
        "drafter_family_labels": sorted(DRAFTER_FAMILY_LABELS),
        "cross_vendor_labels": list(CROSS_VENDOR_LABELS),
        "rows": rows,
        "task_summaries": task_summaries,
        "bucket_summaries": bucket_summaries,
        "model_summaries": model_summaries,
    }


def _write_outputs(report_data_dir: Path, run_prefix: str, payload: dict[str, Any]) -> None:
    json_path = report_data_dir / f"{run_prefix}_dryrun_plan_check.json"
    csv_path = report_data_dir / f"{run_prefix}_dryrun_plan_check.csv"
    md_path = report_data_dir / f"{run_prefix}_dryrun_plan_check.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, payload["rows"])
    md_path.write_text(_render_markdown(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "artifact_id",
        "source_title",
        "source_path",
        "task_title",
        "task_bucket",
        "model_label",
        "is_drafter_family",
        "total_score",
        "outcome",
        "boundary_discipline_score",
        "grounding_score",
        "execution_readiness_score",
        "policy_handling_score",
        "unsupported_assumption_penalty",
        "concise_rationale",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Task Forge Self-Consistency Dry-Run",
        "",
        "Tester models read each `curated_task.json` plus the visible source and propose an execution plan.",
        f"Independent judge `{payload['judge_model']}` scores them `pass / partial / fail`.",
        "",
        f"- Tester models: `{', '.join(payload['tester_models'])}`",
        f"- Drafter-family testers (reference-only column): `{', '.join(payload['drafter_family_labels'])}`",
        f"- Cross-vendor roll-up labels: `{', '.join(payload['cross_vendor_labels'])}`",
        "",
        "## Bucket Summary",
        "",
    ]
    for bucket, summary in payload["bucket_summaries"].items():
        lines.extend(
            [
                f"### `{bucket}`",
                "",
                f"- task_count: `{summary['task_count']}`",
                f"- avg_score_spread (all testers): `{summary['avg_score_spread']}`",
                f"- avg_cross_vendor_score_spread: `{summary['avg_cross_vendor_score_spread']}`",
                f"- avg_pass_count_per_task (all testers): `{summary['avg_pass_count_per_task']}`",
                f"- avg_cross_vendor_pass_count_per_task: `{summary['avg_cross_vendor_pass_count_per_task']}`",
                f"- cross_vendor_pass_rate: `{summary['cross_vendor_pass_rate']}`",
                f"- avg_total_score_by_model: `{summary['avg_total_score_by_model']}`",
                f"- pass_rate_by_model: `{summary['pass_rate_by_model']}`",
                f"- outcome_votes: `{summary['outcome_votes']}`",
                "",
            ]
        )
    lines.append("## Task Summary")
    lines.append("")
    for task in payload["task_summaries"]:
        lines.append(
            f"- `{task['artifact_id']}` / `{task['task_bucket']}` / `{task['source_title']}`: "
            f"scores={task['scores']} cross_vendor_scores={task['cross_vendor_scores']} "
            f"outcomes={task['outcome_votes']}"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- MiMo overlaps with the drafting model family; its column is kept for reference and is excluded from the cross-vendor roll-up.")
    lines.append("- This dry-run is a self-consistency check on task design, not a production-pipeline step.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
