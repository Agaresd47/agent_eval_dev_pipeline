"""Deterministic cohort and anchor assessment for curated tasks."""

from __future__ import annotations

from .schemas import AnchorAssessment, CuratedTask, SourceSummary, SourceTriage

GENERIC_PLANNING_UNITS = {
    "single bounded workflow decision",
    "single source-to-destination move decision",
}


def assess_task(
    source_summary: SourceSummary,
    source_triage: SourceTriage,
    curated_task: CuratedTask,
    validator_issues: list[str] | None,
) -> AnchorAssessment:
    issues = list(validator_issues or [])
    quality_flags = set(curated_task.quality_flags or [])
    reason_codes: list[str] = []

    if "reject_seed" in quality_flags:
        reason_codes.extend(["reject_seed", "source_triage_reject"])
        return AnchorAssessment(
            assessment_id=f"assessment::{curated_task.task_id}",
            final_bucket="reject",
            is_anchor_eligible=False,
            rationale="Rejected intentionally because the source never had enough visible evidence to justify authoring a benchmark task.",
            reason_codes=reason_codes,
        )

    if issues:
        reason_codes.append("validator_issues_present")
        return AnchorAssessment(
            assessment_id=f"assessment::{curated_task.task_id}",
            final_bucket="reject",
            is_anchor_eligible=False,
            rationale="Rejected because deterministic validators still found structural issues.",
            reason_codes=reason_codes,
        )

    if source_triage.verdict == "reject":
        reason_codes.append("source_triage_reject")
        return AnchorAssessment(
            assessment_id=f"assessment::{curated_task.task_id}",
            final_bucket="reject",
            is_anchor_eligible=False,
            rationale="Rejected because source triage found the seed too weak before task authoring.",
            reason_codes=reason_codes,
        )

    if "static_code_review_shape" in quality_flags:
        reason_codes.extend(["static_review_shape", "not_execution_grounded"])
        final_bucket = "reject" if _is_weak_static_review(source_summary, curated_task) else "supporting_candidate"
        rationale = (
            "Static-review-shaped tasks stay out of the anchor lane even when they look locally crisp."
            if final_bucket == "supporting_candidate"
            else "Rejected because the task is both static-review-shaped and too weakly grounded to survive as support."
        )
        return AnchorAssessment(
            assessment_id=f"assessment::{curated_task.task_id}",
            final_bucket=final_bucket,
            is_anchor_eligible=False,
            rationale=rationale,
            reason_codes=reason_codes,
        )

    if "workflow_grounded_shape" in quality_flags:
        reason_codes.append("workflow_grounded_shape")
    if "single_boundary" in quality_flags:
        reason_codes.append("single_boundary")
    if "artifact_ready" in quality_flags:
        reason_codes.append("artifact_ready")
    if "runtime_verification_underdefined" in quality_flags:
        reason_codes.append("runtime_verification_underdefined")
    if "mutation_move" in source_summary.risks:
        reason_codes.append("mutation_seed")
    if "policy_separation_sensitive" in quality_flags:
        reason_codes.append("policy_boundary_present")
    if "harness_observation_required" in quality_flags:
        reason_codes.append("harness_observes_boundary")

    if _has_specific_planning_unit(curated_task):
        reason_codes.append("specific_planning_unit")
    else:
        reason_codes.append("generic_planning_unit")

    if _has_execution_observable_answer(curated_task):
        reason_codes.append("execution_observable_answer")

    if "runtime_verification_underdefined" in quality_flags:
        reason_codes.append("needs_human_curation")
        return AnchorAssessment(
            assessment_id=f"assessment::{curated_task.task_id}",
            final_bucket="supporting_candidate",
            is_anchor_eligible=False,
            rationale=(
                "Kept as support because the task asks for runtime verification that is not fully operationalized by the visible evidence. "
                "It should not enter the anchor lane until the verification rule is made explicit."
            ),
            reason_codes=reason_codes,
        )

    if _is_anchor_ready(curated_task, source_summary, reason_codes):
        return AnchorAssessment(
            assessment_id=f"assessment::{curated_task.task_id}",
            final_bucket="anchor_candidate",
            is_anchor_eligible=True,
            rationale="Anchor-worthy because the final task is workflow-grounded, execution-observable, and scoped to one specific planning unit.",
            reason_codes=reason_codes,
        )

    reason_codes.append("needs_human_curation")
    return AnchorAssessment(
        assessment_id=f"assessment::{curated_task.task_id}",
        final_bucket="supporting_candidate",
        is_anchor_eligible=False,
        rationale="Kept as support because the task is usable, but its final shape is still too generic or weakly execution-grounded for anchor duty.",
        reason_codes=reason_codes,
    )


def _has_specific_planning_unit(curated_task: CuratedTask) -> bool:
    planning_unit = (curated_task.planning_unit or "").strip().lower()
    return bool(planning_unit) and planning_unit not in GENERIC_PLANNING_UNITS


def _has_execution_observable_answer(curated_task: CuratedTask) -> bool:
    text = " ".join(
        [
            curated_task.benchmark_goal.lower(),
            curated_task.core_boundary.lower(),
            curated_task.problem_statement.lower(),
            (curated_task.canonical_answer_shape or "").lower(),
        ]
    )
    execution_markers = ("move", "promot", "link", "emit", "write", "ledger", "execute", "mutat")
    return any(marker in text for marker in execution_markers)


def _is_anchor_ready(curated_task: CuratedTask, source_summary: SourceSummary, reason_codes: list[str]) -> bool:
    quality_flags = set(curated_task.quality_flags or [])
    return (
        "workflow_grounded_shape" in quality_flags
        and "single_boundary" in quality_flags
        and "artifact_ready" in quality_flags
        and _has_specific_planning_unit(curated_task)
        and _has_execution_observable_answer(curated_task)
        and (
            "mutation_move" in source_summary.risks
            or "archive_cleanup" in source_summary.risks
            or "file_link" in source_summary.risks
            or "schema_emit" in source_summary.risks
            or "mutation_seed" in reason_codes
        )
    )


def _is_weak_static_review(source_summary: SourceSummary, curated_task: CuratedTask) -> bool:
    summary_text = " ".join([source_summary.summary.lower(), " ".join(source_summary.key_facts).lower()])
    task_text = " ".join(
        [
            curated_task.benchmark_goal.lower(),
            curated_task.core_boundary.lower(),
            curated_task.problem_statement.lower(),
        ]
    )
    has_code_evidence = "function" in summary_text or "import" in summary_text or "source code" in task_text
    return not has_code_evidence


__all__ = ["assess_task"]
