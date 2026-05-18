"""Deterministic benchmark-shape validators."""

from __future__ import annotations

import re

from .schemas import CuratedTask, DraftTask

PATH_RE = re.compile(r"(?:[A-Za-z]:\\[^\s,;:()]+|/(?:[\w.-]+/)+[\w.-]+)")


def validate_draft_task(task: DraftTask) -> list[str]:
    issues: list[str] = []
    if not task.core_boundary.strip():
        issues.append("missing_core_boundary")
    if not task.planning_unit.strip():
        issues.append("missing_planning_unit")
    if len(task.recoverable_facts) == 0:
        issues.append("missing_recoverable_facts")
    if len(task.user_only_policies) == 0:
        issues.append("missing_user_only_policies")
    overlap = set(_normalize(task.recoverable_facts)) & set(_normalize(task.user_only_policies))
    if overlap:
        issues.append(f"recoverable_user_policy_overlap:{sorted(overlap)!r}")
    if any(PATH_RE.search(item) for item in task.user_only_policies):
        issues.append("user_policy_contains_path_like_detail")
    return issues


def validate_curated_task(task: CuratedTask) -> list[str]:
    issues: list[str] = []
    if len(task.recoverable_facts) == 0:
        issues.append("curated_missing_recoverable_facts")
    if len(task.user_only_policies) == 0:
        issues.append("curated_missing_user_policy")
    if not task.difficulty_knob:
        issues.append("missing_difficulty_knob")
    overlap = set(_normalize(task.recoverable_facts)) & set(_normalize(task.user_only_policies))
    if overlap:
        issues.append(f"curated_overlap:{sorted(overlap)!r}")
    if len(task.harness_contract) == 0:
        issues.append("missing_harness_contract")
    if len(task.guardrail_contract) == 0:
        issues.append("missing_guardrail_contract")
    if not task.benchmark_line:
        issues.append("missing_benchmark_line")
    if len(task.difficulty_levels) == 0:
        issues.append("missing_difficulty_levels")
    if any(PATH_RE.search(item) for item in task.user_only_policies):
        issues.append("curated_user_policy_contains_path_like_detail")
    if any(item.lower().startswith(("verify", "check", "ensure")) for item in task.harness_contract):
        issues.append("harness_too_verificatory")
    return issues


def _normalize(values: list[str]) -> list[str]:
    return [value.strip().lower() for value in values if value.strip()]
