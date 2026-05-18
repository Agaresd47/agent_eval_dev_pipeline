from __future__ import annotations

from task_forge_v2.schemas import CuratedTask, DraftTask
from task_forge_v2.schemas import SourceSummary, SourceTriage
from task_forge_v2.triage_rules import normalize_source_triage
from task_forge_v2.validators import validate_curated_task, validate_draft_task


def test_validate_draft_task_accepts_complete_non_overlapping_task():
    task = DraftTask(
        draft_id="draft-1",
        title="Test",
        problem_statement="Problem",
        core_boundary="Local file inspection only",
        planning_unit="One source file",
        recoverable_facts=["Inspect files"],
        user_only_policies=["Ask user before mutation"],
    )

    assert validate_draft_task(task) == []


def test_validate_draft_task_reports_missing_fields_and_overlap():
    task = DraftTask(
        draft_id="draft-2",
        title="Test",
        problem_statement="Problem",
        core_boundary=" ",
        planning_unit="",
        recoverable_facts=["Inspect files", "Keep list"],
        user_only_policies=[" keep list ", "Ask user first"],
    )

    issues = validate_draft_task(task)

    assert "missing_core_boundary" in issues
    assert "missing_planning_unit" in issues
    assert "recoverable_user_policy_overlap:['keep list']" in issues


def test_validate_draft_task_does_not_treat_plain_slashes_as_paths():
    task = DraftTask(
        draft_id="draft-3",
        title="Test",
        problem_statement="Problem",
        core_boundary="Local file inspection only",
        planning_unit="One source file",
        recoverable_facts=["Inspect files"],
        user_only_policies=["Validation rules for file integrity before/after moves"],
    )

    issues = validate_draft_task(task)

    assert "user_policy_contains_path_like_detail" not in issues


def test_validate_curated_task_reports_required_fields_and_overlap():
    task = CuratedTask(
        task_id="task-1",
        title="Test",
        benchmark_goal="Goal",
        core_boundary="Boundary",
        planning_unit="Unit",
        problem_statement="Problem",
        recoverable_facts=["Inspect files"],
        user_only_policies=[" inspect files "],
        harness_contract=[],
        guardrail_contract=[],
    )

    issues = validate_curated_task(task)

    assert "missing_difficulty_knob" in issues
    assert "missing_harness_contract" in issues
    assert "missing_guardrail_contract" in issues
    assert "missing_benchmark_line" in issues
    assert "missing_difficulty_levels" in issues
    assert "curated_overlap:['inspect files']" in issues


def test_validate_curated_task_accepts_sharp_observation_first_contract():
    task = CuratedTask(
        task_id="task-2",
        title="Test",
        benchmark_line="t1_cli_style",
        benchmark_goal="Goal",
        core_boundary="Boundary",
        planning_unit="single file decision",
        problem_statement="Problem",
        recoverable_facts=["Inspect files"],
        user_only_policies=["Ask user before mutation"],
        harness_contract=["Record inspected files before mutation."],
        guardrail_contract=["Block mutation on unverified evidence."],
        difficulty_knob="how explicit the rule is",
        difficulty_levels=["A0 explicit", "A1 recoverable", "A2 ask"],
    )

    assert validate_curated_task(task) == []


def test_normalize_source_triage_forces_reject_when_no_code_evidence_exists():
    summary = SourceSummary(
        artifact_id="source::weak",
        source_path="code_source/dirty_work/weak.py",
        summary="Mine risky workflow behavior from weak.py with linked paths only.",
        key_facts=["imports: none", "functions: none"],
        recoverable_facts=["Only linked paths are visible."],
        linked_paths=["/data/example/input", "/data/example/output"],
    )
    triage = SourceTriage(
        triage_id="source::weak",
        verdict="supporting_candidate",
        benchmark_line_guess="t1_cli_style",
        confidence="medium",
        rationale="Maybe usable from path structure.",
    )

    normalized = normalize_source_triage(summary, triage)

    assert normalized.verdict == "reject"
    assert normalized.benchmark_line_guess == "unclear"
    assert normalized.recommended_boundary == ""


def test_normalize_source_triage_rescues_strong_workflow_seed_from_false_reject():
    summary = SourceSummary(
        artifact_id="source::compress",
        source_path="code_source/dirty_work/compress.py",
        summary="Compress a source folder, split the resulting archive, and clean up the original archive only after verification.",
        key_facts=["imports: os, tarfile", "functions: compress_folder, split_file, clean_up"],
        recoverable_facts=[
            "Archive output names are recoverable from code.",
            "Whether the original archive still exists is recoverable by inspection.",
        ],
        risks=["archive_cleanup"],
    )
    triage = SourceTriage(
        triage_id="source::compress",
        verdict="reject",
        benchmark_line_guess="t1_cli_style",
        confidence="medium",
        rationale="Overly conservative model reject.",
        blockers=["Source was previously rejected in benchmark context as insufficient for authoring"],
    )

    normalized = normalize_source_triage(summary, triage)

    assert normalized.verdict == "supporting_candidate"
    assert normalized.recommended_boundary == "Delete the original archive only after the split output is fully verified."
