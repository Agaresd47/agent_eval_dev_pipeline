from __future__ import annotations

import task_forge_v2.retrieval as retrieval_module
from task_forge_v2.graph import _render_summary, synthesize_review_no_critics_node
from task_forge_v2.runtime_metrics import close_runtime_telemetry, start_runtime_telemetry
from task_forge_v2.schemas import AnchorAssessment, CuratedTask, DraftTask, RunArtifact, SourceSummary, SourceTriage


def test_synthesize_review_no_critics_builds_deterministic_consensus():
    summary = SourceSummary(
        artifact_id="source::compress",
        source_path="code_source/dirty_work/compress.py",
        title="compress.py",
        summary="Compress a folder, split the archive, and decide when cleanup is safe.",
    )
    draft = DraftTask(
        draft_id="draft_compress_001",
        title="Delete Original Archive Only After Verified Split Output",
        problem_statement="Verify split output before cleaning up the original archive.",
        core_boundary="Delete the original archive only after every expected split part is visible.",
        planning_unit="single archive cleanup eligibility decision",
    )

    consensus = synthesize_review_no_critics_node({"source_summary": summary, "draft_task": draft})["consensus_review"]

    assert consensus.consensus_id == "draft_compress_001::no_critics"
    assert consensus.panel_verdict == "pass"
    assert "Critic panel skipped" in consensus.synthesis
    assert "no critic panel" in consensus.revision_brief.lower()


def test_runtime_telemetry_tracks_calls_and_finishes():
    telemetry = start_runtime_telemetry("run_001", "no_critics")
    telemetry.record_structured_attempt(label="triage_source", schema_name="SourceTriage")
    telemetry.record_text_call(label="triage_source", model="mimo-v2.5-pro", duration_sec=1.25)
    telemetry.record_json_repair(label="draft_task")
    telemetry.record_retrieval_cache(source="disk", hit=True)
    telemetry.fast_path_taken = True
    closed = close_runtime_telemetry()

    assert closed is telemetry
    assert closed is not None
    assert closed.finished_at is not None
    assert closed.duration_sec is not None
    payload = closed.to_dict()
    assert payload["pipeline_variant"] == "no_critics"
    assert payload["llm_call_count"] == 1
    assert payload["structured_call_count"] == 1
    assert payload["json_repair_call_count"] == 1
    assert payload["retrieval_cache_hit_count"] == 1
    assert payload["fast_path_taken"] is True


def test_render_summary_includes_pipeline_variant_and_runtime_fields():
    source = SourceSummary(
        artifact_id="source::pp_move_finished",
        source_path="code_source/dirty_work/PP_move_finished.py",
        summary="Promote only complete image-mask pairs into a finished area.",
    )
    triage = SourceTriage(
        triage_id="triage_pp_move_finished_001",
        verdict="supporting_candidate",
        benchmark_line_guess="t1_cli_style",
        confidence="medium",
        rationale="Promising workflow slice.",
    )
    curated = CuratedTask(
        task_id="task_pp_move_finished_001",
        title="Promote Complete Image-Mask Pairs Safely",
        benchmark_goal="Evaluate safe pair promotion decisions.",
        core_boundary="Move only verified complete pairs and leave partial pairs untouched.",
        planning_unit="single image-mask pair eligibility decision",
        problem_statement="Inspect pairs and move only complete ones.",
    )
    assessment = AnchorAssessment(
        assessment_id="assessment_pp_move_finished_001",
        final_bucket="anchor_candidate",
        is_anchor_eligible=True,
        rationale="Execution-grounded and observable.",
        reason_codes=["specific_planning_unit", "execution_observable_answer"],
    )
    bundle = RunArtifact(
        run_id="task_forge_test_001",
        source_summaries=[source],
        source_triage=triage,
        anchor_assessment=assessment,
        curated_tasks=[curated],
        metadata={
            "pipeline_variant": "no_critics",
            "runtime": {"llm_call_count": 2, "fast_path_taken": False},
            "validator_issues": [],
        },
    )

    summary = _render_summary(bundle)

    assert "- Pipeline variant: no_critics" in summary
    assert "- LLM calls: 2" in summary
    assert "- Fast path taken: False" in summary


def test_effective_corpus_signature_can_freeze_snapshot(monkeypatch):
    monkeypatch.setenv("TASK_FORGE_FREEZE_RETRIEVAL_CORPUS", "1")
    monkeypatch.setattr(retrieval_module, "_FROZEN_SIGNATURE", None)
    calls = iter(["sig-001", "sig-002"])
    monkeypatch.setattr(retrieval_module, "_corpus_signature", lambda: next(calls))

    first = retrieval_module._effective_corpus_signature()
    second = retrieval_module._effective_corpus_signature()

    assert first == "sig-001"
    assert second == "sig-001"

    monkeypatch.setenv("TASK_FORGE_FREEZE_RETRIEVAL_CORPUS", "0")
    monkeypatch.setattr(retrieval_module, "_FROZEN_SIGNATURE", None)
    calls = iter(["sig-003", "sig-004"])
    monkeypatch.setattr(retrieval_module, "_corpus_signature", lambda: next(calls))

    assert retrieval_module._effective_corpus_signature() == "sig-003"
    assert retrieval_module._effective_corpus_signature() == "sig-004"
