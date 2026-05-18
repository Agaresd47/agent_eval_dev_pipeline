from __future__ import annotations

from task_forge_v2.retrieval import _keep_hit, _task_family_from_curated, _task_shape_from_curated
from task_forge_v2.schemas import RetrievalHit, SourceSummary


def test_task_family_from_curated_detects_archive_cleanup():
    curated = {
        "title": "Delete Original Archive Only After Verified Split Output",
        "benchmark_goal": "Evaluate whether archive cleanup is safe.",
        "core_boundary": "Delete the original archive only after every expected split part is present.",
        "planning_unit": "single archive cleanup eligibility decision",
        "problem_statement": "Inspect split output and decide whether cleanup is safe.",
        "quality_flags": ["workflow_grounded_shape"],
    }

    assert _task_family_from_curated(curated) == "archive_cleanup"
    assert _task_shape_from_curated(curated) == "workflow_grounded"


def test_keep_hit_filters_static_review_for_archive_cleanup_seed():
    summary = SourceSummary(
        artifact_id="source::compress",
        source_path="code_source/dirty_work/compress.py",
        summary="Compress a source folder, split the archive, and clean up only after verification.",
        risks=["archive_cleanup"],
    )
    static_hit = RetrievalHit(
        query="q",
        source_id="s1",
        source_path="dev/artifacts/example.json",
        chunk_id="1",
        rank=1,
        score=0.8,
        excerpt='{"quality_flags":["static_code_review_shape"],"title":"Identify Unsafe File Deletion Patterns"}',
        match_reasons=["workflow_analog"],
        highlights=["family:archive_cleanup", "shape:static_review"],
    )
    workflow_hit = RetrievalHit(
        query="q",
        source_id="s2",
        source_path="dev/artifacts/example2.json",
        chunk_id="2",
        rank=2,
        score=0.7,
        excerpt='{"quality_flags":["workflow_grounded_shape"],"title":"Delete Original Archive Only After Verified Split Output"}',
        match_reasons=["workflow_analog"],
        highlights=["family:archive_cleanup", "shape:workflow_grounded"],
    )

    assert _keep_hit(summary, static_hit, "workflow_analog") is False
    assert _keep_hit(summary, workflow_hit, "workflow_analog") is True
