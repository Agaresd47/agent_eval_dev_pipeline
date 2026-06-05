"""Typed artifacts for the task-forge v2 pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(..., description="Stable identifier for this artifact.")
    created_at: datetime = Field(default_factory=_utc_now)
    source_ids: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    @field_validator("source_ids", "tags", "notes", mode="before")
    @classmethod
    def _normalize_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


class SourceSummary(ArtifactBase):
    """Compact summary of one mined source file or document."""

    source_path: str
    source_kind: str = Field(default="unknown")
    title: Optional[str] = None
    summary: str
    key_facts: List[str] = Field(default_factory=list)
    recoverable_facts: List[str] = Field(default_factory=list)
    user_only_policies: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    linked_paths: List[str] = Field(default_factory=list)

    @field_validator(
        "key_facts",
        "recoverable_facts",
        "user_only_policies",
        "risks",
        "open_questions",
        "linked_paths",
        mode="before",
    )
    @classmethod
    def _normalize_text_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    source_id: str
    source_path: str
    chunk_id: Optional[str] = None
    rank: int = 0
    score: float = 0.0
    excerpt: str = ""
    match_reasons: List[str] = Field(default_factory=list)
    highlights: List[str] = Field(default_factory=list)

    @field_validator("match_reasons", "highlights", mode="before")
    @classmethod
    def _normalize_hit_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


class DraftTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_id: str
    title: str
    problem_statement: str
    core_boundary: str
    planning_unit: str
    recoverable_facts: List[str] = Field(default_factory=list)
    user_only_policies: List[str] = Field(default_factory=list)
    harness_ideas: List[str] = Field(default_factory=list)
    guardrail_ideas: List[str] = Field(default_factory=list)
    difficulty_knob: Optional[str] = None
    acceptance_signals: List[str] = Field(default_factory=list)
    leakage_risks: List[str] = Field(default_factory=list)
    scope_risks: List[str] = Field(default_factory=list)

    @field_validator(
        "recoverable_facts",
        "user_only_policies",
        "harness_ideas",
        "guardrail_ideas",
        "acceptance_signals",
        "leakage_risks",
        "scope_risks",
        mode="before",
    )
    @classmethod
    def _normalize_task_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


CriticLabel = Literal["boundary", "leakage", "scope", "harness"]
ReviewVerdict = Literal["pass", "revise", "fail"]


class CriticReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    critic: CriticLabel
    review_id: str
    verdict: ReviewVerdict
    thesis: str
    blocking_issues: List[str] = Field(default_factory=list)
    secondary_issues: List[str] = Field(default_factory=list)
    recommended_fixes: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)

    @field_validator(
        "blocking_issues",
        "secondary_issues",
        "recommended_fixes",
        "evidence",
        mode="before",
    )
    @classmethod
    def _normalize_review_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


class ConsensusReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consensus_id: str
    panel_verdict: ReviewVerdict
    synthesis: str
    agreed_changes: List[str] = Field(default_factory=list)
    unresolved_disagreements: List[str] = Field(default_factory=list)
    must_fix: List[str] = Field(default_factory=list)
    nice_to_have: List[str] = Field(default_factory=list)
    revision_brief: str = ""

    @field_validator(
        "agreed_changes",
        "unresolved_disagreements",
        "must_fix",
        "nice_to_have",
        mode="before",
    )
    @classmethod
    def _normalize_consensus_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


SuitabilityVerdict = Literal["anchor_candidate", "supporting_candidate", "reject"]


class SourceTriage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    triage_id: str
    verdict: SuitabilityVerdict
    benchmark_line_guess: Literal["t1_cli_style", "t2_handoff_style", "unclear"]
    confidence: Literal["low", "medium", "high"]
    rationale: str
    strengths: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)
    recommended_boundary: str = ""

    @field_validator("strengths", "blockers", mode="before")
    @classmethod
    def _normalize_triage_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


class AnchorAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    final_bucket: SuitabilityVerdict
    is_anchor_eligible: bool
    rationale: str
    reason_codes: List[str] = Field(default_factory=list)

    @field_validator("reason_codes", mode="before")
    @classmethod
    def _normalize_reason_codes(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


class CuratedTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    title: str
    benchmark_line: Optional[str] = None
    benchmark_goal: str
    core_boundary: str
    planning_unit: str
    problem_statement: str
    recoverable_facts: List[str] = Field(default_factory=list)
    user_only_policies: List[str] = Field(default_factory=list)
    harness_contract: List[str] = Field(default_factory=list)
    guardrail_contract: List[str] = Field(default_factory=list)
    difficulty_knob: Optional[str] = None
    difficulty_levels: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    failure_modes: List[str] = Field(default_factory=list)
    quality_flags: List[str] = Field(default_factory=list)
    memory_anchors: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)
    canonical_answer_shape: Optional[str] = None

    @field_validator(
        "recoverable_facts",
        "user_only_policies",
        "harness_contract",
        "guardrail_contract",
        "difficulty_levels",
        "acceptance_criteria",
        "failure_modes",
        "quality_flags",
        "memory_anchors",
        "source_ids",
        mode="before",
    )
    @classmethod
    def _normalize_curated_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


class RunArtifact(BaseModel):
    """Container for one end-to-end task-forge run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: datetime = Field(default_factory=_utc_now)
    repo_root: Optional[str] = None
    pipeline_name: str = "task_forge_v2"
    input_query: Optional[str] = None
    model_name: Optional[str] = None
    retrieval_query: Optional[str] = None
    source_summaries: List[SourceSummary] = Field(default_factory=list)
    retrieval_hits: List[RetrievalHit] = Field(default_factory=list)
    draft_tasks: List[DraftTask] = Field(default_factory=list)
    critic_reviews: List[CriticReview] = Field(default_factory=list)
    consensus_reviews: List[ConsensusReview] = Field(default_factory=list)
    source_triage: Optional[SourceTriage] = None
    anchor_assessment: Optional[AnchorAssessment] = None
    curated_tasks: List[CuratedTask] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ArtifactBase",
    "SourceSummary",
    "RetrievalHit",
    "DraftTask",
    "CriticReview",
    "ConsensusReview",
    "SourceTriage",
    "AnchorAssessment",
    "CuratedTask",
    "RunArtifact",
    "CriticLabel",
    "ReviewVerdict",
    "SuitabilityVerdict",
]
