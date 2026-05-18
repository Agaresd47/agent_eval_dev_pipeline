"""Deterministic cleanup for model-produced source triage."""

from __future__ import annotations

from .schemas import SourceSummary, SourceTriage


def normalize_source_triage(source_summary: SourceSummary, triage: SourceTriage) -> SourceTriage:
    if _has_no_code_evidence(source_summary):
        blockers = list(triage.blockers)
        hard_blocker = "No recoverable code evidence is visible; linked paths alone cannot justify benchmark authoring."
        if hard_blocker not in blockers:
            blockers.append(hard_blocker)
        return triage.model_copy(
            update={
                "verdict": "reject",
                "benchmark_line_guess": "unclear",
                "confidence": "high",
                "blockers": blockers,
                "recommended_boundary": "",
                "rationale": (
                    "Forced reject by deterministic triage rules because the source has linked paths but no imports, functions, "
                    "or other concrete code facts to ground a benchmark boundary."
                ),
            }
        )
    if triage.verdict == "reject" and _is_strong_workflow_seed(source_summary):
        blockers = [item for item in triage.blockers if "previously rejected" not in item.lower()]
        if not blockers:
            blockers = ["Model triage was overly conservative relative to visible code evidence."]
        return triage.model_copy(
            update={
                "verdict": "supporting_candidate",
                "confidence": "medium",
                "blockers": blockers,
                "rationale": (
                    "Raised to supporting_candidate by deterministic triage rules because the source has concrete code evidence, "
                    "a visible workflow family, and enough recoverable facts to justify task authoring."
                ),
                "recommended_boundary": triage.recommended_boundary
                or _default_boundary_hint(source_summary),
            }
        )
    return triage


def _has_no_code_evidence(source_summary: SourceSummary) -> bool:
    key_facts = " ".join(item.lower() for item in source_summary.key_facts)
    has_imports = "imports:" in key_facts and "imports: none" not in key_facts
    has_functions = "functions:" in key_facts and "functions: none" not in key_facts
    explicit_code_markers = ("def ", "class ", "import ", "function ")
    summary_text = " ".join(
        [
            source_summary.summary.lower(),
            " ".join(source_summary.recoverable_facts).lower(),
            key_facts,
        ]
    )
    has_other_code_marker = any(marker in summary_text for marker in explicit_code_markers)
    return not has_imports and not has_functions and not has_other_code_marker


def _is_strong_workflow_seed(source_summary: SourceSummary) -> bool:
    workflow_risks = {"archive_cleanup", "mutation_move", "symlink_layout", "schema_generation"}
    if not workflow_risks.intersection(source_summary.risks):
        return False
    key_facts = " ".join(item.lower() for item in source_summary.key_facts)
    has_imports = "imports:" in key_facts and "imports: none" not in key_facts
    has_functions = "functions:" in key_facts and "functions: none" not in key_facts
    return has_imports and has_functions and len(source_summary.recoverable_facts) >= 2


def _default_boundary_hint(source_summary: SourceSummary) -> str:
    if "archive_cleanup" in source_summary.risks:
        return "Delete the original archive only after the split output is fully verified."
    if "mutation_move" in source_summary.risks:
        return "Execute only verified safe promotions and leave ambiguous cases untouched."
    if "symlink_layout" in source_summary.risks:
        return "Create only verified links and surface missing sources explicitly."
    if "schema_generation" in source_summary.risks:
        return "Emit only schema entries grounded in visible paths and stated defaults."
    return ""


__all__ = ["normalize_source_triage"]
