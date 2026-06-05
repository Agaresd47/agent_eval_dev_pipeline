"""Run-scoped telemetry helpers for task-forge v2."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class RuntimeTelemetry:
    run_id: str
    pipeline_variant: str
    started_at: str = field(default_factory=_utc_now_iso)
    finished_at: str | None = None
    duration_sec: float | None = None
    llm_call_count: int = 0
    structured_call_count: int = 0
    text_call_count: int = 0
    critic_call_count: int = 0
    json_repair_call_count: int = 0
    retrieval_cache_hit_count: int = 0
    retrieval_cache_miss_count: int = 0
    input_token_total: int = 0
    output_token_total: int = 0
    fast_path_taken: bool = False
    model_names: list[str] = field(default_factory=list)
    call_records: list[dict[str, Any]] = field(default_factory=list)
    retrieval_events: list[dict[str, Any]] = field(default_factory=list)

    def mark_finished(self) -> None:
        self.finished_at = _utc_now_iso()
        started = datetime.fromisoformat(self.started_at)
        finished = datetime.fromisoformat(self.finished_at)
        self.duration_sec = round((finished - started).total_seconds(), 3)

    def record_text_call(
        self,
        *,
        label: str,
        model: str,
        duration_sec: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.llm_call_count += 1
        self.text_call_count += 1
        self.input_token_total += max(0, int(input_tokens or 0))
        self.output_token_total += max(0, int(output_tokens or 0))
        self._remember_model(model)
        self.call_records.append(
            {
                "kind": "text",
                "label": label,
                "model": model,
                "duration_sec": round(duration_sec, 3),
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
            }
        )

    def record_structured_attempt(self, *, label: str, schema_name: str) -> None:
        self.structured_call_count += 1
        if label.startswith("critic:"):
            self.critic_call_count += 1
        self.call_records.append(
            {
                "kind": "structured_attempt",
                "label": label,
                "schema": schema_name,
            }
        )

    def record_json_repair(self, *, label: str) -> None:
        self.json_repair_call_count += 1
        self.call_records.append({"kind": "json_repair", "label": label})

    def record_retrieval_cache(self, *, source: str, hit: bool) -> None:
        if hit:
            self.retrieval_cache_hit_count += 1
        else:
            self.retrieval_cache_miss_count += 1
        self.retrieval_events.append({"source": source, "hit": hit})

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pipeline_variant": self.pipeline_variant,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": self.duration_sec,
            "llm_call_count": self.llm_call_count,
            "structured_call_count": self.structured_call_count,
            "text_call_count": self.text_call_count,
            "critic_call_count": self.critic_call_count,
            "json_repair_call_count": self.json_repair_call_count,
            "retrieval_cache_hit_count": self.retrieval_cache_hit_count,
            "retrieval_cache_miss_count": self.retrieval_cache_miss_count,
            "input_token_total": self.input_token_total,
            "output_token_total": self.output_token_total,
            "fast_path_taken": self.fast_path_taken,
            "model_names": list(self.model_names),
            "call_records": list(self.call_records),
            "retrieval_events": list(self.retrieval_events),
        }

    def _remember_model(self, model: str) -> None:
        if model and model not in self.model_names:
            self.model_names.append(model)


_ACTIVE_TELEMETRY: ContextVar[RuntimeTelemetry | None] = ContextVar("task_forge_runtime_telemetry", default=None)


def start_runtime_telemetry(run_id: str, pipeline_variant: str) -> RuntimeTelemetry:
    telemetry = RuntimeTelemetry(run_id=run_id, pipeline_variant=pipeline_variant)
    _ACTIVE_TELEMETRY.set(telemetry)
    return telemetry


def get_runtime_telemetry() -> RuntimeTelemetry | None:
    return _ACTIVE_TELEMETRY.get()


def close_runtime_telemetry() -> RuntimeTelemetry | None:
    telemetry = _ACTIVE_TELEMETRY.get()
    if telemetry is not None and telemetry.finished_at is None:
        telemetry.mark_finished()
    _ACTIVE_TELEMETRY.set(None)
    return telemetry
