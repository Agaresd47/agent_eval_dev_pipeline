"""Structured LLM helpers using LangChain's OpenAI-compatible client."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Type, TypeVar

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import get_model_defaults, load_dotenv
from .runtime_metrics import get_runtime_telemetry

T = TypeVar("T", bound=BaseModel)
_LAST_API_CALL_AT: float | None = None


@dataclass(slots=True)
class ModelCallMeta:
    model: str
    temperature: float
    max_completion_tokens: int


class MimoStructuredClient:
    """Thin wrapper around ChatOpenAI for structured JSON-ish outputs."""

    def __init__(self, model: str | None = None, temperature: float | None = None, max_completion_tokens: int | None = None):
        load_dotenv()
        defaults = get_model_defaults()
        self.model = model or defaults.chat_model
        self.temperature = temperature if temperature is not None else defaults.temperature
        self.max_completion_tokens = max_completion_tokens if max_completion_tokens is not None else defaults.max_output_tokens
        api_key = (
            os.getenv("TASK_FORGE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("Xiaomi")
        )
        if not api_key:
            raise RuntimeError("Missing API key. Set TASK_FORGE_API_KEY or OPENAI_API_KEY in .env.")
        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": api_key,
            "base_url": defaults.base_url,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
        }
        if os.getenv("TASK_FORGE_DISABLE_THINKING", "").strip().lower() in {"1", "true", "yes", "on"}:
            llm_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        self.llm = ChatOpenAI(**llm_kwargs)

    @property
    def meta(self) -> ModelCallMeta:
        return ModelCallMeta(
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
        )

    def invoke_text(self, system_prompt: str, user_prompt: str, *, label: str = "llm_call") -> str:
        _respect_api_spacing()
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{user_prompt}")]
        )
        chain = prompt | self.llm
        started_at = time.perf_counter()
        response = chain.invoke({"user_prompt": user_prompt})
        elapsed = time.perf_counter() - started_at
        telemetry = get_runtime_telemetry()
        if telemetry is not None:
            telemetry.record_text_call(label=label, model=self.model, duration_sec=elapsed)
        text = getattr(response, "content", "") or ""
        if isinstance(text, list):
            parts = []
            for item in text:
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                elif hasattr(item, "text"):
                    parts.append(str(item.text))
            text = "".join(parts)
        return str(text).strip()

    def invoke_model(self, schema: Type[T], system_prompt: str, user_prompt: str, *, label: str = "structured_call") -> T:
        telemetry = get_runtime_telemetry()
        if telemetry is not None:
            telemetry.record_structured_attempt(label=label, schema_name=schema.__name__)
        parser = PydanticOutputParser(pydantic_object=schema)
        format_instructions = parser.get_format_instructions()
        full_user_prompt = (
            f"{user_prompt}\n\n"
            "Return exactly one JSON object that matches the required schema.\n"
            f"{format_instructions}"
        )
        raw_text = self.invoke_text(system_prompt, full_user_prompt, label=label)
        try:
            return parser.parse(raw_text)
        except Exception:
            repaired = self._repair_json(raw_text=raw_text, label=label)
            try:
                return parser.parse(repaired)
            except Exception:
                payload = _extract_json_object(repaired or raw_text)
                return schema.model_validate(payload)

    def _repair_json(self, raw_text: str, *, label: str) -> str:
        telemetry = get_runtime_telemetry()
        if telemetry is not None:
            telemetry.record_json_repair(label=label)
        repair_prompt = (
            "The previous answer was supposed to be one valid JSON object but was malformed.\n"
            "Repair it into one valid JSON object matching the same schema. Do not add commentary.\n\n"
            f"Malformed output:\n{raw_text}"
        )
        return self.invoke_text(
            "You repair malformed JSON outputs for a benchmark-authoring pipeline.",
            repair_prompt,
            label=f"{label}:json_repair",
        )


def _extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def _respect_api_spacing() -> None:
    global _LAST_API_CALL_AT
    min_gap_seconds = float(os.getenv("TASK_FORGE_MIN_API_SPACING_SEC", "0"))
    now = time.monotonic()
    if _LAST_API_CALL_AT is not None:
        elapsed = now - _LAST_API_CALL_AT
        if elapsed < min_gap_seconds:
            time.sleep(min_gap_seconds - elapsed)
    _LAST_API_CALL_AT = time.monotonic()
