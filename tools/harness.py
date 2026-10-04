"""Per-request guardrails for LLM-requested tool execution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from tools.registry import dispatch_tool


MAX_TOOL_CALLS = 3
MAX_ATTRACTION_RESULTS = 10
WEATHER_DATE_REJECTION = (
    "Weather can only be requested for dates within the trip start and end dates. "
    "Use a valid trip date or continue with the information already available."
)
TOOL_LIMIT_MESSAGE = (
    "Maximum tool-call limit reached. No more tools can be used. Produce the best final plan "
    "using information already collected and mention anything that could not be verified."
)


def _normalize(value: Any) -> Any:
    """Normalize JSON arguments for stable duplicate detection."""
    if isinstance(value, dict):
        return {str(key).strip().casefold(): _normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]).casefold())}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip().casefold()
    return value


@dataclass
class ToolCallHarness:
    """Guardrail state scoped to one planner request."""

    trip_start_date: date
    trip_end_date: date
    max_successful_calls: int = MAX_TOOL_CALLS
    successful_call_count: int = 0
    executed_signatures: set[str] = field(default_factory=set)

    def _parse_arguments(self, arguments: str | dict[str, Any]) -> dict[str, Any]:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must be a JSON object")
        return parsed

    def _signature(self, name: str, arguments: dict[str, Any]) -> str:
        arguments = dict(arguments)
        defaults = {
            "search_attractions": {"category": None, "limit": 6},
            "get_weather": {"city": None, "latitude": None, "longitude": None},
        }
        for key, value in defaults.get(name, {}).items():
            arguments.setdefault(key, value)
        canonical = json.dumps(_normalize(arguments), sort_keys=True, ensure_ascii=False,
                              separators=(",", ":"), default=str)
        return f"{name.strip().casefold()}:{canonical}"

    def validate(self, name: str, arguments: str | dict[str, Any]
                 ) -> tuple[dict[str, Any] | None, str | None, str | None]:
        """Return parsed arguments, signature and any pre-execution rejection."""
        if name not in {"search_attractions", "get_weather", "get_travel_time"}:
            return None, None, "Tool is not available."

        try:
            parsed = self._parse_arguments(arguments)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return None, None, f"Invalid tool arguments: {exc}"

        if name == "get_weather":
            requested = parsed.get("trip_date")
            try:
                requested_date = date.fromisoformat(requested) if isinstance(requested, str) else requested
                if not isinstance(requested_date, date):
                    raise ValueError("trip_date is required")
            except (TypeError, ValueError):
                return None, None, WEATHER_DATE_REJECTION
            if not self.trip_start_date <= requested_date <= self.trip_end_date:
                return None, None, WEATHER_DATE_REJECTION

        signature = self._signature(name, parsed)
        if signature in self.executed_signatures:
            return None, None, "duplicate request"

        if self.successful_call_count >= self.max_successful_calls:
            return None, None, "tool-call limit reached"

        return parsed, signature, None

    def execute_validated(self, name: str, parsed: dict[str, Any], signature: str
                          ) -> tuple[Any, bool]:
        """Execute validated arguments and commit state only on successful return."""
        try:
            result = dispatch_tool(name, parsed)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}, False

        if name == "search_attractions":
            if isinstance(result, list):
                result = result[:MAX_ATTRACTION_RESULTS]
            elif isinstance(result, dict):
                for key in ("results", "attractions", "places"):
                    if isinstance(result.get(key), list):
                        result[key] = result[key][:MAX_ATTRACTION_RESULTS]
                        break

        self.successful_call_count += 1
        self.executed_signatures.add(signature)
        return result, True
