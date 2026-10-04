"""Simple JSON file storage for previously validated trip plans."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


DATE_TOLERANCE_DAYS = 30
BUDGET_TOLERANCE = 500


def _destination_key(destination: str) -> str:
    return " ".join(destination.casefold().split())


def _filename_slug(destination: str) -> str:
    normalized = unicodedata.normalize("NFKD", destination).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
    return slug or "trip"


def find_reusable_plan(
    memory_dir: Path,
    *,
    destination: str,
    trip_date: date,
    budget_per_person: float,
) -> dict[str, Any] | None:
    """Return the closest passing plan matching the requested destination/date/budget."""
    if not memory_dir.is_dir():
        return None

    matches: list[tuple[int, float, str, dict[str, Any]]] = []
    for path in memory_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("eval_status") != "PASS":
                continue
            if not isinstance(record.get("final_plan"), dict):
                continue
            if _destination_key(str(record.get("destination", ""))) != _destination_key(destination):
                continue
            saved_date = date.fromisoformat(str(record["trip_date"]))
            days_apart = abs((saved_date - trip_date).days)
            saved_budget = float(record["budget_per_person"])
            budget_apart = abs(saved_budget - budget_per_person)
            if days_apart <= DATE_TOLERANCE_DAYS and budget_apart <= BUDGET_TOLERANCE:
                matches.append((days_apart, budget_apart, path.name, record))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Ignore incomplete or malformed memory files and continue searching.
            continue

    if not matches:
        return None
    selected = min(matches, key=lambda match: match[:3])
    result = selected[3]
    result["_filename"] = selected[2]
    return result


def save_successful_plan(
    memory_dir: Path,
    *,
    destination: str,
    trip_date: date,
    budget_per_person: float,
    people: int,
    start_time: str,
    end_time: str,
    evaluation: dict[str, Any],
    final_plan: dict[str, Any],
) -> Path | None:
    """Persist a plan only when its deterministic evaluation passed."""
    if evaluation.get("status") != "PASS":
        return None

    memory_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{_filename_slug(destination)}_{trip_date.isoformat()}_{round(budget_per_person)}"
    path = memory_dir / f"{base_name}.json"
    if path.exists():
        created_stamp = datetime.now().strftime("%H%M%S_%f")
        path = memory_dir / f"{base_name}_{created_stamp}.json"

    record = {
        "destination": destination.strip(),
        "trip_date": trip_date.isoformat(),
        "budget_per_person": budget_per_person,
        "number_of_people": people,
        "start_time": start_time,
        "end_time": end_time,
        "created_time": datetime.now(timezone.utc).isoformat(),
        "eval_status": "PASS",
        "evaluation": evaluation,
        "final_plan": final_plan,
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
