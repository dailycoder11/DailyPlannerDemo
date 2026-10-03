"""Shared helpers for external tool requests and call logging."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from config.settings import ROOT_DIR, load_settings

SETTINGS = load_settings()
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger("daytrip.api_calls")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False
if not LOGGER.handlers:
    handler = logging.FileHandler(LOG_DIR / "api_calls.jsonl", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)


def log_api_call(service: str, operation: str, params: dict[str, Any], started: float,
                 status: str, error: str | None = None) -> None:
    """Record one outgoing service request without credentials."""
    record: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "requestId": str(uuid.uuid4()),
        "service": service,
        "operation": operation,
        "status": status,
        "durationMs": round((time.monotonic() - started) * 1000),
        "params": params,
    }
    if error:
        record["error"] = error
    LOGGER.info(json.dumps(record, ensure_ascii=False, default=str))


def get_json(service: str, operation: str, url: str, params: dict[str, Any],
             *, timeout: float = 12.0, secret_params: set[str] | None = None,
             headers: dict[str, str] | None = None) -> Any:
    """GET JSON and log success or failure. Sensitive query fields are redacted."""
    started = time.monotonic()
    safe_params = {key: ("[REDACTED]" if secret_params and key in secret_params else value)
                   for key, value in params.items()}
    status = "http_error"
    error = None
    try:
        response = requests.get(url, params=params, timeout=timeout, headers=headers)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, (dict, list)):
            raise ValueError("Expected a JSON object or array from provider.")
        status = "success"
        return payload
    except requests.Timeout as exc:
        status, error = "timeout", "Request timed out."
        raise RuntimeError(f"{service} request timed out.") from exc
    except requests.HTTPError as exc:
        status, error = "http_error", f"HTTP {exc.response.status_code}"
        raise RuntimeError(f"{service} returned HTTP {exc.response.status_code}.") from exc
    except (requests.RequestException, ValueError) as exc:
        status, error = "invalid_response", str(exc)
        raise RuntimeError(f"{service} returned an invalid response: {exc}") from exc
    finally:
        log_api_call(service, operation, safe_params, started, status, error)
