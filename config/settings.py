"""Settings loader for the Daytrip Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "app.json"


def load_settings() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        settings = json.load(config_file)
    if settings.get("provider") != "groq":
        raise ValueError("This application currently supports the Groq provider only.")
    return settings
