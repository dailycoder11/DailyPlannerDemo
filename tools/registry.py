"""LLM tool schemas and safe dispatcher."""

from __future__ import annotations

import json
from typing import Any, Callable

from tools.attractions import search_attractions
from tools.routing import get_travel_time
from tools.weather import get_weather

TOOLS: dict[str, Callable[..., Any]] = {
    "search_attractions": search_attractions,
    "get_weather": get_weather,
    "get_travel_time": get_travel_time,
}

TOOL_DEFINITIONS = [
    {"type": "function", "function": {"name": "search_attractions", "description": "Search OpenTripMap for useful attractions in a city or place. Returns attraction names, categories, coordinates and short descriptions.", "parameters": {"type": "object", "properties": {"place": {"type": "string", "description": "City or place name"}, "category": {"type": "string", "description": "Optional OpenTripMap attraction kind/category"}, "limit": {"type": "integer", "description": "Maximum results (1-10); defaults to 6"}}, "required": ["place"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Get daily and hourly weather for a trip date by city name or coordinates. Dates in the past use archive data; future dates are limited to forecast availability.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "trip_date": {"type": "string", "description": "Date in YYYY-MM-DD format"}, "latitude": {"type": "number"}, "longitude": {"type": "number"}}, "required": ["trip_date"]}}},
    {"type": "function", "function": {"name": "get_travel_time", "description": "Estimate driving distance and duration with OSRM. Supply places by name or coordinates; coordinates from search_attractions can be reused.", "parameters": {"type": "object", "properties": {"origin": {"type": "string", "description": "Place name or 'latitude,longitude', or object with name/latitude/longitude"}, "destination": {"type": "string", "description": "Place name or 'latitude,longitude', or object with name/latitude/longitude"}}, "required": ["origin", "destination"]}}},
]


def dispatch_tool(name: str, arguments: str | dict[str, Any]) -> Any:
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object")
    return TOOLS[name](**arguments)
