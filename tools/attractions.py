"""OpenTripMap attraction search."""

from __future__ import annotations

import os
from typing import Any

from tools.common import get_json, log_api_call

BASE_URL = "https://api.opentripmap.com/0.1/en/places"


def search_attractions(place: str, category: str | None = None, limit: int = 6) -> list[dict[str, Any]]:
    """Find a small list of attractions in a city/place, optionally by category."""
    api_key = os.getenv("OPENTRIPMAP_API_KEY")
    if not api_key or api_key == "your_opentripmap_api_key":
        raise RuntimeError("OPENTRIPMAP_API_KEY is not configured.")
    if not place or not place.strip():
        raise ValueError("place must be provided")
    limit = max(1, min(int(limit), 10))
    safe_input = {"place": place.strip(), "category": category, "limit": limit}

    geo = get_json("opentripmap", "geoname", f"{BASE_URL}/geoname", {
        "name": place.strip(), "apikey": api_key,
    }, secret_params={"apikey"})
    try:
        lat, lon = float(geo["lat"]), float(geo["lon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"OpenTripMap could not resolve the place '{place}'.") from exc

    radius_params: dict[str, Any] = {
        "radius": 12000,
        "lon": lon,
        "lat": lat,
        "rate": 2,
        "format": "json",
        "limit": limit,
        "apikey": api_key,
    }
    if category:
        radius_params["kinds"] = category
    places = get_json("opentripmap", "radius", f"{BASE_URL}/radius", radius_params,
                      secret_params={"apikey"})
    if isinstance(places, list):
        candidates = places
    elif isinstance(places.get("features"), list):
        candidates = places["features"]
    elif isinstance(places.get("places"), list):
        candidates = places["places"]
    else:
        candidates = []
    results = []
    for candidate in candidates[:limit]:
        try:
            item = _details(candidate, api_key)
            if item:
                results.append(item)
        except (RuntimeError, ValueError, KeyError):
            # Each failed details request is logged by get_json; keep other hits.
            continue
    return results


def _details(place: dict[str, Any], api_key: str) -> dict[str, Any] | None:
    xid = place.get("xid")
    if not xid:
        return None
    details = get_json("opentripmap", "place_details", f"https://api.opentripmap.com/0.1/en/places/xid/{xid}",
                       {"apikey": api_key}, secret_params={"apikey"})
    point = details.get("point") or {}
    kinds = details.get("kinds") or place.get("kinds")
    result: dict[str, Any] = {
        "name": details.get("name") or place.get("name") or "Unnamed attraction",
        "category": kinds,
        "latitude": point.get("lat", place.get("lat")),
        "longitude": point.get("lon", place.get("lon")),
        "source": "OpenTripMap",
    }
    description = (details.get("wikipedia_extracts") or {}).get("text")
    if description:
        result["description"] = description[:500]
    return result
