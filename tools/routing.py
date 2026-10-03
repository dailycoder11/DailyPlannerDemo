"""OSRM route lookup with OpenStreetMap geocoding for place-name inputs."""

from __future__ import annotations

from typing import Any

from tools.common import get_json

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"


def _coordinates(place: str | dict[str, Any]) -> tuple[float, float, str]:
    if isinstance(place, dict):
        lat = place.get("latitude", place.get("lat"))
        lon = place.get("longitude", place.get("lon"))
        if lat is not None and lon is not None:
            return float(lon), float(lat), str(place.get("name", "coordinates"))
        place = str(place.get("name", ""))
    if not isinstance(place, str) or not place.strip():
        raise ValueError("origin and destination must be place names or coordinates")
    parts = place.split(",")
    if len(parts) == 2:
        try:
            lat, lon = float(parts[0]), float(parts[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lon, lat, place
        except ValueError:
            pass
    results = get_json("nominatim", "geocode", NOMINATIM_URL,
                       {"q": place.strip(), "format": "jsonv2", "limit": 1}, timeout=15.0,
                       headers={"User-Agent": "DaytripPlanner/1.0 (trip itinerary app)"})
    if not results or not isinstance(results, list):
        raise RuntimeError(f"Could not geocode '{place}'.")
    return float(results[0]["lon"]), float(results[0]["lat"]), place.strip()


def get_travel_time(origin: str | dict[str, Any], destination: str | dict[str, Any]) -> dict[str, Any]:
    """Return driving distance and estimated duration between two places."""
    origin_lon, origin_lat, origin_name = _coordinates(origin)
    dest_lon, dest_lat, dest_name = _coordinates(destination)
    coordinates = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    data = get_json("osrm", "route", f"{OSRM_URL}/{coordinates}",
                    {"overview": "false", "alternatives": "false", "steps": "false"})
    routes = data.get("routes") or []
    if data.get("code") != "Ok" or not routes:
        raise RuntimeError(f"OSRM could not find a route ({data.get('code', 'no route')}).")
    route = routes[0]
    try:
        distance_km = round(float(route["distance"]) / 1000, 1)
        duration_minutes = round(float(route["duration"]) / 60)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("OSRM returned an invalid route response.") from exc
    return {"origin": origin_name, "destination": dest_name, "distanceKm": distance_km,
            "durationMinutes": duration_minutes, "mode": "driving", "source": "OSRM"}
