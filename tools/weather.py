"""Open-Meteo geocoding and daily/hourly forecast lookup."""

from __future__ import annotations

from datetime import date
from typing import Any

from tools.common import get_json

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Drizzle",
    55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Freezing drizzle",
    61: "Slight rain", 63: "Rain", 65: "Heavy rain", 66: "Light freezing rain",
    67: "Heavy freezing rain", 71: "Slight snow", 73: "Snow", 75: "Heavy snow",
    77: "Snow grains", 80: "Slight rain showers", 81: "Rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def get_weather(city: str | None = None, trip_date: str | date | None = None,
                latitude: float | None = None, longitude: float | None = None) -> dict[str, Any]:
    """Get daily weather and useful hourly conditions for a city or coordinates."""
    if not trip_date:
        raise ValueError("trip_date must be provided as YYYY-MM-DD")
    day = date.fromisoformat(trip_date) if isinstance(trip_date, str) else trip_date
    if latitude is None or longitude is None:
        if not city or not city.strip():
            raise ValueError("Provide a city or both latitude and longitude")
        results = get_json("open_meteo", "geocoding", GEOCODING_URL,
                           {"name": city.strip(), "count": 1, "language": "en", "format": "json"})
        matches = results.get("results") or []
        if not matches:
            raise RuntimeError(f"Open-Meteo could not find '{city}'.")
        latitude = float(matches[0]["latitude"])
        longitude = float(matches[0]["longitude"])
        location = {key: matches[0].get(key) for key in ("name", "admin1", "country") if matches[0].get(key)}
    else:
        latitude, longitude = float(latitude), float(longitude)
        location = {"latitude": latitude, "longitude": longitude}
        if city:
            location["name"] = city

    today = date.today()
    base_url = ARCHIVE_URL if day < today else FORECAST_URL
    daily_fields = ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
    if base_url == FORECAST_URL:
        daily_fields.append("precipitation_probability_max")
    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(daily_fields),
        "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code",
        "timezone": "auto",
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
    }
    # Forecast endpoint exposes forecast_days rather than start/end dates.
    if base_url == FORECAST_URL:
        params.pop("start_date")
        params.pop("end_date")
        params["forecast_days"] = min(max((day - today).days + 1, 1), 16)
    data = get_json("open_meteo", "forecast" if base_url == FORECAST_URL else "archive", base_url, params)

    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    try:
        index = dates.index(day.isoformat())
    except ValueError as exc:
        raise RuntimeError(f"Open-Meteo has no weather data for {day.isoformat()}.") from exc
    daily_values = {}
    for key, output in (("temperature_2m_min", "temperatureMinC"),
                         ("temperature_2m_max", "temperatureMaxC"),
                         ("precipitation_probability_max", "precipitationProbabilityMaxPct"),
                         ("precipitation_sum", "precipitationMm"),
                         ("weather_code", "weatherCode")):
        values = daily.get(key) or []
        daily_values[output] = values[index] if index < len(values) else None
    daily_values["condition"] = WEATHER_CODES.get(daily_values.get("weatherCode"), "Unknown")

    hourly_data = data.get("hourly") or {}
    times = hourly_data.get("time") or []
    hourly = []
    for idx, timestamp in enumerate(times):
        if timestamp[:10] != day.isoformat():
            continue
        hour_entry = {"time": timestamp}
        for key, output in (("temperature_2m", "temperatureC"),
                            ("precipitation_probability", "precipitationProbabilityPct"),
                            ("precipitation", "precipitationMm"),
                            ("weather_code", "weatherCode")):
            values = hourly_data.get(key) or []
            hour_entry[output] = values[idx] if idx < len(values) else None
        hour_entry["condition"] = WEATHER_CODES.get(hour_entry.get("weatherCode"), "Unknown")
        hourly.append(hour_entry)
    return {"date": day.isoformat(), "location": location, "timezone": data.get("timezone"),
            **daily_values, "hourly": hourly}
