"""
Open-Meteo weather service — synchronous HTTP calls via httpx.

All public functions return plain dicts so callers need no extra imports.
"""

import httpx

# WMO Weather Interpretation Codes → human-readable label
WMO_DESCRIPTIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Icy fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Light rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Light snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Light showers",
    81: "Moderate showers",
    82: "Heavy showers",
    85: "Light snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}

# Codes where outdoor activity is discouraged
_UNFAVORABLE: set[int] = {45, 48, 51, 53, 55, 61, 63, 65, 71, 73, 75, 80, 81, 82, 85, 86, 95, 96, 99}


def fetch_forecast(latitude: float, longitude: float, timezone: str, date: str) -> dict:
    """
    Call Open-Meteo forecast API and return raw JSON.
    Raises httpx.HTTPError on network/HTTP failure.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation_probability,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": timezone,
        "start_date": date,
        "end_date": date,
        "wind_speed_unit": "kmh",
    }
    with httpx.Client(timeout=10.0) as client:
        r = client.get("https://api.open-meteo.com/v1/forecast", params=params)
        r.raise_for_status()
        return r.json()


def geocode_city(city: str) -> dict | None:
    """
    Use Open-Meteo geocoding API to resolve a city name to lat/lon.
    Returns {"latitude": ..., "longitude": ..., "name": ..., "country": ...} or None.
    """
    with httpx.Client(timeout=8.0) as client:
        r = client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
        )
        r.raise_for_status()
        results = r.json().get("results") or []
        if not results:
            return None
        top = results[0]
        return {
            "latitude": top["latitude"],
            "longitude": top["longitude"],
            "name": top.get("name", city),
            "country": top.get("country", ""),
            "admin1": top.get("admin1", ""),
        }


def summarize_forecast(
    data: dict,
    work_start: str,
    work_end: str,
    location_name: str = "",
) -> dict:
    """
    Convert raw Open-Meteo JSON into an agent-readable summary dict.

    Returns keys: location, temperature_range, precipitation_mm,
    outdoor_conditions ("good"/"moderate"/"poor"), precipitation_risk (bool),
    best_outdoor_window (str|None), summary (str), hourly_work_hours (list).
    """
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precips = hourly.get("precipitation_probability", [])
    codes = hourly.get("weather_code", [])
    winds = hourly.get("wind_speed_10m", [])
    units = data.get("hourly_units", {})
    temp_unit = units.get("temperature_2m", "°C")

    daily = data.get("daily", {})
    temp_max = (daily.get("temperature_2m_max") or [None])[0]
    temp_min = (daily.get("temperature_2m_min") or [None])[0]
    precip_sum = (daily.get("precipitation_sum") or [0])[0] or 0

    ws_h = int(work_start.split(":")[0])
    we_h = int(work_end.split(":")[0])

    work_hours = []
    for i, t in enumerate(times):
        try:
            h = int(t.split("T")[1].split(":")[0])
        except (IndexError, ValueError):
            continue
        if ws_h <= h < we_h:
            code = codes[i] if i < len(codes) else 0
            work_hours.append({
                "time": t.split("T")[1][:5],
                "temp": temps[i] if i < len(temps) else None,
                "precip_prob": precips[i] if i < len(precips) else 0,
                "wind_kmh": winds[i] if i < len(winds) else None,
                "code": code,
                "description": WMO_DESCRIPTIONS.get(code, "Unknown"),
                "unfavorable": code in _UNFAVORABLE,
            })

    if not work_hours:
        return {"error": "No hourly forecast data available for the work window."}

    bad = [h for h in work_hours if h["unfavorable"]]
    high_precip = [h for h in work_hours if (h["precip_prob"] or 0) >= 50]
    good = [h for h in work_hours if not h["unfavorable"] and (h["precip_prob"] or 0) < 30]

    if bad or len(high_precip) > len(work_hours) // 2:
        outdoor_conditions = "poor"
    elif high_precip:
        outdoor_conditions = "moderate"
    else:
        outdoor_conditions = "good"

    temp_range = (
        f"{temp_min}{temp_unit}–{temp_max}{temp_unit}"
        if temp_min is not None and temp_max is not None
        else "N/A"
    )

    best_window = f"{good[0]['time']}–{good[-1]['time']}" if good else None

    lines = [f"Weather for {location_name or 'your location'}:"]
    lines.append(f"  Temperature: {temp_range}, Precipitation: {precip_sum:.1f}mm")

    if bad:
        bad_labels = sorted({h["description"] for h in bad})
        lines.append(f"  Adverse conditions: {', '.join(bad_labels)}")
        lines.append(f"  Affected hours: {', '.join(h['time'] for h in bad)}")

    if high_precip:
        lines.append(
            f"  Rain probability ≥50%: {', '.join(h['time'] for h in high_precip)}"
        )

    condition_msg = {
        "good": "Outdoor conditions look good during work hours.",
        "moderate": "Mixed conditions — some rain risk during work hours.",
        "poor": "Poor outdoor conditions — rain, snow, or storms expected.",
    }[outdoor_conditions]
    lines.append(f"  {condition_msg}")

    if best_window:
        lines.append(f"  Best outdoor window: {best_window}")

    return {
        "location": location_name or "Unknown",
        "temperature_range": temp_range,
        "precipitation_mm": float(precip_sum),
        "outdoor_conditions": outdoor_conditions,
        "precipitation_risk": len(high_precip) > 0,
        "best_outdoor_window": best_window,
        "summary": "\n".join(lines),
        "hourly_work_hours": [
            {
                "time": h["time"],
                "temp": h["temp"],
                "temp_unit": temp_unit,
                "precip_probability": h["precip_prob"],
                "weather": h["description"],
                "wind_kmh": h["wind_kmh"],
            }
            for h in work_hours
        ],
    }