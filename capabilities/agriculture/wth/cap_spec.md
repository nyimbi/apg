# agr_wth Capability Specification

**Capability ID**: `agr_wth`

## Description

agr_wth capability for the APG platform.

## Provides

- `weather_forecast`
- `climate_analytics`
- `agri_alerts`
- `temperature_monitoring`
- `rainfall_tracking`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
