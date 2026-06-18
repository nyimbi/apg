# hos_ana Capability Specification

**Capability ID**: `hos_ana`

## Description

hos_ana capability for the APG platform.

## Provides

- `revpar_analytics`
- `occupancy_reporting`
- `guest_satisfaction`
- `competitor_benchmarking`
- `forecast_reporting`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
