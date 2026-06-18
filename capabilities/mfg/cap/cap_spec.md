# Capacity Planning Capability Specification

**Capability ID**: `mfg_cap`

## Description

Capacity Planning capability for the APG platform.

## Provides

- `work_centre_capacity`
- `capacity_load_analysis`
- `constraint_identification`
- `capacity_simulation`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
