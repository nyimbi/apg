# ins_reg Capability Specification

**Capability ID**: `ins_reg`

## Description

ins_reg capability for the APG platform.

## Provides

- `regulatory_submissions`
- `solvency_reporting`
- `statutory_returns`
- `compliance_monitoring`
- `regulator_portal`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
