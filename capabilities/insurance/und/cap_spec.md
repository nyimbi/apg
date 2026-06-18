# ins_und Capability Specification

**Capability ID**: `ins_und`

## Description

ins_und capability for the APG platform.

## Provides

- `risk_assessment`
- `premium_calculation`
- `acceptance_decision`
- `rating_engine`
- `underwriting_rules`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
