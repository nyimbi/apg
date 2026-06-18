# Tax Calculation Engine Capability Specification

**Capability ID**: `common_tax_calc`

## Description

Tax Calculation Engine capability for the APG platform.

## Provides

- `tax_calculation_workflow`
- `tax_rate_lookup`
- `tax_period_management`
- `tax_audit_trail`
- `tax_cross_capability_api`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
