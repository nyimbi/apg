# Product Costing Capability Specification

**Capability ID**: `mfg_pco`

## Description

Product Costing capability for the APG platform.

## Provides

- `standard_cost_management`
- `cost_rollup`
- `variance_analysis`
- `period_costing_close`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
