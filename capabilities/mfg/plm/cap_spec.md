# Product Lifecycle Management Capability Specification

**Capability ID**: `mfg_plm`

## Description

Product Lifecycle Management capability for the APG platform.

## Provides

- `product_portfolio`
- `npi_stage_gate`
- `design_release`
- `product_discontinuation`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
