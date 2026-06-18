# agr_coo Capability Specification

**Capability ID**: `agr_coo`

## Description

agr_coo capability for the APG platform.

## Provides

- `member_management`
- `pooled_input_procurement`
- `bulk_sales`
- `dividend_distribution`
- `cooperative_accounting`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
