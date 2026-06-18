# leg_bil Capability Specification

**Capability ID**: `leg_bil`

## Description

leg_bil capability for the APG platform.

## Provides

- `time_entry`
- `expense_capture`
- `invoice_generation`
- `trust_accounting`
- `billing_analytics`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
