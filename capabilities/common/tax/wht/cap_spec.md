# common_tax_wht Capability Specification

**Capability ID**: `common_tax_wht`

## Description

common_tax_wht capability for the APG platform.

## Provides

- `wht_rate_lookup`
- `wht_certificate_workflow`
- `wht_return_workflow`
- `wht_payment_record`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
