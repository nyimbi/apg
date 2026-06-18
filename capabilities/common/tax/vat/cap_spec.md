# common_tax_vat Capability Specification

**Capability ID**: `common_tax_vat`

## Description

common_tax_vat capability for the APG platform.

## Provides

- `vat_rate_lookup`
- `vat_return_workflow`
- `vat_exemption_registry`
- `vat_country_config`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
