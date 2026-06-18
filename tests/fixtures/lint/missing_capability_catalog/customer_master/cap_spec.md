# Customer Master Capability Specification

**Capability ID**: `customer_master`

## Description

Customer Master capability for the APG platform.

## Provides

- `customer_master`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
