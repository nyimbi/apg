# Audit Log Capability Specification

**Capability ID**: `audit_log`

## Description

Audit Log capability for the APG platform.

## Provides

- `audit_log`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
