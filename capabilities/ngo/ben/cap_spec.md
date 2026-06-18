# ngo_ben Capability Specification

**Capability ID**: `ngo_ben`

## Description

ngo_ben capability for the APG platform.

## Provides

- `beneficiary_registration`
- `vulnerability_assessment`
- `case_management`
- `deduplication`
- `beneficiary_analytics`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
