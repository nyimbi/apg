# ngo_grn Capability Specification

**Capability ID**: `ngo_grn`

## Description

ngo_grn capability for the APG platform.

## Provides

- `grant_application`
- `award_management`
- `disbursement_tracking`
- `donor_reporting`
- `grant_compliance`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
