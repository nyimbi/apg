# ngo_don Capability Specification

**Capability ID**: `ngo_don`

## Description

ngo_don capability for the APG platform.

## Provides

- `donor_profiles`
- `giving_history`
- `pledge_management`
- `stewardship_activities`
- `donation_receipting`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
