# hos_loy Capability Specification

**Capability ID**: `hos_loy`

## Description

hos_loy capability for the APG platform.

## Provides

- `points_management`
- `tier_management`
- `redemption`
- `member_benefits`
- `loyalty_campaigns`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
