# leg_ctr Capability Specification

**Capability ID**: `leg_ctr`

## Description

leg_ctr capability for the APG platform.

## Provides

- `contract_drafting`
- `negotiation_management`
- `contract_execution`
- `obligation_tracking`
- `renewal_management`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
