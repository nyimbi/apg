# ins_pol Capability Specification

**Capability ID**: `ins_pol`

## Description

ins_pol capability for the APG platform.

## Provides

- `policy_issuance`
- `endorsement_management`
- `renewal_processing`
- `cancellation_management`
- `policy_inquiry`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
