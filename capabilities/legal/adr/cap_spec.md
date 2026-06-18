# leg_adr Capability Specification

**Capability ID**: `leg_adr`

## Description

leg_adr capability for the APG platform.

## Provides

- `dispute_filing`
- `mediator_assignment`
- `arbitration_management`
- `settlement_recording`
- `customary_adr`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
