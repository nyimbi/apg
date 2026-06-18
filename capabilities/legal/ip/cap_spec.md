# leg_ip Capability Specification

**Capability ID**: `leg_ip`

## Description

leg_ip capability for the APG platform.

## Provides

- `ip_registration`
- `renewal_management`
- `ip_portfolio`
- `infringement_tracking`
- `ip_valuation`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
