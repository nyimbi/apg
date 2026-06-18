# common_ussd Capability Specification

**Capability ID**: `common_ussd`

## Description

common_ussd capability for the APG platform.

## Provides

- `ussd_session_management`
- `menu_rendering`
- `mpesa_callback`
- `gateway_integration`
- `i18n_menus`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
