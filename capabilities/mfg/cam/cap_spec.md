# Computer-Aided Manufacturing Capability Specification

**Capability ID**: `mfg_cam`

## Description

Computer-Aided Manufacturing capability for the APG platform.

## Provides

- `cnc_program_management`
- `tool_library`
- `cutting_parameters`
- `nc_post_processing`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
