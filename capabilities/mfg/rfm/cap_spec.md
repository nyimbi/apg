# Repetitive Manufacturing Capability Specification

**Capability ID**: `mfg_rfm`

## Description

Repetitive Manufacturing capability for the APG platform.

## Provides

- `production_line_management`
- `rate_scheduling`
- `backflush_reporting`
- `takt_time_analysis`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
