# Incident Management Capability Specification

**Capability ID**: `itsm_inc`

## Description

Incident Management capability for the APG platform.

## Provides

- `incident_lifecycle_workflow`
- `incident_sla_tracking`
- `incident_escalation_workflow`
- `major_incident_workflow`
- `post_incident_review`
- `incident_analytics`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
