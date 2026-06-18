# Configuration Management Database Capability Specification

**Capability ID**: `itsm_cmdb`

## Description

Configuration Management Database capability for the APG platform.

## Provides

- `cmdb_ci_registry`
- `cmdb_relationship_graph`
- `cmdb_discovery_workflow`
- `cmdb_change_tracking`
- `cmdb_health_scoring`
- `cmdb_dependency_map`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
