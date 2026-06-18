# ngo_rbf Capability Specification

**Capability ID**: `ngo_rbf`

## Description

ngo_rbf capability for the APG platform.

## Provides

- `rbf_contract_management`
- `result_verification`
- `disbursement_triggers`
- `compliance_reporting`
- `independent_verification`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
