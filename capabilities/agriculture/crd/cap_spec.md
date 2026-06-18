# agr_crd Capability Specification

**Capability ID**: `agr_crd`

## Description

agr_crd capability for the APG platform.

## Provides

- `credit_scoring`
- `loan_eligibility`
- `group_lending`
- `credit_profile`
- `repayment_tracking`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
