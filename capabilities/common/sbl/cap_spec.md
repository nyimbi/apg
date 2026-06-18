# SaaS Billing Engine Capability Specification

**Capability ID**: `common_sbl`

## Description

SaaS Billing Engine capability for the APG platform.

## Provides

- `subscription_management`
- `usage_metering`
- `invoice_generation`
- `tenant_provisioning`
- `billing_analytics`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
