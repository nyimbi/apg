# Consent and Privacy Management Capability Specification

- **Capability Name**: Consent and Privacy Management
- **Capability ID**: `cons`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package provides the executable APG consent and privacy runtime for
`cons`. It owns tenant privacy purposes, published notices, consent events,
preference profiles, consent-gated processing decisions, privacy requests, and
audit-event metadata behind the APG capability contract.

The implementation is dependency-light and deterministic so generated APG
applications can compose consent and privacy behavior without requiring a live
identity provider, DLP engine, document repository, audit-log sink, marketing
platform, or regulator integration. Those integrations remain explicit APG
capability boundaries.

## Provided Services

- `purpose_registry`
- `privacy_notice_publication`
- `consent_capture`
- `preference_center`
- `consent_gated_processing`
- `privacy_request_fulfillment`
- `privacy_audit_events`

## Required Services

- `comp`
- `auth`
- `dlpd`

Optional composition targets include `i18n`, `audl`, `mchn`, and `wsbl`.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `purpose_requires_legal_basis`
- `consent_capture_requires_notice`
- `processing_requires_active_consent`
- `privacy_request_requires_identity_verification`
- `stale_consent_requires_review`

## UI

The package exposes APG Python route contracts through `views.py` and provides
dashboard, purpose registry, privacy notices, consent ledger, preference
center, privacy request queue, processing-decision, and audit timeline view
models.

## Theme

The package uses the `cons_privacy_center` APG theme contract.
