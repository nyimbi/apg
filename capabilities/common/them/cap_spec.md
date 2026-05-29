# UI/UX Theming and Branding Capability Specification

- **Capability Name**: UI/UX Theming and Branding
- **Capability ID**: `them`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package provides the executable APG runtime for `them`.
It gives composed applications a deterministic theme and brand-governance
surface for theme creation, token versioning, licensed brand assets, live
preview evidence, accessibility contrast gates, governed publishing, large
rollout review, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `theme_registry`
- `token_versioning`
- `brand_asset_governance`
- `theme_preview_workflow`
- `theme_publication_governance`
- `theme_audit_events`

## Required Services

- `tenant_context`
- `identity_authorization`
- `accessibility_contrast_validation`
- `brand_asset_store`
- `audit_sink`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `theme_requires_owner`
- `publish_requires_approval`
- `brand_asset_requires_license`
- `accessible_contrast_required`
- `large_rollout_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

The view helpers provide dashboard, theme console, token editor, brand
guidelines, brand asset manager, live preview, publishing policies, and settings
models.

## Theme

The package uses the `them_brand_system` APG theme contract.

## Runtime Behavior

`ThemService` is intentionally dependency-light so it can run inside generated
applications, tests, and publish-plan probes without external infrastructure.
It supports:

- `create_theme()` for tenant-scoped design systems with owner, brand name,
  guidelines, and fallback theme metadata.
- `update_tokens()` for governed token groups with versioning and contrast
  validation evidence.
- `add_brand_asset()` for license-verified, approved brand assets.
- `create_preview()` for viewport/surface preview evidence and contrast status.
- `publish_theme()` for approval, contrast, and large-rollout review gates.
- `dashboard_summary()` and list helpers for API and UI composition.

## Adapter Boundaries

The in-package runtime stores records in memory by design. Production adapters
are expected to bind identity/authorization, durable asset storage,
accessibility validation engines, generated preview renderers, rollout
orchestration, and audit sinks at the APG composition layer without changing the
deterministic package contract.

## Focused Verification

- `./.venv/bin/python -m py_compile capabilities/common/them/__init__.py capabilities/common/them/models.py capabilities/common/them/theme_runtime.py capabilities/common/them/service.py capabilities/common/them/api.py capabilities/common/them/views.py capabilities/common/them/capability_contract.py capabilities/common/them/app.py capabilities/common/them/test_capability_contract.py capabilities/common/them/tests/test_package_contract.py`
- `./.venv/bin/pytest -q capabilities/common/them/test_capability_contract.py capabilities/common/them/tests/test_package_contract.py`
- `./.venv/bin/apg capabilities implementation-audit --root capabilities/common/them --json`
- `./.venv/bin/apg capabilities publish-plan capabilities/common/them --json`
