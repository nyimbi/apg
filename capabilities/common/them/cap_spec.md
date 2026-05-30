# UI/UX Theming and Branding Capability Packet

- Capability Name: UI/UX Theming and Branding
- Capability ID: `them`
- Category: common
- Version: 1.0.0

## Purpose

THEM provides executable APG theme, token, brand, preview, publication, and
visual-governance behavior. It lets generated applications compose tenant theme
systems, governed design tokens, licensed brand assets, preview evidence,
accessibility contrast gates, AI-assisted review lanes, audit trails, and
Bytewax lifecycle events.

## Provides

- `theme_tokens`
- `brand_governance`
- `asset_libraries`
- `preview_workflows`
- `theme_publication_governance`
- `visual_theming`
- `them_agents`

## Requires

- `conf`
- `auth`
- `i18n`
- `audl`
- `accs`

## Configuration Areas

THEM configuration is defined by `capability_contract.py` and covers:

- tenant context;
- theme ownership, fallback, preview, and guideline policy;
- governed token groups, versioning, contrast validation, and token review;
- brand asset license and approval rules;
- first-class theme-agent runtimes, roles, and human approval;
- publication governance and large-rollout review thresholds;
- Bytewax lifecycle-stream observability;
- adapter boundaries for identity, audit, assets, preview rendering, accessibility, and event streaming;
- UI route toggles and theme tokens.

## Lifecycle

THEM supports the following lifecycle:

1. Create a tenant theme with owner, brand, and guideline evidence.
2. Update governed design tokens with reviewer attribution and versioning.
3. Add licensed and approved brand assets.
4. Create surface and viewport preview evidence with contrast status.
5. Publish the theme through approval, contrast, rollout review, and Bytewax stream gates.
6. Register and govern AI agents that review tokens, brand assets, previews, accessibility, localization, and rollouts.
7. Record audit events for theme, token, asset, preview, publication, and agent activity.

## Deterministic Rules

- `tenant_context_required`
- `theme_requires_owner`
- `theme_requires_guidelines`
- `token_update_requires_reviewer`
- `brand_asset_requires_license`
- `brand_asset_requires_approval`
- `preview_requires_artifact`
- `publish_requires_approval`
- `accessible_contrast_required`
- `publish_requires_bytewax_stream`
- `large_rollout_requires_review`
- `them_agent_runtime_supported`
- `them_agent_role_supported`
- `privileged_agent_theme_action_requires_human_approval`
- `batch_theme_rollout_requires_bytewax`

## UI

THEM exposes APG Python view models for dashboard, theme console, token editor,
brand guidelines, brand asset manager, preview, agent workbench, policies, and
settings.

## Theme

THEM uses the `them_brand_system` theme with compact density, theme cards,
token tables, asset grids, preview shells, agent review lanes, and policy-rule
grids.

## Streaming

THEM lifecycle events are described by the Bytewax stream manifest:

- processor: `bytewax`
- stream: `apg.them.lifecycle`
- key: `tenant_id`
- events: `theme_created`, `tokens_updated`, `brand_asset_added`,
  `theme_preview_created`, `theme_published`, `them_agent_registered`

## Adapter Boundaries

The in-package service is dependency-light and stores records in memory for
generated apps, tests, and publish-plan probes. Production deployments should
bind identity providers, audit sinks, asset stores, preview renderers,
accessibility engines, rollout orchestrators, and Bytewax workers through APG
adapters without weakening the deterministic contract.

## Focused Verification

- `./.venv/bin/python -m py_compile capabilities/common/them/__init__.py capabilities/common/them/models.py capabilities/common/them/theme_runtime.py capabilities/common/them/service.py capabilities/common/them/api.py capabilities/common/them/views.py capabilities/common/them/capability_contract.py capabilities/common/them/app.py capabilities/common/them/test_capability_contract.py capabilities/common/them/tests/test_package_contract.py`
- `./.venv/bin/pytest -q capabilities/common/them/test_capability_contract.py capabilities/common/them/tests/test_package_contract.py`
- `./.venv/bin/apg capabilities implementation-audit --root capabilities/common/them --json`
- `./.venv/bin/apg capabilities publish-plan capabilities/common/them --json`
