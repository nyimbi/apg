# THEM Capability Specification

## Identity

- Capability name: UI/UX Theming and Branding
- Capability ID: `them`
- Category: common
- Runtime target: APG Python capability package

## Mission

THEM gives generated APG applications a governed visual system. It coordinates
tenant theme records, token versioning, brand assets, preview evidence,
accessibility contrast gates, publication approvals, rollout review, visual
theming, AI-assisted review, audit events, and Bytewax lifecycle streaming.

## Functional Scope

THEM owns the executable lifecycle for:

- tenant-scoped theme creation and ownership;
- brand guideline evidence and fallback theme mapping;
- governed token groups for color, typography, spacing, density, and components;
- token versioning and reviewer attribution;
- licensed and approved brand assets;
- rendered preview evidence for surfaces and viewports;
- contrast validation before publication;
- approval and review gates before publication;
- large rollout review;
- first-class theme agents for design, accessibility, brand, preview,
  localization, and rollout review;
- batch theme rollout validation;
- audit and dashboard evidence.

## Configuration Contract

The configuration schema requires:

- `tenant_id`
- `themes`
- `tokens`
- `branding`
- `them_agents`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

THEM must expose these through `get_capability_contract()`, generated semantic
model evidence, and the package registration metadata.

## Domain Records

### Theme

A theme contains tenant, name, owner, brand name, status, guideline reference,
fallback theme, token version, and timestamps.

### Theme Token

A token record contains tenant, theme, token group, token values, version,
contrast validation status, reviewer attribution, and timestamp.

### Brand Asset

A brand asset contains tenant, theme, asset name, asset type, license reference,
approval actor, status, and timestamp.

### Preview

A preview contains tenant, theme, surface, viewport, preview artifact reference,
contrast result, creator, and timestamp.

### Publication

A publication contains tenant, theme, target-tenant count, approval reference,
status, publisher, matched rules, required actions, and timestamp.

### THEM Agent

A THEM agent is a first-class composition element with tenant, name, runtime,
role, scope, owner, status, and human approval policy.

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `design_token_reviewer`
- `accessibility_reviewer`
- `brand_reviewer`
- `preview_reviewer`
- `rollout_reviewer`
- `localization_reviewer`

## Lifecycle States

Theme states:

- `draft`
- `preview_ready`
- `approved`
- `published`
- `review_required`
- `blocked`

Asset states:

- `pending_license`
- `approved`
- `blocked`

## Rules

The deterministic rule engine must enforce:

- tenant context on all executable operations;
- owner on theme creation;
- guideline evidence on theme creation;
- reviewer attribution on token updates;
- license verification for brand assets;
- approval for brand assets;
- preview artifact before preview creation;
- approval before publication;
- contrast validation before publication;
- Bytewax event stream for publication;
- rollout review for broad target sets;
- approved theme-agent runtimes;
- approved theme-agent roles;
- human approval for privileged agent theme actions;
- Bytewax stream coordination for batch theme rollout.

## Service Requirements

`ThemService` must provide:

- `describe()`
- `evaluate()`
- `create_theme()`
- `update_tokens()`
- `add_brand_asset()`
- `create_preview()`
- `publish_theme()`
- `register_them_agent()`
- `validate_agent_theme_action()`
- `validate_batch_theme_rollout()`
- list helpers for every record type;
- `dashboard_summary()`.

## API Requirements

`api.py` must expose payload-oriented helpers for status, creation, token
updates, brand assets, previews, publishing, theme agents, agent-action
validation, batch rollout validation, compatibility record creation, and system
listing.

## UI Requirements

THEM exposes APG Python view models for:

- `/them/dashboard`
- `/them/themes`
- `/them/tokens`
- `/them/branding`
- `/them/assets`
- `/them/preview`
- `/them/agents`
- `/them/policies`
- `/them/settings`

The UI contract must expose rules, summaries, agent policy, Bytewax streaming,
and visual theme tokens.

## Visual Theming

The default visual theme is `them_brand_system`. It defines compact density,
theme pills, contrast bands, token tables, changed-token chips, license chips,
approval chips, review lanes, and guardrail chips.

## Streaming

THEM lifecycle events use Bytewax:

- processor: `bytewax`
- stream: `apg.them.lifecycle`
- key: `tenant_id`

Events:

- `theme_created`
- `tokens_updated`
- `brand_asset_added`
- `theme_preview_created`
- `theme_published`
- `them_agent_registered`

## Adapter Boundaries

The in-package runtime must stay dependency-light. Production deployments bind
identity providers, audit sinks, asset stores, preview renderers, accessibility
engines, rollout systems, and Bytewax workers through adapters.

## Acceptance Criteria

- README, specification, plan, and capability summary exist.
- Contract shape validates.
- Generated app evidence is refreshed from the contract.
- Tests cover contract, rules, service, API, views, agent guardrails, and
  Bytewax guardrails.
- Focused package tests pass.
- Implementation audit reports domain-specific behavior with no warnings.
- Publish plan reports side-effect-free output with no warnings.
