# WSBL Capability Specification

## Identity

- Capability name: Website Builder
- Capability ID: `wsbl`
- Category: common
- Runtime target: APG Python capability package

## Mission

WSBL gives generated APG applications a governed website-builder core. It
coordinates tenant sites, validated domains, versioned pages, governed
components, public-site policy, accessibility evidence, consent policy,
publication approvals, rollback, AI-assisted review, audit events, and Bytewax
lifecycle streaming.

## Functional Scope

WSBL owns the executable lifecycle for:

- tenant site creation and ownership;
- domain registration and validation;
- multi-locale and environment-preview metadata;
- custom and standard components;
- component review and policy attribution;
- page creation and structured section composition;
- publication requests and approvals;
- accessibility, consent, and preview gates;
- rollback of published site versions;
- first-class website-builder agents for site, component, accessibility,
  privacy, publish, and SEO review;
- website-builder audit and dashboard evidence.

## Configuration Contract

The configuration schema requires:

- `tenant_id`
- `sites`
- `pages`
- `publishing`
- `wsbl_agents`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

WSBL must expose these through `get_capability_contract()`, generated semantic
model evidence, and package registration metadata.

## Domain Records

### Site

A site contains tenant, name, owner, locale, public-site flag, privacy-banner
flag, status, domains, published version, required actions, metadata, and
timestamps.

### Domain

A domain contains tenant, site, domain, validation state, validation method,
creation timestamp, and validation timestamp.

### Component

A component contains tenant, name, type, custom flag, review state, reviewer,
policy, policy decision, matched rules, review reasons, audit evidence,
metadata, and timestamp.

### Page

A page contains tenant, site, slug, title, status, version, structured
sections, metadata, and timestamps.

### Publish Request

A publish request contains tenant, site, requester, environment, status,
approval state, accessibility state, consent-policy state, required actions,
policy decision, matched rules, review reasons, audit evidence, published
version, and timestamps.

### WSBL Agent

A WSBL agent is a first-class composition element with tenant, name, runtime,
role, scope, owner, status, human approval policy, and policy evidence fields.

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `site_reviewer`
- `component_reviewer`
- `accessibility_reviewer`
- `privacy_reviewer`
- `publish_reviewer`
- `seo_reviewer`

## Lifecycle States

Site states:

- `draft`
- `domain_pending`
- `ready`
- `published`
- `archived`

Page states:

- `draft`
- `review_ready`
- `published`
- `archived`

Publish states:

- `approved`
- `review_required`
- `published`
- `rolled_back`
- `denied`

Lifecycle stream states:

- `draft`
- `domain_pending`
- `ready`
- `review_required`
- `approved`
- `published`
- `rolled_back`
- `blocked`

## Rules

The deterministic rule engine must enforce:

- tenant context on all executable operations;
- owner on site creation;
- domain validation before publish;
- structured sections before publish;
- preview evidence before publish;
- approval before publish;
- Bytewax stream metadata for publish;
- review evidence for custom component registration;
- review before custom component use;
- policy attribution for custom component review;
- accessibility pass for public sites;
- consent policy for privacy banners;
- Bytewax stream metadata for rollback;
- Bytewax stream coordination for batch publishing;
- approved WSBL-agent runtimes;
- approved WSBL-agent roles;
- human approval for privileged agent publish actions.

## Service Requirements

`WsblService` must provide:

- `describe()`
- `evaluate()`
- `create_site()`
- `register_domain()`
- `validate_domain()`
- `create_component()`
- `review_component()`
- `create_page()`
- `add_page_section()`
- `create_publish_request()`
- `publish_site()`
- `rollback_site()`
- `register_wsbl_agent()`
- `validate_agent_publish_action()`
- `validate_batch_publish()`
- `list_pending_reviews()`
- list helpers for every record type;
- `dashboard_summary()`.

Review-required components and publish requests must expose `decision`,
`matched_rules`, `review_reasons`, and `audit_evidence`. Denied publish
requests must persist the same evidence before the service raises a hard
`PermissionError`. Agent publish-action and batch-publish validations must
write audit events with policy evidence.

## API Requirements

`api.py` must expose payload-oriented helpers for status, sites, domains,
components, pages, publish requests, publish, rollback, agents, agent-action
validation, batch publish validation, compatibility record creation, and system
listing.

## UI Requirements

WSBL exposes APG Python view models for:

- `/wsbl/dashboard`
- `/wsbl/sites`
- `/wsbl/pages`
- `/wsbl/editor`
- `/wsbl/components`
- `/wsbl/publishing`
- `/wsbl/analytics`
- `/wsbl/agents`
- `/wsbl/policy`
- `/wsbl/settings`

The UI contract must expose rules, summaries, agent policy, Bytewax streaming,
pending review queues, denied publish requests, and visual theme tokens.

## Visual Theming

The default visual theme is `wsbl_site_builder`. It defines compact density,
site cards, site pills, publish bands, section builders, component chips,
release checklists, approval chips, traffic grids, trend chips, review lanes,
and guardrail chips.

## Streaming

WSBL lifecycle events use Bytewax:

- processor: `bytewax`
- stream: `apg.wsbl.lifecycle`
- key: `tenant_id`

Events:

- `site_created`
- `domain_registered`
- `domain_validated`
- `component_created`
- `component_reviewed`
- `page_created`
- `page_section_added`
- `publish_request_created`
- `site_published`
- `site_rolled_back`
- `wsbl_agent_registered`

## Adapter Boundaries

The in-package runtime must stay dependency-light. Production deployments bind
visual editors, asset stores, preview renderers, accessibility scanners,
consent platforms, analytics collectors, CDN or static-host deployment,
search/sitemap systems, audit sinks, and Bytewax workers through adapters.

## Acceptance Criteria

- README, specification, plan, and capability summary exist.
- Contract shape validates.
- Generated app evidence is refreshed from the contract.
- Tests cover contract, rules, service, API, views, agent guardrails, and
  Bytewax guardrails.
- Focused package tests pass.
- Implementation audit reports domain-specific behavior with no warnings.
- Publish plan reports side-effect-free output with no warnings.
