# Website Builder Capability Packet

- Capability Name: Website Builder
- Capability ID: `wsbl`
- Category: common
- Version: 1.0.0

## Purpose

WSBL provides executable APG website-builder behavior for tenant sites, domains,
pages, components, publishing, rollback, privacy, accessibility, agent review,
audit, visual theming, and Bytewax lifecycle streams. It lets generated
applications compose public or internal sites from governed sections while
keeping publication controls explicit.

## Provides

- `site_management`
- `page_composition`
- `component_library`
- `publishing_workflows`
- `site_theming`
- `website_governance`
- `wsbl_agents`
- `review_evidence`

## Requires

- `them`
- `auth`
- `ncod`
- `accs`
- `cons`

## Configuration Areas

WSBL configuration is defined by `capability_contract.py` and covers:

- tenant context;
- site ownership, domain validation, locales, and preview evidence;
- page sections, custom component review, autosave, and versioning;
- publishing approval, accessibility pass, privacy consent policy, rollback, and stream routing;
- first-class website-builder agent runtimes, roles, and human approval;
- durable review evidence for custom components, publish requests, denied
  publish attempts, agent publish checks, batch publish checks, and audit
  events;
- audit, component policy, and public-site governance;
- Bytewax lifecycle-stream observability;
- adapter boundaries for theming, authorization, consent, accessibility, analytics, and event streaming;
- UI route toggles and theme tokens.

## Lifecycle

WSBL supports the following lifecycle:

1. Create a tenant site with owner, locale, public-site controls, privacy banner state, and optional domain.
2. Register and validate domains.
3. Create governed standard or custom components.
4. Review custom components with policy attribution.
5. Create pages and add structured sections.
6. Request publication with approval, validated domains, structured sections, preview evidence, accessibility pass, consent policy, and Bytewax stream metadata.
7. Publish approved requests and version the site/pages.
8. Roll back through Bytewax-governed lifecycle metadata.
9. Register governed AI agents that review sites, components, accessibility, privacy, publishing, and SEO evidence.
10. Compose pending review queues from persisted component and publish request
    policy evidence.

## Deterministic Rules

- `tenant_context_required`
- `site_requires_owner`
- `domain_requires_validation_before_publish`
- `page_requires_structured_sections`
- `preview_requires_evidence`
- `publish_requires_approval`
- `publish_requires_bytewax_stream`
- `custom_component_registration_requires_review`
- `custom_component_requires_review`
- `custom_component_requires_policy`
- `public_site_requires_accessibility_pass`
- `privacy_banner_requires_consent_policy`
- `rollback_requires_bytewax_stream`
- `batch_publish_requires_bytewax`
- `wsbl_agent_runtime_supported`
- `wsbl_agent_role_supported`
- `privileged_agent_publish_action_requires_human_approval`

## UI

WSBL exposes APG Python view models for dashboard, site console, page library,
page editor, component library, publish queue, analytics, agent workbench,
policy center, and settings.

The dashboard, component library, publish queue, and policy center expose
pending review queues and denied publish evidence so generated applications can
render approval workbenches directly from service state.

## Theme

WSBL uses the `wsbl_site_builder` theme with compact density, site cards,
publish bands, section builders, component chips, release checklists, approval
chips, traffic grids, trend chips, review lanes, and guardrail chips.

## Streaming

WSBL lifecycle events are described by the Bytewax stream manifest:

- processor: `bytewax`
- stream: `apg.wsbl.lifecycle`
- key: `tenant_id`
- events: `site_created`, `domain_registered`, `domain_validated`,
  `component_created`, `component_reviewed`, `page_created`,
  `page_section_added`, `publish_request_created`, `site_published`,
  `site_rolled_back`, `wsbl_agent_registered`

## Adapter Boundaries

The in-package service is dependency-light and stores records in memory for
generated apps, tests, and publish-plan probes. Production deployments should
bind visual editors, asset stores, preview renderers, accessibility scanners,
consent platforms, analytics collectors, CDN or static-host deployment,
search/sitemap systems, audit sinks, and Bytewax workers through APG adapters
without weakening the deterministic contract.

## Focused Verification

- `./.venv/bin/python -m py_compile capabilities/common/wsbl/__init__.py capabilities/common/wsbl/models.py capabilities/common/wsbl/website_runtime.py capabilities/common/wsbl/service.py capabilities/common/wsbl/api.py capabilities/common/wsbl/views.py capabilities/common/wsbl/capability_contract.py capabilities/common/wsbl/app.py capabilities/common/wsbl/test_capability_contract.py capabilities/common/wsbl/tests/test_package_contract.py`
- `./.venv/bin/pytest -q capabilities/common/wsbl/test_capability_contract.py capabilities/common/wsbl/tests/test_package_contract.py`
- `./.venv/bin/apg capabilities implementation-audit --root capabilities/common/wsbl --json`
- `./.venv/bin/apg capabilities publish-plan capabilities/common/wsbl --json`
