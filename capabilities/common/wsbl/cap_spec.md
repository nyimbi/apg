# Website Builder Capability Specification

- **Capability Name**: Website Builder
- **Capability ID**: `wsbl`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`wsbl` provides an executable website-builder runtime for APG applications. It
owns tenant sites, domain validation state, page composition, governed component
reuse, publication requests, rollback, audit events, UI route metadata, theme
metadata, and publish-plan evidence.

The package is dependency-light and deterministic. Live CMS stores, CDN
publishing, asset pipelines, visual editors, analytics collectors, consent
management systems, and public-hosting providers are adapter boundaries around
the local runtime, not prerequisites for package proof.

## Provided Services

- `site_management`
- `page_composition`
- `component_library`
- `publishing_workflows`
- `site_theming`
- `wsbl_operations`

## Required Services

- `tenant_context`
- `them` for visual theme alignment
- `auth` for permissions and actor identity
- `ncod` for governed no-code component composition

Optional adapters may integrate `i18n`, `accs`, `mchn`, and `cons` when a
capacity requires localization, accessibility scanning, machine/analytics
signals, or consent policy management.

## Runtime Behavior

The current package runtime is implemented in `website_runtime.py`,
`service.py`, `api.py`, and `views.py`.

Executable lifecycles:

- create tenant-owned sites with owner, locale, public-site, privacy-banner,
  domain, status, and required-action metadata;
- register and validate domains, marking sites `domain_pending` until the
  domain proof lands;
- create standard and custom components, forcing custom components through
  review before page use;
- create pages and add structured sections from approved components;
- create governed publish requests with approval, accessibility, and privacy
  consent-policy checks;
- publish approved requests and mark pages/sites published;
- rollback a published site to a previous version;
- expose dashboard, site console, page library, page editor, component library,
  publish queue, analytics, and settings view models;
- append audit events for site, domain, component, page, publishing, and
  rollback actions.

Compatibility helpers `create_record()` and `list_records()` remain available
for generic package tooling, but they delegate to site creation/listing rather
than storing generic records.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`.

Required configuration sections:

- `tenant_id`
- `sites`
- `pages`
- `publishing`
- `governance`
- `ui`
- `theme`

Important default controls:

- site owner required;
- domain validation required;
- multi-locale and environment preview enabled;
- structured sections required;
- custom component review required;
- publishing approval required;
- accessibility pass required;
- privacy banner consent policy required;
- tenant context and publication audit required.

## Rules

The deterministic rule engine exposes these rule IDs:

- `tenant_context_required`
- `site_requires_owner`
- `publish_requires_approval`
- `custom_component_requires_review`
- `public_site_requires_accessibility_pass`
- `privacy_banner_requires_consent_policy`

Service guardrails enforce the same decisions:

- creating a site without tenant context raises `tenant_context_required`;
- creating a site without an owner raises `site_owner_required`;
- using an unreviewed custom component in a page raises
  `component_review_required`;
- publishing without approval raises `site_publish_approval_required`;
- publishing a public site without accessibility evidence raises
  `accessibility_pass_required`;
- publishing with a privacy banner but without consent policy evidence creates
  a `review_required` publication request with required action
  `attach_consent_policy`.

## UI

The package exposes eight APG Python UI routes:

- `/wsbl/dashboard` via `WSBLDashboard`
- `/wsbl/sites` via `SiteConsole`
- `/wsbl/pages` via `PageLibrary`
- `/wsbl/editor` via `PageEditor`
- `/wsbl/components` via `ComponentLibrary`
- `/wsbl/publishing` via `PublishQueue`
- `/wsbl/analytics` via `SiteAnalytics`
- `/wsbl/settings` via `WSBLSettings`

`views.py` returns dependency-light view models for these routes. The view
models include route names, tenant context, relevant records, available
actions, summary counts, and theme/configuration metadata.

## Theme

The package uses the `wsbl_site_builder` APG theme contract. Current component
theme metadata covers site cards, page editor sections, publish queue states,
and analytics panels.

## Adapter Boundaries

Keep these integrations behind APG composition adapters:

- public CDN or static-host deployment;
- visual drag-and-drop editors;
- asset storage and image transformation;
- accessibility scanner engines;
- consent-management platforms;
- traffic analytics collectors;
- search indexing and sitemap generation;
- external CMS import/export;
- localized content translation.

Local package proof must remain deterministic without those providers.

## Focused Verification

Use these battery-conscious commands after WSBL package changes:

```bash
./.venv/bin/python -m py_compile capabilities/common/wsbl/__init__.py capabilities/common/wsbl/models.py capabilities/common/wsbl/website_runtime.py capabilities/common/wsbl/service.py capabilities/common/wsbl/api.py capabilities/common/wsbl/views.py capabilities/common/wsbl/capability_contract.py capabilities/common/wsbl/app.py capabilities/common/wsbl/test_capability_contract.py capabilities/common/wsbl/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/wsbl/test_capability_contract.py capabilities/common/wsbl/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/wsbl --json
./.venv/bin/apg capabilities publish-plan capabilities/common/wsbl --json
```

When global readiness changes, also run:

```bash
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```
