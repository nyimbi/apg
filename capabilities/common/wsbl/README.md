# WSBL - Website Builder

WSBL is the APG capability for governed website and page composition. It gives
generated applications a composable runtime for tenant sites, domains, pages,
components, public-site controls, publishing, rollback, accessibility,
privacy-consent policy, AI-assisted review, and Bytewax lifecycle events.

Use WSBL when an application needs a website builder that can move from draft
composition to controlled publication without skipping governance evidence.

## What WSBL Provides

- Tenant-scoped site registry.
- Domain registration and validation state.
- Versioned pages built from structured sections.
- Governed standard and custom component library.
- Custom component review and policy attribution.
- Publication requests with approval, domain, section, preview, accessibility,
  and privacy-consent gates.
- Durable review evidence for custom components, publish requests, denied
  publish attempts, agent publish checks, batch publish checks, and audit
  events.
- Rollback of published site versions.
- First-class WSBL agents for Codex, Claude Code, OpenCode, and Pi based review
  lanes.
- Bytewax lifecycle stream metadata.
- Dashboard, site, page, editor, component, publishing, analytics, agent,
  policy, and settings view models.

## Quick Start

```python
from capabilities.common.wsbl import WsblService

service = WsblService()

site = service.create_site(
    site_key="marketing",
    tenant_id="tenant-a",
    name="Marketing Site",
    owner_id="owner-1",
    primary_domain="www.example.test",
    domain_validated=True,
)

component = service.create_component(
    component_key="hero",
    tenant_id="tenant-a",
    name="Hero Banner",
    custom=True,
    reviewed=True,
    reviewed_by="reviewer-1",
    policy_id="component-policy",
)

page = service.create_page(
    site_id=site["id"],
    slug="home",
    title="Home",
    tenant_id="tenant-a",
)

page = service.add_page_section(
    page_id=page["id"],
    component_id=component["id"],
    content={"headline": "Welcome"},
    actor_id="editor-1",
)
```

## Publishing

Publishing requires approval, validated domains, structured sections, preview
evidence, accessibility evidence, consent-policy handling, and Bytewax stream
metadata.

```python
request = service.create_publish_request(
    site_id=site["id"],
    requested_by="publisher-1",
    approval_recorded=True,
    accessibility_passed=True,
    consent_policy_attached=True,
    preview_evidence_present=True,
    event_stream="bytewax",
)

published = service.publish_site(
    publish_request_id=request["id"],
    actor_id="publisher-1",
)
```

## WSBL Agents

WSBL treats website-builder agents as governed composition elements.

```python
agent = service.register_wsbl_agent(
    tenant_id="tenant-a",
    name="Publish reviewer",
    runtime="codex",
    role="publish_reviewer",
    scope="review publish evidence and accessibility gates",
)

decision = service.validate_agent_publish_action(
    tenant_id="tenant-a",
    agent_id=agent["id"],
    action="publish_site",
    privileged_scope=True,
)

assert decision["decision"] == "deny"
```

Privileged agent publish actions require human approval:

```python
decision = service.validate_agent_publish_action(
    tenant_id="tenant-a",
    agent_id=agent["id"],
    action="publish_site",
    privileged_scope=True,
    human_approval_ref="approval://agent/publish",
)

assert decision["decision"] == "allow"
```

## Review Evidence

WSBL keeps review and denial evidence durable. Custom components created
without review are stored as `review_required` with `decision`,
`matched_rules`, `review_reasons`, and `audit_evidence`. Publish requests that
need consent-policy review are stored the same way, while hard-denied publish
attempts are persisted as `denied` before `PermissionError` is raised.

Generated applications can call `list_pending_reviews()` or use the dashboard,
component library, publish queue, and policy-center view models to compose
approval queues without replaying policy rules.

## Batch Publish Guardrail

Batch publishing must use Bytewax stream coordination.

```python
decision = service.validate_batch_publish(
    tenant_id="tenant-a",
    site_count=3,
    event_stream="bytewax",
)

assert decision["decision"] == "allow"
```

## Deterministic Rules

WSBL enforces:

- tenant context on all executable operations;
- site ownership;
- domain validation before publishing;
- structured sections before publishing;
- preview evidence before publishing;
- approval before publishing;
- Bytewax stream metadata for publish and rollback;
- custom component review and policy attribution;
- accessibility pass for public sites;
- consent policy review for privacy banners;
- Bytewax coordination for batch publishing;
- supported WSBL-agent runtime and role;
- human approval for privileged agent actions.

## API Helpers

`api.py` provides payload-oriented helpers:

- `capability_status()`
- `create_site()`
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
- `create_record()`
- `list_records()`
- `list_website_builder()`

## UI Routes

- dashboard: `/wsbl/dashboard`
- sites: `/wsbl/sites`
- pages: `/wsbl/pages`
- editor: `/wsbl/editor`
- components: `/wsbl/components`
- publishing: `/wsbl/publishing`
- analytics: `/wsbl/analytics`
- agents: `/wsbl/agents`
- policy: `/wsbl/policy`
- settings: `/wsbl/settings`

## Bytewax Stream

WSBL publishes lifecycle metadata for Bytewax:

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

The in-package service stores records in memory so generated applications,
tests, and publish-plan probes can execute without external infrastructure.
Production systems should attach visual editors, asset stores, preview
renderers, accessibility scanners, consent platforms, analytics collectors, CDN
or static-host deployment, search/sitemap systems, audit sinks, and Bytewax
workers through APG adapters.

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/wsbl/__init__.py capabilities/common/wsbl/capability_contract.py capabilities/common/wsbl/models.py capabilities/common/wsbl/website_runtime.py capabilities/common/wsbl/service.py capabilities/common/wsbl/api.py capabilities/common/wsbl/views.py capabilities/common/wsbl/app.py capabilities/common/wsbl/test_capability_contract.py capabilities/common/wsbl/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/wsbl/test_capability_contract.py capabilities/common/wsbl/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/wsbl --json
./.venv/bin/apg capabilities publish-plan capabilities/common/wsbl --json
```
