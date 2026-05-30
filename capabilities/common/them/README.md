# THEM - UI/UX Theming and Branding

THEM is the APG capability for governed visual systems. It gives generated
applications a composable runtime for tenant theme records, design tokens, brand
assets, preview evidence, accessibility contrast gates, publication approvals,
AI-assisted review, and Bytewax lifecycle events.

Use THEM when an application needs consistent tenant branding, safe visual
customization, reviewable design-token changes, licensed brand assets, and
auditable publication workflows.

## What THEM Provides

- Tenant-scoped theme registry.
- Governed token versioning for color, typography, spacing, density, and
  component tokens.
- Brand guideline evidence and fallback theme mapping.
- Licensed and approved brand asset records.
- Preview evidence for APG surfaces and viewport sizes.
- Contrast validation and approval gates before publication.
- Large-rollout review guardrails.
- First-class THEM agents for Codex, Claude Code, OpenCode, and Pi based review
  lanes.
- Bytewax lifecycle stream metadata.
- Dashboard, console, token editor, asset manager, preview, agent, policy, and
  settings view models.

## Quick Start

```python
from capabilities.common.them import ThemService

service = ThemService()

theme = service.create_theme(
    tenant_id="tenant-a",
    name="Operations",
    owner="design-lead",
    brand_name="Operations Brand",
    guidelines_ref="brand://guidelines/operations",
)

service.update_tokens(
    tenant_id="tenant-a",
    theme_id=theme["id"],
    group="color",
    tokens={"color.primary": "#235789", "color.accent": "#F1A208"},
    updated_by="designer",
    contrast_validated=True,
)

service.add_brand_asset(
    tenant_id="tenant-a",
    theme_id=theme["id"],
    asset_name="primary-logo",
    asset_type="logo",
    license_ref="license://logo/1",
    approved_by="brand-owner",
)

service.create_preview(
    tenant_id="tenant-a",
    theme_id=theme["id"],
    surface="erp_shell",
    viewport="desktop",
    preview_ref="preview://theme/1",
    contrast_passed=True,
    created_by="designer",
)

publication = service.publish_theme(
    tenant_id="tenant-a",
    theme_id=theme["id"],
    published_by="release-manager",
    approval_ref="approval://theme/1",
    target_tenant_count=3,
)

print(publication["status"])
```

## THEM Agents

THEM treats theme review agents as governed composition elements.

```python
agent = service.register_them_agent(
    tenant_id="tenant-a",
    name="Contrast reviewer",
    runtime="codex",
    role="accessibility_reviewer",
    scope="review contrast and preview evidence",
)

decision = service.validate_agent_theme_action(
    tenant_id="tenant-a",
    agent_id=agent["id"],
    action="publish_theme",
    privileged_scope=True,
)

assert decision["decision"] == "deny"
```

Privileged agent theme actions require human approval:

```python
decision = service.validate_agent_theme_action(
    tenant_id="tenant-a",
    agent_id=agent["id"],
    action="publish_theme",
    privileged_scope=True,
    human_approval_ref="approval://agent/theme",
)

assert decision["decision"] == "allow"
```

## Batch Rollout Guardrail

Batch theme rollout must use Bytewax stream coordination:

```python
decision = service.validate_batch_theme_rollout(
    tenant_id="tenant-a",
    target_tenant_count=12,
    event_stream="bytewax",
    rollout_review_recorded=True,
)

assert decision["decision"] == "allow"
```

## Deterministic Rules

THEM enforces:

- tenant context on all executable operations;
- owner and guideline evidence for themes;
- reviewer attribution for token updates;
- license and approval for brand assets;
- preview artifact before preview creation;
- publication approval;
- accessibility contrast validation;
- Bytewax lifecycle stream metadata for publication;
- review for broad rollouts;
- supported theme-agent runtime and role;
- human approval for privileged agent actions;
- Bytewax coordination for batch rollout.

## API Helpers

`api.py` provides payload-oriented helpers:

- `capability_status()`
- `create_theme()`
- `update_tokens()`
- `add_brand_asset()`
- `create_preview()`
- `publish_theme()`
- `register_them_agent()`
- `validate_agent_theme_action()`
- `validate_batch_theme_rollout()`
- `create_record()`
- `list_records()`
- `list_theme_system()`

## UI Routes

- dashboard: `/them/dashboard`
- themes: `/them/themes`
- tokens: `/them/tokens`
- branding: `/them/branding`
- assets: `/them/assets`
- preview: `/them/preview`
- agents: `/them/agents`
- policies: `/them/policies`
- settings: `/them/settings`

## Bytewax Stream

THEM publishes lifecycle metadata for Bytewax:

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

The in-package service stores records in memory so generated applications,
tests, and publish-plan probes can execute without external infrastructure.
Production systems should attach identity providers, audit sinks, asset stores,
preview renderers, accessibility engines, rollout orchestrators, and Bytewax
workers through APG adapters.

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/them/__init__.py capabilities/common/them/capability_contract.py capabilities/common/them/models.py capabilities/common/them/theme_runtime.py capabilities/common/them/service.py capabilities/common/them/api.py capabilities/common/them/views.py capabilities/common/them/app.py capabilities/common/them/test_capability_contract.py capabilities/common/them/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/them/test_capability_contract.py capabilities/common/them/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/them --json
./.venv/bin/apg capabilities publish-plan capabilities/common/them --json
```
