# THEM - UI/UX Theming and Branding

THEM is the APG capability for governed visual systems. It gives generated
applications a composable runtime for tenant theme records, design tokens, brand
assets, preview evidence, accessibility contrast gates, publication approvals,
AI-assisted review, and Bytewax lifecycle events.

Use THEM when an application needs consistent tenant branding, safe visual
customization, reviewable design-token changes, licensed brand assets, and
auditable publication workflows.

## What THEM Provides

- Tenant-scoped theme registry with parent-child inheritance graph.
- Governed token versioning for color, typography, spacing, density, and
  component tokens.
- Semantic token aliases (`{color.brand.blue.500}`) with recursive resolution.
- Brand guideline evidence and fallback theme mapping.
- Licensed and approved brand asset records.
- Preview evidence for APG surfaces and viewport sizes, including responsive
  multi-breakpoint generation.
- Multi-surface WCAG contrast matrix (all foreground/background pairs).
- Contrast validation and approval gates before publication.
- Large-rollout review guardrails with canary cohort strategy.
- Immutable point-in-time theme snapshots for compliance time-travel.
- Multi-dimension governance scorecard (token freshness, a11y, licensing,
  publication governance, brand coverage).
- First-class THEM agents for Codex, Claude Code, OpenCode, and Pi review lanes.
- Bytewax lifecycle stream metadata.
- NATS JetStream delivery for async publish, token-update, rollout progress,
  and real-time preview events.
- Component-level token overrides and white-label configuration.
- CSS custom properties export and Style Dictionary / DTCG format support.
- Dark-mode variant generation (naive invert; OKLCH perceptual strategy in v2.0).
- AI-assisted colour palette generation seeded from a single brand hex.
- Figma Variables two-way sync via Figma REST API.
- Dashboard, console, token editor, asset manager, preview, agent, policy, and
  settings view models.

---

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

---

## New Methods

### Async publish and token update with NATS

```python
import asyncio

async def main():
    service = ThemService()

    result = await service.async_publish_theme(
        tenant_id="tenant-a",
        theme_id=theme_id,
        published_by="release-manager",
        approval_ref="approval://1",
        nats_client=nats_client,  # optional; omit for in-process only
    )

    await service.async_update_tokens(
        tenant_id="tenant-a",
        theme_id=theme_id,
        group="color",
        tokens={"color.primary": "#0052CC"},
        updated_by="designer",
        nats_client=nats_client,
    )

asyncio.run(main())
```

### Token diff and rollback

```python
async def inspect():
    diff = await service.token_diff(
        tenant_id="tenant-a",
        theme_id=theme_id,
        from_version=1,
        to_version=3,
    )
    print(diff["changed"])  # {"color.primary": {"old": "#aaa", "new": "#0052CC"}}

    await service.token_rollback(
        tenant_id="tenant-a",
        theme_id=theme_id,
        group="color",
        target_version=1,
        rolled_back_by="design-lead",
    )
```

### Multi-surface contrast matrix

```python
matrix = await service.contrast_matrix(
    tenant_id="tenant-a",
    theme_id=theme_id,
    wcag_level="AA",
)
print(f"{matrix['pass_rate_pct']}% of pairs pass WCAG AA")
for fail in matrix["failures"]:
    print(f"{fail['fg_token']} on {fail['bg_token']}: {fail['ratio']}:1")
```

### Theme snapshots (time-travel compliance)

```python
snap = await service.snapshot_theme(
    tenant_id="tenant-a",
    theme_id=theme_id,
    label="pre-v2-release",
    snapshotted_by="release-manager",
)

await service.restore_theme_snapshot(
    tenant_id="tenant-a",
    snapshot_id=snap["id"],
    restored_by="release-manager",
)
```

### Canary rollout with accessibility halt

```python
result = await service.canary_rollout(
    tenant_id="platform",
    theme_id=theme_id,
    target_tenant_ids=["t1", "t2", ..., "t1000"],
    cohort_size=50,
    halt_on_violation_rate=0.05,  # halt if >5% WCAG violations
    applied_by="platform-ops",
    nats_client=nats_client,
)
print(result["applied_count"], result["halted"])
```

### Governance scorecard

```python
scorecard = await service.governance_scorecard(
    tenant_id="tenant-a",
    period_days=30,
)
print(scorecard["grade"])          # "A", "B", ..., "F"
print(scorecard["overall_score"])  # 0-100
print(scorecard["dimensions"])     # per-dimension breakdown
```

---

## World-Class Enhancements (v2.0)

| # | Name | Category | Summary |
|---|------|----------|---------|
| I1 | Async-Native Service Layer | Architecture | All public methods are `async def`; `asyncio.gather` for fan-out; DI-injected async adapters. Eliminates thread-pool overhead, enables 10-50x concurrency. |
| I2 | NATS JetStream Lifecycle Events | Streaming | `publish_theme`, `update_tokens`, `add_brand_asset`, `apply_tenant_theme` each publish to `apg.them.<event_type>.<tenant_id>` after committing state. Sub-millisecond durable delivery without a Kafka broker. |
| I3 | Perceptual Dark-Mode via OKLCH | Colour Science | Flips only the L channel in OKLCH (not RGB), preserving hue and chroma. Produces perceptually uniform dark palettes that pass WCAG contrast without manual correction. Exposed as `dark_mode_oklch` strategy. |
| I4 | Live Token Diff and Rollback | Governance | `token_diff(from_version, to_version)` returns added/removed/changed keys. `token_rollback(group, target_version)` re-applies historical values as a new version; history is never mutated. |
| I5 | Multi-Surface Contrast Matrix | Accessibility | `contrast_matrix()` computes WCAG ratios for every fg/bg token pair, not just against white. Returns violation report with fix suggestions per pair; critical for WCAG 2.1 AA/AAA certification. |
| I6 | W3C DTCG Schema Validation | Interoperability | `validate_dtcg()` exports tokens as DTCG JSON and validates against the published schema. `export_design_tokens` gains a `dtcg` format option. Ensures round-trip fidelity with Figma, Style Dictionary 4, Tokens Studio. |
| I7 | Tenant Theme Inheritance Graph | Composability | `resolve_token_graph()` walks `fallback_theme_id` pointers (cycle-guarded, depth 10), merges bottom-up, returns resolved token map with per-key provenance. `theme_inheritance_graph()` returns adjacency list + depth metrics. |
| I8 | Real-Time Preview via NATS WebSocket Bridge | Developer Experience | `stream_preview_updates()` subscribes to `apg.them.preview.<tenant_id>` and fires a callback per message. Flask-AppBuilder `/them/preview/stream` SSE endpoint backed by the same subject. Eliminates the 30-60 s refresh cycle. |
| I9 | AI-Assisted Colour Palette Generation | AI Augmentation | `generate_palette(seed_hex, palette_size, wcag_level, brand_keywords)` computes OKLCH candidates, filters by contrast, optionally calls a local Ollama model for brand-aligned naming and mood annotation. Returns scored candidates ready for `update_tokens`. |
| I10 | Semantic Token Aliases | Design System | Token values in `{token.path}` form are resolved recursively (cycle-detected). `resolve_aliases()` returns fully-resolved token map. `generate_css_variables` resolves aliases before emitting. Matches the Atlassian / IBM / Salesforce alias pattern. |
| I11 | Multi-Tenant Canary Rollout | Deployment Safety | `canary_rollout()` partitions target tenants into cohorts; after each cohort it samples `theme_audit_accessibility` and halts if the WCAG violation rate exceeds the configured threshold; publishes progress to `apg.them.rollout.<tenant_id>`. |
| I12 | Style Dictionary Build Pipeline | Toolchain | `build_style_dictionary(platforms)` exports tokens in Style Dictionary format and invokes the CLI via `asyncio.create_subprocess_exec`; returns artefact paths for `css`, `scss`, `ios_swift`, `android_xml`, `js_es6`. |
| I13 | Theme Snapshot and Time-Travel Audit | Compliance | `snapshot_theme()` serialises all tokens, assets, overrides, breakpoints, and a11y results into an immutable blob. `restore_theme_snapshot()` re-applies as new versions without touching history. Enables forensic replay for GDPR/SOC 2 audits. |
| I14 | Figma Variables Two-Way Sync | Design-Dev Handoff | `import_figma_variables()` maps Figma variable collections to token groups via the Figma REST API. `export_to_figma()` PATCHes updated tokens back. Uses `httpx.AsyncClient` throughout. |
| I15 | Governance Scorecard and SLA Reporting | Observability | `governance_scorecard(period_days)` aggregates token freshness, a11y compliance, asset licensing, publication approval lag, and rollout success into per-dimension scores (0-100) and an overall A-F grade. Persists history for trend analysis. |

---

## Semantic Token Aliases

```python
service.update_tokens(
    tenant_id="tenant-a",
    theme_id=theme_id,
    group="semantic",
    tokens={
        "color.action.primary": "{color.brand.blue.500}",
        "color.brand.blue.500": "#0052CC",
    },
    updated_by="designer",
)

resolved = await service.resolve_aliases(tenant_id="tenant-a", theme_id=theme_id)
print(resolved["resolved_tokens"]["color.action.primary"])  # "#0052CC"
```

## Theme Inheritance Graph

```python
parent = service.create_theme(...)
child = service.theme_inherit(
    tenant_id="tenant-a",
    parent_theme_id=parent["id"],
    child_name="Operations Dark",
    overrides={"color.background": "#1a1a2e"},
)

graph = await service.resolve_token_graph(
    tenant_id="tenant-a",
    theme_id=child["id"],
)
print(graph["provenance"]["color.primary"])  # parent theme id
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

---

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

---

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

---

## UI Routes

- dashboard: `/them/dashboard`
- themes: `/them/themes`
- tokens: `/them/tokens`
- branding: `/them/branding`
- assets: `/them/assets`
- preview: `/them/preview`
- preview stream: `/them/preview/stream` (SSE — real-time token updates)
- agents: `/them/agents`
- policies: `/them/policies`
- settings: `/them/settings`

---

## Bytewax / NATS Stream

THEM publishes lifecycle metadata via NATS for Bytewax consumption:

- processor: `bytewax`
- stream: `apg.them.lifecycle`
- key: `tenant_id`

NATS subjects:

| Subject | Trigger |
|---------|---------|
| `apg.them.theme_published.<tenant_id>` | Theme published (async) |
| `apg.them.tokens_updated.<tenant_id>` | Tokens updated (async) |
| `apg.them.rollout.<tenant_id>` | Canary rollout progress |
| `apg.them.preview.<tenant_id>` | Real-time preview artifact (WebSocket bridge) |
| `apg.them.lifecycle` | All lifecycle events (Bytewax) |

Events:

- `theme_created`
- `tokens_updated`
- `brand_asset_added`
- `theme_preview_created`
- `theme_published`
- `them_agent_registered`
- `token_rollback`
- `theme_snapshot_created`
- `theme_snapshot_restored`
- `governance_scorecard_computed`
- `tenant_theme_applied`
- `css_variables_generated`
- `dark_mode_variant_created`
- `breakpoints_configured`
- `component_override_registered`
- `accessibility_audit_completed`
- `design_tokens_exported`

---

## Adapter Boundaries

The in-package service stores records in memory so generated applications,
tests, and publish-plan probes can execute without external infrastructure.
Production systems should attach identity providers, audit sinks, asset stores,
preview renderers, accessibility engines, rollout orchestrators, NATS clients,
and Bytewax workers through APG adapters.

---

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/them/__init__.py capabilities/common/them/capability_contract.py capabilities/common/them/models.py capabilities/common/them/theme_runtime.py capabilities/common/them/service.py capabilities/common/them/api.py capabilities/common/them/views.py capabilities/common/them/app.py capabilities/common/them/test_capability_contract.py capabilities/common/them/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/them/test_capability_contract.py capabilities/common/them/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/them --json
./.venv/bin/apg capabilities publish-plan capabilities/common/them --json
```
