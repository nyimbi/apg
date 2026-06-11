# UI/UX Theming and Branding — User Guide

**Capability ID**: `them` | **Domain**: `common` | **Version**: `1.1.0`

## Overview

THEM is the APG governed visual system. It manages tenant-scoped themes,
versioned design tokens, licensed brand assets, accessibility compliance, and
publication workflows. The service is fully composable via APG adapters and
publishes lifecycle events to NATS for Bytewax consumption.

---

## Installation

```bash
pip install apg-common-them
```

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Theme** | Named visual identity for a tenant; owns tokens, assets, and previews. |
| **Design Token** | Atomic named value (colour, spacing, typography) grouped by category. |
| **Semantic Alias** | Token value of the form `{other.token.name}` resolved at export time. |
| **Brand Asset** | Logo, icon, or illustration with licence and approval evidence. |
| **Preview** | Rendered evidence of a theme applied to a surface at a viewport size. |
| **Publication** | Governed release of a theme to one or more tenants. |
| **Snapshot** | Immutable point-in-time copy of all theme state for compliance replay. |
| **Scorecard** | Multi-dimension governance health report (token freshness, a11y, etc.). |

---

## Creating a Theme

```python
from capabilities.common.them import ThemService

service = ThemService()

theme = service.create_theme(
    tenant_id="acme",
    name="Acme Light",
    owner="design-lead@acme.com",
    brand_name="ACME Corporation",
    guidelines_ref="brand://acme/guidelines/v3",
    brand_colors={
        "primary": "#0052CC",
        "secondary": "#FF5630",
        "neutral.100": "#F4F5F7",
    },
    typography={"font.family": "Inter, sans-serif", "font.size.base": "16px"},
)
```

Required fields: `tenant_id`, `name`, `owner`, `brand_name`, `guidelines_ref`.
Optional seed data (`brand_colors`, `typography`, `spacing`, `border_radius`)
creates initial token records automatically.

---

## Design Tokens

### Updating Tokens

```python
service.update_tokens(
    tenant_id="acme",
    theme_id=theme["id"],
    group="color",
    tokens={
        "color.primary": "#0052CC",
        "color.primary.hover": "#0065FF",
        "color.text.on-primary": "#FFFFFF",
    },
    updated_by="designer@acme.com",
    contrast_validated=True,
    reviewer="a11y-reviewer@acme.com",
)
```

Token keys use dot-notation by convention. Groups: `color`, `typography`,
`spacing`, `density`, `component`, `animation_*`.

### Semantic Aliases

Reference another token by wrapping its key in braces:

```python
service.update_tokens(
    tenant_id="acme",
    theme_id=theme["id"],
    group="semantic",
    tokens={
        "color.action.default": "{color.primary}",
        "color.action.hover":   "{color.primary.hover}",
    },
    updated_by="designer@acme.com",
)

# Resolve all aliases (async)
resolved = await service.resolve_aliases(tenant_id="acme", theme_id=theme["id"])
# resolved["resolved_tokens"]["color.action.default"] == "#0052CC"
```

Aliases resolve recursively up to 20 hops. Cycles are detected and left
unresolved (flagged in `unresolved_aliases`).

### Token Diff and Rollback

```python
# What changed between versions 2 and 5?
diff = await service.token_diff(
    tenant_id="acme",
    theme_id=theme["id"],
    from_version=2,
    to_version=5,
    group="color",  # omit to diff all groups
)

# Roll color group back to version 2
await service.token_rollback(
    tenant_id="acme",
    theme_id=theme["id"],
    group="color",
    target_version=2,
    rolled_back_by="design-lead@acme.com",
)
```

Rollback creates a new version record; no history is mutated.

---

## Theme Inheritance

Child themes inherit all tokens from their parent and override selectively:

```python
child = service.theme_inherit(
    tenant_id="acme",
    parent_theme_id=light_theme_id,
    child_name="Acme Dark",
    overrides={
        "color.background": "#1A1A2E",
        "color.surface":    "#16213E",
        "color.text.default": "#E0E0E0",
    },
    owner="design-lead@acme.com",
)

# Fully resolved token set with provenance
graph = await service.resolve_token_graph(tenant_id="acme", theme_id=child["id"])
# graph["provenance"]["color.primary"] -> light_theme_id  (inherited)
# graph["provenance"]["color.background"] -> child["id"]  (overridden)
```

The graph resolver traverses `fallback_theme_id` pointers up to 10 levels with
cycle detection.

---

## Dark Mode

```python
# Quick inversion (RGB)
dark = service.dark_mode_variant(tenant_id="acme", theme_id=theme["id"])

# Strategy-based
dark = service.dark_mode_generate(
    tenant_id="acme",
    theme_id=theme["id"],
    strategy="surface_swap",  # invert | surface_swap | custom
)
```

For perceptually uniform dark palettes use the OKLCH strategy described in
`WORLD_CLASS_IMPROVEMENTS.md` (I3).

---

## Responsive Breakpoints

```python
service.mobile_breakpoints(
    tenant_id="acme",
    theme_id=theme["id"],
    breakpoints={"xs": 0, "sm": 576, "md": 768, "lg": 992, "xl": 1200},
    updated_by="designer@acme.com",
)

# Generate previews at every breakpoint
previews = service.responsive_preview(
    tenant_id="acme",
    theme_id=theme["id"],
    surface="app_shell",
    created_by="designer@acme.com",
)
```

---

## Component Overrides and White-Labelling

```python
# Per-component token override
service.register_component_override(
    tenant_id="acme",
    theme_id=theme["id"],
    component_type="button.primary",
    tokens={
        "button.primary.bg": "#0052CC",
        "button.primary.text": "#FFFFFF",
        "button.primary.radius": "4px",
    },
    registered_by="designer@acme.com",
)

# White-label config (creates a component override namespace)
service.white_label_config(
    tenant_id="acme",
    theme_id=theme["id"],
    client_name="partner-xyz",
    brand_overrides={
        "color.primary": "#E63946",
        "logo.url": "https://cdn.partner-xyz.com/logo.svg",
    },
    configured_by="platform-ops@acme.com",
)
```

---

## Accessibility

### Single-Token Audit

```python
audit = service.theme_audit_accessibility(
    tenant_id="acme",
    theme_id=theme["id"],
    audited_by="a11y-reviewer@acme.com",
)
print(f"WCAG AA compliance: {audit['compliance_pct']}%")
```

### Multi-Surface Contrast Matrix (async)

Checks every foreground/background token pair:

```python
matrix = await service.contrast_matrix(
    tenant_id="acme",
    theme_id=theme["id"],
    wcag_level="AA",  # or "AAA"
)
for fail in matrix["failures"]:
    print(f"{fail['fg_token']} on {fail['bg_token']}: {fail['ratio']}:1 (need 4.5:1)")
```

### Accessible Palette Suggestions

```python
palette = service.accessible_palette(
    tenant_id="acme",
    theme_id=theme["id"],
    audited_by="a11y-reviewer@acme.com",
)
for s in palette["accessible_suggestions"]:
    print(f"Fix {s['token']}: current={s['current_value']} → suggest={s['suggestion']}")
```

---

## Exporting Tokens

```python
# CSS custom properties
css = service.generate_css_variables(
    tenant_id="acme",
    theme_id=theme["id"],
    selector=":root",
)
print(css["css_block"])

# Dedicated export formats
for fmt in ["json", "css", "style_dictionary", "figma_tokens"]:
    export = service.export_design_tokens(
        tenant_id="acme",
        theme_id=theme["id"],
        format=fmt,
        exported_by="build-bot",
    )
    print(f"{fmt}: {export['token_count']} tokens")

# Figma Tokens JSON
figma = service.token_export_figma(tenant_id="acme", theme_id=theme["id"])
```

---

## Publishing

```python
publication = service.publish_theme(
    tenant_id="acme",
    theme_id=theme["id"],
    published_by="release-manager@acme.com",
    approval_ref="approval://acme/theme/v2",
    target_tenant_count=1,
)

# Async publish with NATS delivery
publication = await service.async_publish_theme(
    tenant_id="acme",
    theme_id=theme["id"],
    published_by="release-manager@acme.com",
    approval_ref="approval://acme/theme/v2",
    nats_client=nats_client,
)
```

`status` will be `"published"` or `"review_required"` depending on rollout
size and whether `rollout_review_recorded=True`.

---

## Canary Rollout

Progressive rollout with automatic halt on WCAG regression:

```python
result = await service.canary_rollout(
    tenant_id="platform",
    theme_id=theme_id,
    target_tenant_ids=all_tenant_ids,
    cohort_size=25,
    halt_on_violation_rate=0.05,  # stop if >5% colour tokens fail WCAG AA
    applied_by="platform-ops",
    nats_client=nats_client,
)
if result["halted"]:
    print(f"Halted at cohort {result['halted_at_cohort']}: {result['halt_reason']}")
```

Progress events are published to `apg.them.rollout.<tenant_id>` on NATS.

---

## Snapshots (Compliance Time-Travel)

```python
# Capture state before a major release
snap = await service.snapshot_theme(
    tenant_id="acme",
    theme_id=theme["id"],
    label="pre-v3-release",
    snapshotted_by="release-manager@acme.com",
)

# Restore if something goes wrong
await service.restore_theme_snapshot(
    tenant_id="acme",
    snapshot_id=snap["id"],
    restored_by="incident-responder@acme.com",
)
```

Restore adds new token versions; the snapshot itself is never mutated. Both
operations emit audit events.

---

## Governance Scorecard

```python
scorecard = await service.governance_scorecard(
    tenant_id="acme",
    period_days=30,
)
# {
#   "grade": "B",
#   "overall_score": 78,
#   "dimensions": {
#     "token_freshness": 90,
#     "accessibility_compliance": 82,
#     "asset_licensing": 100,
#     "publication_governance": 60,
#     "brand_coverage": 60,
#   }
# }
```

Grade scale: A ≥ 90, B ≥ 75, C ≥ 60, D ≥ 45, F < 45.

---

## Analytics

```python
stats = service.theme_analytics(tenant_id="acme", period="last_30d")
```

Returns theme counts, publication counts, accessibility compliance mean, CSS
export count, dark-variant count, and component override count.

---

## THEM Agents

Register AI agents for theme review lanes:

```python
agent = service.register_them_agent(
    tenant_id="acme",
    name="A11y Bot",
    runtime="claude_code",   # codex | claude_code | opencode | pi
    role="accessibility_reviewer",
    scope="automated WCAG contrast check on every token update",
    human_approval_required=True,
)

# Validate before allowing a privileged action
decision = service.validate_agent_theme_action(
    tenant_id="acme",
    agent_id=agent["id"],
    action="publish_theme",
    privileged_scope=True,
    human_approval_ref="approval://agent/a11y/42",
)
assert decision["decision"] == "allow"
```

Privileged actions without a human approval ref are always denied.

---

## NATS / Bytewax Integration

THEM emits NATS events for downstream Bytewax processors:

| Subject | When |
|---------|------|
| `apg.them.theme_published.<tenant_id>` | After async publish |
| `apg.them.tokens_updated.<tenant_id>` | After async token update |
| `apg.them.rollout.<tenant_id>` | Each canary cohort |

```python
import nats

async def run():
    nc = await nats.connect("nats://localhost:4222")
    result = await service.async_publish_theme(
        ...,
        nats_client=nc,
    )
    await nc.drain()
```

For Bytewax, subscribe to `apg.them.lifecycle` — all events are mirrored there
via the APG adapter layer.

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/them/dashboard` | `them:view` | Overview |
| `/them/themes` | `them:design` | Design |
| `/them/tokens` | `them:design` | Design |
| `/them/branding` | `them:manage_brand` | Brand |
| `/them/assets` | `them:manage_brand` | Brand |
| `/them/preview` | `them:view` | Review |
| `/them/agents` | `them:admin` | Automation |
| `/them/policies` | `them:admin` | Governance |
| `/them/settings` | `them:admin` | Settings |

---

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or
environment variables prefixed with `THEM_`.

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `THEM_NATS_URL` | `nats://localhost:4222` | NATS server for lifecycle events |
| `THEM_WCAG_LEVEL` | `AA` | Default WCAG compliance level for audits |
| `THEM_SNAPSHOT_RETENTION_DAYS` | `365` | Snapshot retention for compliance |
| `THEM_CANARY_COHORT_SIZE` | `25` | Default canary rollout cohort size |
| `THEM_SCORECARD_PERIOD_DAYS` | `30` | Governance scorecard default window |

---

## APG Composition

Reference THEM in `.apg` source files:

```apg
use them;
```

THEM provides: `theme_tokens`, `brand_governance`, `asset_libraries`,
`preview_workflows`, `theme_publication_governance`.

THEM requires: `conf`, `auth`, `i18n`, `audl`.

---

## Further Reading

- `service.py` — Business logic and all service methods
- `models.py` — Data models (re-exports from `theme_runtime.py`)
- `api.py` — REST API payload helpers
- `views.py` — Flask-AppBuilder view models and Pydantic schemas
- `capability_contract.py` — Policy rules and governance contract
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 prioritised enhancement proposals
- `README.md` — Quick reference
