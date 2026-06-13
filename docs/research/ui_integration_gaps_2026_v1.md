# APG Research Report: Beautiful Interfaces, Capability Integration & Missing Capabilities

**Date**: 2026-06-13 | **Author**: Nyimbi Odero | © 2025 Datacraft

---

## Executive Summary

APG has 351 world-class capabilities with robust business logic, reliability infrastructure, and a clean architecture. However three critical gaps prevent it from being a complete enterprise platform:

1. **UI**: The generated `/ui` is functionally correct but visually circa 2005 — no CSS framework, no JavaScript, 5 design tokens, pure `html.escape()` string concatenation. This undermines the platform's credibility despite the backend quality.

2. **Integration**: `requires: [...]` in MANIFEST is metadata-only. Most capabilities (fintech_gateway, fintech_fraud, fintech_aml) don't actually call each other — they're isolated services that happen to be registered together. NATS event streams are emitted but **nothing subscribes to them cross-capability**.

3. **Gaps**: ~40 high-value capabilities are still missing. The most impactful: AI model governance, API monetisation, tenant/billing management, workflow designer UI, and 5 underweight domains (crm=1, eam=1).

---

## Part 1: Beautiful State-of-the-Art Interfaces

### Current State (Accurate Assessment)

The generated `app.py` UI produces:
- **Zero CSS frameworks** — pure server-rendered HTML, no Tailwind, no Bootstrap, no nothing
- **Zero JavaScript** — no htmx, no Alpine, no React, no event handlers
- **5 design tokens** — `--apg-accent`, `--apg-surface`, `--apg-border`, `--apg-text`, `--apg-muted`
- **No component classes** — raw `<h1>`, `<table>`, `<form>` elements with minimal styling
- **No layout grid** — single-column `max-width: 1100px` block flow
- **No responsive design** — single breakpoint only

The APG Studio (`/studio`) runs a completely separate, high-quality design system (dark theme, glassmorphism, Inter font, CSS Grid, animated counters). Generated applications get none of this.

### What World-Class Looks Like

| Platform | UI Approach | Key Principle |
|----------|-------------|---------------|
| **Salesforce Lightning** | Design System (SLDS) with 200+ components | Semantic tokens → component variants → layouts |
| **SAP Fiori** | SAPUI5 framework | Role-based, task-oriented, "intuitive by design" |
| **Retool** | Code-generated from schema | Type inference → widget selection → automatic layout |
| **Appsmith** | Drag-drop + JSON binding | Data schema → UI widget mapping |
| **Shadcn/ui** | Copy-paste Radix + Tailwind | Accessible headless + visual layer separation |
| **Linear** | High-density, keyboard-first | Speed as design principle |

### Recommendation: The HTMX+Tailwind CDN Renderer

**The right solution for APG is not a SPA framework. It's progressive enhancement of server-rendered HTML.**

**Proposed architecture:**
```
APG Theme Tokens → Tailwind Config → Component Classes → Server HTML → htmx interactions
```

**Why this is right for APG:**
1. `app.py` already serves pure HTML — adding CDN Tailwind + htmx requires changing only `_html_page()` and the CSS generator
2. Zero build step — just CDN links in the HTML `<head>`
3. htmx enables reactive updates (`hx-get`, `hx-post`, `hx-trigger`) without a SPA
4. Tailwind maps directly to APG's 11 design tokens → CSS variables → utility classes
5. Alpine.js adds micro-interactions (dropdowns, modals, tabs) with 3KB

**Immediate changes (code-generator.py — 2 function changes):**

```python
def _html_page(title: str, body: str, ...) -> str:
    return f"""<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- htmx for reactivity -->
  <script src="https://unpkg.com/htmx.org@2.0.4"></script>
  <!-- Alpine.js for micro-interactions -->
  <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
  <style>
    :root {{ {theme_vars} }}
    /* Tailwind config using APG tokens */
  </style>
</head>
<body class="bg-gray-50 text-gray-900 h-full">
  <nav class="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-4">
    {nav_html}
  </nav>
  <main class="max-w-7xl mx-auto px-6 py-8">{body}</main>
</body>
</html>"""
```

**APG Theme Token → Tailwind Mapping:**
```python
TOKEN_TAILWIND_MAP = {
    "color.primary":   ("--apg-primary",   "primary"),
    "color.accent":    ("--apg-accent",    "accent"),
    "color.success":   ("--apg-success",   "success"),
    "color.warning":   ("--apg-warning",   "warning"),
    "color.danger":    ("--apg-danger",    "danger"),
    "surface.canvas":  ("--apg-canvas",    None),
    "surface.panel":   ("--apg-panel",     None),
    "text.primary":    ("--apg-text",      None),
    "text.secondary":  ("--apg-muted",     None),
    "border.radius":   ("--apg-radius",    None),
    "density":         ("--apg-density",   None),
}
```

### 3-Tier UI Strategy

**Tier 1 — Generated App UI (app.py) — `immediate`**
- Replace string-built HTML with Tailwind CDN + htmx
- Map APG theme tokens to Tailwind `tailwind.config` at runtime
- Use shadcn-inspired component HTML patterns (cards, tables, badges, sidebars)
- Add hx-get on table rows for inline record detail
- Impact: dramatic visual improvement with ~100 lines of changes to code_generator.py

**Tier 2 — Capability Blueprint UI (blueprint.py) — `medium term`**
- Standardise all FAB blueprints on a shared APG component library
- Build `capabilities/common/ui/` with APG-branded Jinja2 components (nav, sidebar, data-table, form, badge, stat-card)
- Every capability blueprint imports from this shared library
- Screen models from views.py become data props to components

**Tier 3 — Application Shell (SPA option) — `long term`**
- Build a React/Next.js application shell that consumes APG JSON APIs
- The generated `semantic_model.json` + `openapi.json` auto-generates TypeScript types
- Each capability's screen model dict maps to auto-generated React components
- Hosted at `https://studio.datacraft.co.ke` — white-labelled per tenant

### Design System Specification for APG

```
Brand: Datacraft / APG
Primary: #1E5B5A (deep teal — professional, African, trustworthy)
Accent: #D97706 (amber — energy, action)
Typeface: Inter (body) + JetBrains Mono (code)
Density: Compact (data-heavy enterprise apps)
Border radius: 8px (modern, not overly round)
Shadow: 0 1px 3px rgba(0,0,0,0.12)
Grid: 12-column, 1280px max-width, 24px gutter
```

### Accessibility Requirements (Non-Negotiable)

- WCAG 2.1 AA minimum for all generated interfaces
- 4.5:1 contrast ratio on all text
- All interactive elements keyboard-accessible
- ARIA labels on all form fields
- Focus rings visible

---

## Part 2: Capability Integration Architecture

### Current State

**The integration gap is severe.** `requires: [...]` in MANIFEST.json is purely documentary — no code enforcement, no actual wiring.

Observed patterns:
- `fintech_gateway` declares `requires: [crm_adv, bia_anl, cbm_cash_management]` — none are imported in service.py
- `fintech_fraud` declares `requires: [fintech_payments, fintech_kyc, fintech_aml]` — none called
- NATS streams are **emitted but never subscribed to** cross-capability
- Only the SACCO sub-cluster has actual cross-capability calls (gua→dep→lnd chain)

### How World-Class Platforms Do It

**Salesforce approach**: Shared data model (sObject) + Platform Events + Process Builder
- Every object (Account, Contact, Opportunity) is shared across all clouds
- Platform Events trigger automated processes cross-module
- Integration is automatic because data model is unified

**SAP S/4HANA approach**: Universal Journal (ACDOCA table) + BAPI/RFC layer
- Single accounting document that every module writes to
- Remote Function Calls between modules with defined contracts
- Master data (Business Partner) shared by all modules

**Microsoft Dynamics approach**: Common Data Model (CDM) + Power Automate
- Canonical entities shared across Finance, HR, Sales
- Power Automate flows wire modules together
- Azure Service Bus for async cross-module events

### Recommended APG Integration Architecture

#### 1. Canonical Entity Registry (highest impact, implement first)

Create `capabilities/common/entity_registry/` — a shared entity store where capabilities register their "canonical" entity representations:

```python
# capabilities/common/entity_registry/service.py
class EntityRegistry:
    """Single source of truth for cross-capability entity resolution."""
    
    async def register_entity(self, capability_id, entity_type, entity_id, canonical_fields):
        """A capability registers a real-world entity with canonical fields."""
        # e.g. fintech_kyc registers Customer{id, name, dob, id_number, risk_level}
        
    async def resolve_entity(self, entity_type, entity_id) -> dict:
        """Any capability can look up the canonical representation of an entity."""
        # Returns merged view from all capabilities that know this entity
        
    async def link_entities(self, entity_type, canonical_id, capability_id, local_id):
        """Link a capability-local ID to the canonical entity."""
```

#### 2. NATS Subscription Wiring (automatic integration)

**The NATS infrastructure exists. Nothing subscribes to it.** Fix this with automatic subscription wiring in `domain/adapters.py`:

```python
# Template for fintech_fraud/domain/adapters.py
def get_fraud_event_handler():
    """Subscribe to fintech_gateway payment events for automatic fraud scoring."""
    async def handle_payment(event: dict) -> None:
        if event.get("event_type") in ("payment_intent.created", "authorize_payment.completed"):
            svc = FraudService(tenant_id=event.get("tenant_id", "default"))
            await svc.score_signal({
                "transaction_id": event.get("resource_id"),
                "amount": event.get("details", {}).get("amount"),
                "merchant": event.get("details", {}).get("merchant_code"),
                "source": "fintech_gateway_event",
            })
    return handle_payment

# Register on startup
from capabilities.common.nats.nats_adapter import NATSConnector
connector = NATSConnector("fintech_fraud")
await connector.subscribe("apg.events.fintech_gateway.payment_intent.created", handle_payment)
```

#### 3. Integration Contract Standard

Every capability should publish an `integration_contract.py` defining what it produces and consumes:

```python
# Template: capabilities/{domain}/{cap}/integration_contract.py
PRODUCES_EVENTS = [
    {"subject": "apg.events.fintech_gateway.payment_intent.created",
     "schema": {"transaction_id": "str", "amount": "Decimal", "merchant_code": "str"},
     "consumers": ["fintech_fraud", "fintech_aml", "bia_anl"]},
]

CONSUMES_EVENTS = [
    {"subject": "apg.events.fintech_fraud.risk_decision.completed",
     "handler": "apply_risk_decision",
     "description": "Apply fraud risk score to pending payment"},
]

PROVIDES_DATA = {
    "payment_intent": {"endpoint": "/api/fintech/gateway/payments/{id}", "schema": PaymentIntent},
}

REQUIRES_DATA = {
    "customer_risk_profile": {"capability": "fintech_kyc", "endpoint": "/api/fintech/kyc/profiles/{customer_id}"},
}
```

#### 4. Cross-Capability Screen Composition

views.py screen models should automatically include related capability data:

```python
# capabilities/fintech/gateway/views.py
def payment_detail_model(service, tenant_id: str, payment_id: str) -> dict[str, Any]:
    model = _base("payment_detail", tenant_id)
    model["payment"] = service.get_record("payment_intents", payment_id, tenant_id)
    
    # Automatic cross-capability enrichment (lazy-loaded)
    try:
        from capabilities.fintech.fraud.service import FraudService
        fraud = FraudService(tenant_id)
        model["fraud_assessment"] = asyncio.run(fraud.get_risk_assessment(payment_id))
    except Exception:
        model["fraud_assessment"] = None
    
    try:
        from capabilities.fintech.kyc.service import KYCService
        kyc = KYCService(tenant_id)
        customer_id = model["payment"].get("customer_id")
        model["customer_kyc"] = asyncio.run(kyc.get_profile(customer_id)) if customer_id else None
    except Exception:
        model["customer_kyc"] = None
    
    return model
```

#### 5. Workflow Auto-Wiring

APG should auto-generate cross-capability workflows when `requires: [X]` includes a workflow-capable capability:

```apg
// APG source — declares integration intent
application PaymentPlatform {
  capabilities: [fintech_gateway, fintech_fraud, fintech_aml, fintech_kyc];
  
  // APG compiler generates:
  // 1. NATS subscriptions wiring gateway → fraud → aml
  // 2. Temporal workflow: PaymentApprovalWorkflow
  // 3. Shared entity registry entries for Customer, Transaction
  // 4. Cross-capability screen composition in views.py
}
```

---

## Part 3: Missing Capabilities

### Critical Gaps (must-have for enterprise credibility)

| # | Capability | Domain | Priority | Effort | Justification |
|---|-----------|--------|---------|--------|---------------|
| 1 | **AI Model Registry & Governance** | `common/mlr` | Critical | 2 weeks | Every ML model needs versioning, bias monitoring, rollback, A/B. MLflow equivalent. |
| 2 | **Tenant & Billing Management** | `common/tenancy` | Critical | 3 weeks | SaaS model requires per-tenant subscription, usage metering, invoice generation. |
| 3 | **API Monetisation** | `common/apim` | Critical | 2 weeks | Rate limiting + per-API pricing + developer keys + usage analytics. |
| 4 | **Workflow Visual Designer** | `common/wfdesigner` | Critical | 3 weeks | BPMN drag-drop workflow builder. ckm_wfa has engine but no visual designer UI. |
| 5 | **Notification Centre** | `common/notifctr` | High | 1 week | Unified in-app notification centre (bell icon, read/unread, preferences). ntfy is backend; this is the UX layer. |
| 6 | **Developer Portal** | `common/devportal` | High | 2 weeks | API docs, sandbox, SDKs, changelog, deprecation notices. Backstage alternative. |
| 7 | **Event Streaming Dashboard** | `common/streamdash` | High | 1 week | Real-time NATS stream viewer, event replay, dead letter inspection. |
| 8 | **Multi-Currency Accounting** | `fin/mca` | High | 2 weeks | FX revaluation engine, multi-currency P&L, hedge accounting. fin/gl lacks this. |
| 9 | **Consolidation Accounting** | `fin/cons` | High | 2 weeks | Multi-entity consolidation, intercompany elimination, group reporting. |
| 10 | **Revenue Recognition** | `fin/rev` | High | 2 weeks | ASC 606 / IFRS 15 deferred revenue, contract liability, POC recognition. |

### High-Value Africa-Specific Gaps

| # | Capability | Domain | Priority | Justification |
|---|-----------|--------|---------|---------------|
| 11 | **USSD Application Builder** | `common/ussdbuilder` | High | Visual builder for USSD flows targeting feature phones (Kenya, Nigeria, Ghana) |
| 12 | **Informal Economy Digitisation** | `common/informal` | High | Jua kali, mama mbogas, hawkers — digital receipts, savings groups, credit profiles |
| 13 | **M-PESA for Business** | `fintech/mpesabiz` | High | Paybill management, till numbers, reconciliation — beyond the connector |
| 14 | **CRB Integration (East Africa)** | `fintech/crb` | High | TransUnion Kenya, Metropol, CreditInfo — credit bureau query and reporting |
| 15 | **County Government Portal** | `government/portal` | Medium | Kenya's 47 counties each need e-service delivery — revenue, permits, welfare |

### Domain Depth Gaps (existing domains that are too shallow)

| Domain | Current | Should Have | Gap |
|--------|---------|-------------|-----|
| `crm` | 1 cap | 8 caps | crm_contact, crm_account, crm_pipeline, crm_activity, crm_analytics, crm_email, crm_cpq |
| `eam` | 1 cap | 6 caps | eam_maintenance, eam_inspection, eam_spares, eam_condition, eam_lifecycle |
| `ecd` | 1 cap | 5 caps | ecd_cad, ecd_bom, ecd_change, ecd_plm, ecd_simulation |
| `pde` | 1 cap | 5 caps | pde_roadmap, pde_feedback, pde_ab_test, pde_analytics, pde_release |
| `int` | 1 cap | 8 caps | int_etl_visual, int_api_mgmt, int_cdc, int_master_data, int_dq, int_lineage |

### Emerging Technology Gaps

| # | Capability | Priority | Description |
|---|-----------|---------|-------------|
| 16 | **AI Reasoning Chains** | High | Chain-of-thought audit logs, decision explanation, counterfactual analysis |
| 17 | **Vector Database** | High | pgvector/Qdrant integration for semantic search beyond RAG |
| 18 | **Streaming Analytics** | High | Real-time aggregations over NATS events (Flink-style over Bytewax) |
| 19 | **Edge Sync Protocol** | Medium | CRDT-based offline-first sync for mobile/rural deployments |
| 20 | **Smart Contract Audit** | Medium | Solidity/Move contract analysis and formal verification hooks |

---

## Part 4: Top 10 Actionable Recommendations

### Ranked by Impact × Feasibility

#### #1. Replace code_generator.py HTML with Tailwind CDN + htmx `[immediate, 3 days]`

**Action**: Modify `_html_page()` and `theme_stylesheet()` in `compiler/code_generator.py` to add Tailwind CDN, htmx 2.0, and Alpine.js CDN links. Replace raw element styling with Tailwind utility classes in all `_ui_*` functions.

**First step**: Add these 3 CDN links to `_html_page()`:
```html
<script src="https://cdn.tailwindcss.com?plugins=forms,typography"></script>
<script defer src="https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js"></script>
```

**Expected outcome**: Every generated app.py immediately gets a professional UI. Zero changes to capabilities.

---

#### #2. Wire NATS subscriptions between fintech capabilities `[immediate, 1 week]`

**Action**: Create `capabilities/common/nats/wiring.py` with automatic subscription setup. When `fintech_gateway` starts, `fintech_fraud` automatically subscribes to its events.

**First step**: 
```python
# capabilities/common/nats/wiring.py
CAPABILITY_SUBSCRIPTIONS = {
    "fintech_fraud": ["apg.events.fintech_gateway.payment_intent.created"],
    "fintech_aml": ["apg.events.fintech_gateway.payment_intent.created", "apg.events.fintech_payments.transfer.completed"],
    "bia_anl": ["apg.events.*.*.completed"],  # all completions
}
```

**Expected outcome**: Payment events automatically trigger fraud scoring without any code changes to fintech_gateway.

---

#### #3. Build `capabilities/common/entity_registry/` `[1 week]`

**Action**: Create a canonical entity registry where any capability can register/resolve real-world entities (Customer, Transaction, Product). This is the foundation for true cross-capability integration.

**First step**: Implement `EntityRegistry.register_entity()`, `resolve_entity()`, and `link_entities()` with PostgreSQL backing and NATS event emission on resolution.

**Expected outcome**: `fintech_kyc` Customer, `crm_adv` Contact, and `healthcare_emr` Patient can be recognised as the same person. Cross-capability workflows become possible.

---

#### #4. Create `integration_contract.py` standard for all capabilities `[2 weeks]`

**Action**: Add `integration_contract.py` to every capability directory. Run a workflow to generate stubs for all 351 capabilities.

**First step**: Define the standard format (see Part 2) and generate stubs using MANIFEST.json data.

**Expected outcome**: The platform becomes self-documenting about integration topology. Auto-generation of NATS subscription wiring becomes possible.

---

#### #5. Build `capabilities/common/tenancy/` billing & tenant management `[3 weeks]`

**Action**: Implement per-tenant subscription management, usage metering, invoice generation, and plan limits. This is what makes APG a viable SaaS product.

**Expected outcome**: APG can be offered as a multi-tenant SaaS with per-seat billing.

---

#### #6. Expand `crm` domain from 1 to 8 capabilities `[2 weeks]`

**Action**: `crm_adv` is fully implemented. Add: `crm_contact`, `crm_account`, `crm_pipeline`, `crm_activity`, `crm_analytics`, `crm_email`, `crm_cpq` to give CRM feature parity with HubSpot.

**Expected outcome**: APG competes with HubSpot/Salesforce CRM for SME market.

---

#### #7. Build APG Design System Component Library `[3 weeks]`

**Action**: Create `capabilities/common/ui_components/` with a shared Jinja2 + Tailwind component library (nav, sidebar, data-table, form, badge, stat-card, chart). All capability blueprints import from this.

**Expected outcome**: Every capability's UI looks consistent and professional without each team re-inventing CSS.

---

#### #8. Add AI Model Registry `capabilities/common/mlr/` `[2 weeks]`

**Action**: Build MLflow-equivalent model registry: versioning, stage transitions (staging → production), deployment tracking, bias monitoring hooks, rollback.

**Expected outcome**: APG's MLX capability has proper governance for AI models in production.

---

#### #9. Implement Workflow Visual Designer `[3 weeks]`

**Action**: `ckm_wfa` has a powerful workflow engine (Temporal-backed) but no visual designer. Build a drag-drop BPMN designer as a Flask Blueprint that generates `workflow { }` APG source.

**Expected outcome**: Non-developers can design workflows without writing APG code.

---

#### #10. USSD Application Builder `capabilities/common/ussdbuilder/` `[2 weeks]`

**Action**: Build a visual USSD menu tree designer that generates `ussd { menu: [...] }` APG source. This is a differentiator no global platform offers and is critical for the African market.

**Expected outcome**: SACCO, healthcare, and government applications can reach feature phone users through an easily designed USSD interface.

---

## Summary Matrix

| Dimension | Current State | Recommended State | Effort |
|-----------|--------------|------------------|--------|
| Generated UI | String-built HTML, 5 tokens, no JS | Tailwind CDN + htmx + APG design system | 3 days → 3 weeks |
| FAB blueprints | Each capability reinvents UI | Shared component library | 3 weeks |
| NATS wiring | Emitted, nothing subscribes | Auto-subscription wiring | 1 week |
| Entity resolution | No canonical entities | EntityRegistry service | 1 week |
| Integration contracts | MANIFEST metadata only | Code-enforced contracts | 2 weeks |
| CRM domain | 1 capability | 8 capabilities (HubSpot parity) | 2 weeks |
| SaaS billing | Not present | Tenancy + billing capability | 3 weeks |
| AI governance | MLX inference only | Model registry + governance | 2 weeks |
| USSD designer | Engine exists, no designer | Visual USSD builder | 2 weeks |
| Missing caps | ~40 critical gaps | 20 addressed | 8 weeks |

---

*This document is the output of deep parallel research across UI/UX architecture, capability integration patterns, enterprise platform analysis, and codebase introspection.*
*APG Platform · © 2025 Datacraft · www.datacraft.co.ke*
