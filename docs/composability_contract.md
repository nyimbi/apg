# APG Capability Composability Contract — Definitive Reference

> © 2025 Datacraft. Author: Nyimbi Odero.
> This document is the authoritative specification for the APG Capability Composability Contract.
> All capability implementations must conform to every rule stated here.

---

## Table of Contents

1. [What Is the Composability Contract?](#1-what-is-the-composability-contract)
2. [The Contract Schema — Exhaustive Field Reference](#2-the-contract-schema-exhaustive-field-reference)
3. [Provides / Requires Semantics](#3-provides-requires-semantics)
4. [Rule Engine Reference](#4-rule-engine-reference)
5. [UI Contract](#5-ui-contract)
6. [Theme Tokens Reference](#6-theme-tokens-reference)
7. [Streaming Events Contract](#7-streaming-events-contract)
8. [Configuration Schema](#8-configuration-schema)
9. [Composability Patterns](#9-composability-patterns)
10. [The Composability Graph](#10-the-composability-graph)

---

## 1. What Is the Composability Contract?

Every APG capability exposes a **machine-readable contract** through a single Python module named `capability_contract.py` located at the root of the capability directory. The contract is the source of truth for everything a composition layer, deployment tool, or peer capability needs to know about the capability at runtime — without loading the full capability stack.

The contract is returned by a single function:

```python
def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
    ...
```

The registry (`capabilities/capability_contract_registry.py`) discovers, loads, and validates every contract file in the tree. Contracts are called with a `tenant_id` so each tenant gets an isolated configuration view.

### What the contract declares

| Surface | Purpose |
|---|---|
| **Identity** | `capability`, `display_name`, `version` — stable identifiers used by the registry |
| **Services provided** | `provides` — what this capability makes available to others |
| **Services required** | `requires` — what must be deployed before this capability starts |
| **Tenant configuration** | `configuration` + `configuration_schema` — runtime config with defaults and schema |
| **Governance rules** | `rule_engine` — deterministic rules evaluated before every guarded operation |
| **UI surfaces** | `ui` — shell type, routes, template roots |
| **Theme tokens** | `theme` — design tokens and per-entity component descriptors |
| **Streaming events** | `streaming` — Bytewax stream definition, event names, guardrails |

The contract does **not** contain runtime logic, database schemas, or service implementation details. It contains only the information needed by tooling and peer capabilities.

---

## 2. The Contract Schema — Exhaustive Field Reference

### 2.1 Top-Level Keys

The registry enforces these required top-level keys:

```python
REQUIRED_CONTRACT_KEYS = {"configuration", "configuration_schema", "rule_engine", "ui", "theme"}
```

Additionally, `capability` must be a non-empty string. The full top-level shape:

```python
{
    "capability":            str,          # required — unique capability ID
    "display_name":          str,          # optional — human-readable name
    "version":               str,          # optional — semver string
    "provides":              list[str],    # optional — services this cap exposes
    "requires":              list[str],    # optional — capability IDs this cap needs
    "configuration":         dict,         # required
    "configuration_schema":  dict,         # required
    "rule_engine":           dict,         # required
    "ui":                    dict,         # required
    "theme":                 dict,         # required
    "streaming":             dict,         # optional but conventional
}
```

---

### 2.2 `capability`

| Property | Value |
|---|---|
| Type | `str` |
| Required | Yes |
| Validation | Non-empty string |
| Registry use | Primary key in the registry; used as `dict[str, CapabilityContractRecord]` key |

Naming convention: `<domain>_<short_name>` for domain capabilities, bare short code for platform capabilities.

```python
# Platform capability
"capability": "auth"

# Domain capability
"capability": "intel_alerts"
"capability": "arc_accounts_receivable"
```

---

### 2.3 `display_name`

| Property | Value |
|---|---|
| Type | `str` |
| Required | No |
| Fallback | If absent or empty, `CapabilityContractRecord.display_name` falls back to the `capability` ID |

```python
"display_name": "Alert Management"
"display_name": "Authentication & RBAC"
```

---

### 2.4 `version`

| Property | Value |
|---|---|
| Type | `str` |
| Required | No (conventional) |
| Format | Semver: `MAJOR.MINOR.PATCH` |

```python
"version": "1.1.0"
"version": "2.1.0"
```

The registry does not enforce version format, but tooling and marketplace consumers depend on semver.

---

### 2.5 `configuration`

| Property | Value |
|---|---|
| Type | `dict` |
| Required | Yes |
| Validation | Must be a `dict`; `configuration["tenant_id"]` must be a non-empty string |

The `configuration` dict holds **runtime defaults**. When `get_capability_contract(tenant_id)` is called, the implementation deep-copies the default configuration and sets `configuration["tenant_id"] = tenant_id`. This guarantees tenant isolation at the contract level.

```python
DEFAULT_CONFIGURATION: dict[str, Any] = {
    "tenant_id": "default",
    "alerts": {
        "supported_alert_types": SUPPORTED_ALERT_TYPES,
        "supported_severities": SUPPORTED_SEVERITIES,
        "signal_required": True,
        "evidence_required": True,
    },
    "governance": {
        "require_tenant_context": True,
        "cross_tenant_alert_denied": True,
    },
    "observability": {
        "event_stream": "apg.intel.alerts.lifecycle",
        "stream_processor": "bytewax",
    },
    "adapters": {
        "auth": "auth",
        "audit": "audl",
        "event_stream": "bytewax",
    },
    "ui": {"enable_dashboard": True},
    "theme": {"default_theme": "intel_alerts_control", "allow_tenant_overrides": True},
}
```

The top-level keys of `configuration` must match the keys listed in `configuration_schema.required`.

---

### 2.6 `configuration_schema`

| Property | Value |
|---|---|
| Type | `dict` |
| Required | Yes |
| Validation | `required` list must contain at minimum `{"tenant_id", "ui", "theme"}` |

JSON Schema–style descriptor. The registry checks `configuration_schema.required` and rejects any contract where `tenant_id`, `ui`, or `theme` are absent from it.

```python
"configuration_schema": {
    "type": "object",
    "required": ["tenant_id", "ui", "theme", "alerts", "governance", "observability", "adapters"],
    "properties": {
        "tenant_id": {"type": "string", "minLength": 1},
        "ui":        {"type": "object"},
        "theme":     {"type": "object"},
        "alerts":    {"type": "object"},
    },
}
```

The registry constant enforcing the minimum:

```python
REQUIRED_SCHEMA_KEYS = {"tenant_id", "ui", "theme"}
```

---

### 2.7 `rule_engine`

Full reference in [Section 4](#4-rule-engine-reference). Shape summary:

```python
"rule_engine": {
    "type":             "deterministic",  # required, only valid value
    "default_decision": "allow",          # conventional
    "rules":            list[dict],       # required, non-empty
    "inputs":           list[str],        # optional — documented context fields
    "outputs":          list[str],        # optional — documented effect keys
}
```

---

### 2.8 `rule_engine.rules[]` — per-rule fields

Each rule is a `dict` with three required keys:

```python
REQUIRED_RULE_KEYS = {"name", "condition", "effect"}
```

| Field | Type | Required | Validation |
|---|---|---|---|
| `name` | `str` | Yes | Non-empty string; used in matched_rules output |
| `condition` | `dict` | Yes | Dict of field-to-value predicates; see [Section 4.3](#43-condition-operators) |
| `effect` | `dict` | Yes | Must contain `decision`; see [Section 4.4](#44-effect-fields) |
| `description` | `str` | No | Human-readable explanation |

```python
{
    "name": "tenant_context_required",
    "description": "All operations require tenant context.",
    "condition": {"tenant_context_present": False},
    "effect": {
        "decision": "deny",
        "reason": "tenant_context_required",
        "required_action": "attach_tenant_context",
    },
}
```

---

### 2.9 `ui`

Full reference in [Section 5](#5-ui-contract). Shape summary:

```python
"ui": {
    "shell":          "apg_python",       # required, non-empty string
    "requires_theme": True,               # required, must be exactly True
    "template_roots": ["templates/"],     # required, non-empty list
    "routes":         list[dict],         # required, non-empty list
    "api_prefix":     "/cap/api/v1",      # optional
    "view_module":    "views.py",         # optional
}
```

---

### 2.10 `ui.routes[]` — per-route fields

```python
REQUIRED_ROUTE_KEYS = {"name", "path", "component", "permission"}
```

| Field | Type | Required | Validation |
|---|---|---|---|
| `name` | `str` | Yes | Non-empty string |
| `path` | `str` | Yes | Non-empty string; must start with `/` |
| `component` | `str` | Yes | Non-empty string; component class or template name |
| `permission` | `str` | Yes | Non-empty string; permission key checked at route guard |
| `nav_group` | `str` | No | Navigation grouping label |

```python
{
    "name":       "alerts",
    "path":       "/intel-alerts/alerts",
    "component":  "AlertQueue",
    "permission": "intel_alerts:alerts",
    "nav_group":  "Operations",
}
```

---

### 2.11 `theme`

Full reference in [Section 6](#6-theme-tokens-reference). Shape summary:

```python
"theme": {
    "name":       str,          # required, non-empty
    "tokens":     dict[str, str],  # required, non-empty; must contain "border.radius"
    "components": dict[str, dict], # required, non-empty
}
```

---

### 2.12 `streaming`

Full reference in [Section 7](#7-streaming-events-contract). Shape summary:

```python
"streaming": {
    "processor":  "bytewax",           # required value
    "stream":     "apg.<d>.<c>.lifecycle",
    "key":        "tenant_id",
    "events":     list[str],
    "guardrails": list[str],
    "states":     list[str],           # optional
}
```

---

### 2.13 `provides`

| Property | Value |
|---|---|
| Type | `list[str]` |
| Required | No (strongly conventional) |

Service advertisement list. Full semantics in [Section 3](#3-provides-requires-semantics).

---

### 2.14 `requires`

| Property | Value |
|---|---|
| Type | `list[str]` |
| Required | No (strongly conventional) |

Hard dependency list. Full semantics in [Section 3](#3-provides-requires-semantics).

---

## 3. Provides / Requires Semantics

### 3.1 `provides` — service advertisement

`provides` is a **declaration of named services** this capability makes available for other capabilities and applications to bind to. Service names are free-form strings that describe a workflow, lifecycle, or data contract.

```python
PROVIDES = [
    "alert_authority_workflow",
    "alert_rule_workflow",
    "alert_signal_workflow",
    "alert_record_workflow",
    "alert_escalation_workflow",
    "alert_resolution_workflow",
]
```

Key properties:

- Service names are advertising labels, not Python identifiers.
- A service name is consumed by other capabilities' `requires` lists **or** at application-layer integration time.
- Most provided services are **orphaned** in the dependency graph (not referenced in any `requires` list) — this is normal. The composability audit counts 1,997 orphaned service names across 259 capabilities. Orphaned provides are service advertisements, not bugs.
- The full registry has 2,050 total provide entries and 1,900 require edges.

### 3.2 `requires` — hard runtime dependencies

`requires` is a **list of capability IDs** that must be deployed and available before this capability can start. Each entry must resolve to a known `capability` ID in the registry.

```python
# intel_alerts requires these capability IDs:
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn", "geos"]

# arc_accounts_receivable requires:
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "mqeb", "glr_general_ledger", "cbm_cash_management"]
```

Key properties:

- `requires` entries are **capability IDs**, not service names.
- A missing required capability causes a deployment-order failure, not a runtime error.
- The registry does not enforce requires at load time; enforcement is the responsibility of the deployment orchestrator.
- Hard `requires` is reserved for capabilities without which the declaring capability cannot function at all (auth for identity, audl for compliance, etc.).

### 3.3 Null adapter pattern for optional dependencies

When a capability lists a dependency that may not be present in a given deployment (e.g., `geos` for geospatial services), the capability's adapter layer should implement a **null adapter**: a no-op implementation that satisfies the interface contract without requiring the external service. This allows the capability to degrade gracefully.

```python
# In configuration
"adapters": {
    "geospatial": "geos",    # uses real adapter when geos is available
    "nlp": "nlpc",
}
```

If `geos` is absent, the adapter layer returns empty results rather than raising. The contract still declares `requires: ["geos"]` to document the dependency; deployment tooling decides whether to satisfy it.

### 3.4 Dependency graph and deployment order

The composability graph (`capabilities/COMPOSABILITY.md`) provides a full topological sort. The key rules:

1. Capabilities are grouped into **tiers**. All capabilities in tier N can be deployed in parallel once tier N-1 is fully live.
2. Within a domain, the deploy order is a left-to-right linearization of the dependency chain.
3. No capability may require another in the same tier (that would be a circular dependency).

```
Tier 1: audl, conf, mten, ntfy, comp, chat, colb, dlpd, help, vidc, ztna
Tier 2: keym
Tier 3: auth
Tier 4: cach, cons, mchn, mdm, plfd, secu
Tier 5: encr, meta, mqeb, plgn
...
```

**Critical**: `auth` is Tier 3 because it depends on `keym` (Tier 2) which depends on `audl`/`mten` (Tier 1). Do not attempt to deploy `auth` before `keym` is live.

### 3.5 Foundation tier

The Foundation Tier consists of capabilities required by 10 or more other capabilities. These must be deployed first in any APG installation.

| ID | Dependents | Role |
|---|---|---|
| `audl` | 220 | Immutable audit log |
| `auth` | 212 | Identity, sessions, RBAC |
| `ntfy` | 171 | Notification dispatch |
| `conf` | 154 | Central configuration |
| `mten` | 117 | Multi-tenancy and tenant isolation |
| `mqeb` | 111 | Message queue / event bus |
| `wflo` | 99 | Workflow engine |
| `moni` | 91 | Monitoring and observability |
| `nlpc` | 82 | NLP core |
| `schd` | 64 | Scheduler |

### 3.6 Registry resolution

The registry loads contracts in filesystem-discovery order (alphabetical). After loading, capability IDs are the keys:

```python
registry: dict[str, CapabilityContractRecord] = load_contract_registry(root, tenant_id)
record = registry["intel_alerts"]
```

`CapabilityContractRecord` fields:

```python
@dataclass(frozen=True)
class CapabilityContractRecord:
    capability_id: str
    display_name:  str
    path:          Path      # filesystem path to capability_contract.py
    module_name:   str       # dynamically assigned module name
    contract:      dict[str, Any]
    module:        ModuleType
```

Capabilities installed as standalone PyPI packages are also discovered via the `apg.capabilities` entry-point group — no source tree presence required.

---

## 4. Rule Engine Reference

### 4.1 Engine type

The only valid `rule_engine.type` is `"deterministic"`. The registry rejects any other value:

```python
if rule_engine.get("type") != "deterministic":
    raise ValueError(f"{source} rule_engine.type must be deterministic")
```

The deterministic engine evaluates **all rules in sequence**. There is no short-circuit on first match (rules continue evaluating) except that `deny` is terminal for the decision accumulator: once a rule produces `deny`, the final decision is `deny` regardless of subsequent rules.

### 4.2 `default_decision`

The conventional value is `"allow"`. When no rule matches the context, the result is `allow`. The registry's default evaluator implements this:

```python
def _evaluate_default(rules, context):
    decision = "allow"
    for rule in rules:
        if _matches(rule["condition"], context):
            ...
            if effect["decision"] == "deny":
                decision = "deny"
            elif effect["decision"] == "require_review" and decision != "deny":
                decision = "require_review"
    return {"decision": decision, ...}
```

The `auth` contract omits `default_decision` from its rule engine dict — the registry still defaults to `allow` via the `_evaluate_default` path.

### 4.3 Condition operators

A condition is a `dict[str, Any]` where **all key-value pairs must match** the context for the condition to trigger (logical AND). Key suffixes control comparison operators:

| Suffix | Operator | Example | Meaning |
|---|---|---|---|
| *(none)* | `==` | `{"tenant_context_present": False}` | Exact equality |
| `_lt` | `<` | `{"credit_score_lt": 0.6}` | Field value strictly less than |
| `_lte` | `<=` | `{"invoice_total_lte": 0}` | Field value less than or equal |
| `_gt` | `>` | `{"unapplied_amount_gt": 0}` | Field value strictly greater than |
| `_gte` | `>=` | `{"invoice_line_count_gte": 1}` | Field value greater than or equal |
| `_ne` | `!=` | `{"event_stream_ne": "bytewax"}` | Field value not equal |

The registry's base `_matches` function supports `_lt`, `_gt`, and `_ne`. Individual capability contracts (e.g., `arc_accounts_receivable`, `auth`) implement extended `_matches_condition` functions that additionally support `_lte` and `_gte`. When writing rules, use only the operators your local `_matches` function supports.

Missing keys in context resolve to `None` for numeric comparisons. A missing key does **not** automatically match `False` — the rule `{"tenant_context_present": False}` only fires when `context["tenant_context_present"]` is explicitly `False`, not when the key is absent.

```python
# Fires when operation is "assess_credit" AND credit_score < 0.6
# AND credit_review_recorded is not True
{
    "condition": {
        "operation": "assess_credit",
        "credit_score_lt": 0.6,
        "credit_review_recorded": False,
    },
    "effect": {"decision": "require_review", ...},
}
```

### 4.4 Effect fields

| Field | Required | Valid values |
|---|---|---|
| `decision` | Yes | `"allow"`, `"deny"`, `"require_review"`, `"warn"`, `"audit"` |
| `reason` | No | String code identifying why this decision was reached |
| `required_action` | No | String code for the action the caller must take to proceed |

```python
"effect": {
    "decision":        "deny",
    "reason":          "authority_expiry_required",
    "required_action": "set_expiry",
}
```

Decision semantics:

| Decision | Meaning | Behavior |
|---|---|---|
| `allow` | Operation is permitted | No action required |
| `deny` | Operation is blocked | Hard block; accumulates to final `deny` regardless of other rules |
| `require_review` | Operation needs human review | Raises final decision to `require_review` unless a `deny` also matched; audit flag set |
| `warn` | Advisory; operation proceeds | Does not change final decision; recorded in actions |
| `audit` | Log the operation | Does not change final decision; triggers audit sink |

`require_review` vs `deny`: `deny` is a hard block — the operation must not proceed. `require_review` allows the operation to proceed **only after** a human reviewer records approval. The context should contain `human_approval_recorded: True` to satisfy `require_review` rules on re-evaluation.

### 4.5 Calling `evaluate_capability_rules`

Every contract module must expose either `evaluate_capability_rules(context)` or rely on the registry's default evaluator. The registry calls the module-level function when present:

```python
if hasattr(record.module, "evaluate_capability_rules"):
    result = record.module.evaluate_capability_rules(context)
else:
    result = _evaluate_default(record.contract["rule_engine"]["rules"], context)
```

Via the registry:

```python
from capabilities.capability_contract_registry import evaluate_rules

result = evaluate_rules(
    capability_id="intel_alerts",
    context={
        "tenant_id": "acme",
        "tenant_context_present": True,
        "operation": "record_alert",
        "operation_type": "write",
        "policy_attached": True,
        "signal_present": True,
        "alert_type_supported": True,
        "severity_supported": True,
        "alert_reference_present": True,
        "evidence_present": True,
    },
)
# result["decision"] == "allow"
# result["matched_rules"] == []
# result["actions"] == []
```

Registry normalizes the result to a consistent shape regardless of which evaluator ran:

```python
{
    "decision":      str,         # "allow" | "deny" | "require_review" | ...
    "matched_rules": list[str],   # names of rules that matched
    "actions":       list[dict],  # effect dicts for matched rules
    "context":       dict,        # the original context
}
```

### 4.6 Common context fields

These fields appear consistently across capability contracts:

| Field | Type | Used by |
|---|---|---|
| `tenant_id` | `str` | All — tenant isolation |
| `tenant_context_present` | `bool` | All — guard against missing tenant |
| `operation` | `str` | All — the operation being attempted |
| `operation_type` | `str` | All — `"read"` or `"write"` |
| `policy_attached` | `bool` | All write operations |
| `evidence_present` | `bool` | Governance operations |
| `human_approval_recorded` | `bool` | Privileged operations |
| `privileged_scope` | `bool` | Agent operations |
| `event_stream_ne` | `str` | Bytewax guardrails |

---

## 5. UI Contract

### 5.1 Shell

```python
PYTHON_UI_SHELL = "apg_python"
```

The only production UI shell is `"apg_python"` — Python stdlib HTTP server (not Flask-AppBuilder, not FastAPI, not Django). The registry normalizes legacy shell names at load time:

```python
LEGACY_UI_SHELL_ALIASES = {
    "flask_appbuilder",
    "fastapi_flask_appbuilder",
    "flask",
    "fastapi",
    "django",
}

# normalize_contract moves legacy value to ui["legacy_shell"]
# and replaces ui["shell"] with "apg_python"
```

If an existing contract uses `"flask_appbuilder"`, the normalized runtime contract has `ui["shell"] == "apg_python"` and `ui["legacy_shell"] == "flask_appbuilder"`. New contracts must use `"apg_python"` directly.

### 5.2 `requires_theme`

Must be exactly `True`. The registry raises if it is `False`, missing, or any other value:

```python
if ui.get("requires_theme") is not True:
    raise ValueError(f"{source} ui.requires_theme must be true")
```

### 5.3 `template_roots`

A non-empty list of relative paths (relative to the capability directory) where the shell looks for templates and static assets.

```python
"template_roots": ["templates/", "static/"]
```

The shell resolves templates in order — first match wins.

### 5.4 Route structure

Each route in `ui["routes"]` must have `name`, `path`, `component`, and `permission`. The `nav_group` field is strongly conventional.

**Path rules**:
- Must start with `/`
- Convention: `/<capability-slug>/<screen-name>` using kebab-case
- The capability slug must be consistent across all routes of the same capability

```python
# All intel_alerts routes use /intel-alerts/ prefix
"/intel-alerts/dashboard"
"/intel-alerts/alerts"
"/intel-alerts/settings"

# All arc_accounts_receivable routes use /arc-accounts-receivable/ prefix
"/arc-accounts-receivable/dashboard"
"/arc-accounts-receivable/invoices"
```

**Permission format**: `<capability_id>:<action>` where action is a lowercase verb or noun.

```python
"permission": "intel_alerts:view"
"permission": "intel_alerts:admin"
"permission": "arc_accounts_receivable:invoice"
```

**Nav groups** organize routes into navigation sections. Common groups across APG:

| Group | Contents |
|---|---|
| `Overview` | Dashboard, summary views |
| `Operations` | Day-to-day operational screens |
| `Governance` | Audit, authorities, reviews |
| `Configuration` | Settings, rules, admin |
| `Automation` | Agent workbenches |
| `Administration` | Settings, system admin |

### 5.5 Screen relationships

Routes compose into the application navigation automatically. The shell reads all capability contracts in the registry and builds a unified navigation tree, grouping routes by `nav_group`. A capability with `nav_group: "Operations"` routes will have those routes appear under the Operations section of the global nav alongside Operations routes from other capabilities.

Route permissions are enforced at the routing layer — a user without `intel_alerts:alerts` permission will not see the `/intel-alerts/alerts` route in their navigation and will receive a 403 if they navigate to it directly.

### 5.6 Optional UI fields

| Field | Purpose |
|---|---|
| `api_prefix` | Base path for the capability's REST API, e.g. `"/intel-alerts/api/v1"` |
| `view_module` | The Python module containing view classes, e.g. `"views.py"` |

---

## 6. Theme Tokens Reference

### 6.1 Required tokens

The registry enforces exactly one required token:

```python
REQUIRED_THEME_TOKENS = {"border.radius"}
```

Every capability theme must declare `border.radius`. Typical value: `"8px"`.

### 6.2 Conventional token names

All APG capabilities use this token vocabulary (not enforced by the registry, but required for visual consistency):

| Token | Type | Example | Purpose |
|---|---|---|---|
| `border.radius` | CSS length | `"8px"` | Card and button corner radius |
| `color.primary` | Hex color | `"#12344D"` | Primary brand color |
| `color.accent` | Hex color | `"#0F8B8D"` | Accent / interactive color |
| `color.success` | Hex color | `"#2D6A4F"` | Success state |
| `color.warning` | Hex color | `"#B7791F"` | Warning state |
| `color.danger` | Hex color | `"#C05621"` | Error / danger state |
| `surface.canvas` | Hex color | `"#F4F7FA"` | Page background |
| `surface.panel` | Hex color | `"#FFFFFF"` | Card / panel background |
| `text.primary` | Hex color | `"#102A43"` | Body text |
| `text.secondary` | Hex color | `"#486581"` | Secondary / muted text |
| `density` | Enum | `"compact"` or `"comfortable"` | Layout density |

### 6.3 Theme naming convention

`theme.name` uses the pattern `<capability_id>_control` for operational capabilities or `<capability_id>_fabric` for platform capabilities.

```python
"name": "intel_alerts_control"         # intel domain, operational
"name": "arc_accounts_receivable_control"  # finance domain, operational
"name": "auth_trust_fabric"            # platform capability
```

### 6.4 Tenant theme overrides

The `configuration.theme.allow_tenant_overrides` flag controls whether tenants may supply their own token values. When `True`, the deployment layer merges tenant token overrides on top of capability defaults at request time. Capability code must never hardcode token values — always read from the active theme context.

### 6.5 Component tokens

`theme.components` is a `dict[str, dict]` keyed by entity name. Each entity dict provides rendering hints:

```python
"components": {
    "alerts": {
        "icon":             "bell-ring",     # Lucide icon name
        "status_indicator": "severity-chip", # Component variant for status
    },
    "agents": {
        "icon":             "bot",
        "status_indicator": "agent-runtime-chip",
    },
}
```

Financial capabilities use richer component descriptors:

```python
"components": {
    "customers": {
        "icon":             "users",
        "status_indicator": "customer-pill",
        "risk_style":       "credit-band",    # domain-specific variant
    },
    "invoices": {
        "visual":           "invoice-grid",
        "status_style":     "invoice-chip",
    },
}
```

The registry validates only that `components` is a non-empty `dict`. Content is interpreted by the shell's renderer.

---

## 7. Streaming Events Contract

### 7.1 Processor

```python
"processor": "bytewax"
```

The only supported stream processor in APG is **Bytewax**. This value is hardcoded in the contract and enforced by guardrail rules. Any capability that emits lifecycle events must declare `processor: "bytewax"`.

### 7.2 Stream naming convention

```
apg.<domain>.<capability_short>.lifecycle
```

Examples:

```python
"apg.intel.alerts.lifecycle"   # intel domain, alerts capability
"apg.fin.arc.lifecycle"        # fin domain, accounts receivable
"apg.auth.lifecycle"           # auth platform capability (no domain prefix)
```

The stream name is the partition namespace. All events from a capability flow into this stream, partitioned by `key`.

### 7.3 Partition key

```python
"key": "tenant_id"
```

All APG streams use `tenant_id` as the Bytewax partition key. This ensures all events for a given tenant land in the same partition, enabling ordered processing of tenant lifecycles and preventing cross-tenant data leakage in stream processing.

### 7.4 Event naming convention

```
<capability_short>_<operation>_<past_tense>
```

Events are past-tense facts — they record what happened, not what is happening:

```python
# intel_alerts events
"alert_authority_recorded"
"alert_workspace_recorded"
"alert_rule_recorded"
"alert_signal_recorded"
"alert_recorded"
"alert_escalation_recorded"
"alert_resolution_recorded"
"alert_agent_registered"

# arc_accounts_receivable events
"customer_created"
"credit_assessed"
"invoice_created"
"invoice_issued"
"payment_recorded"
"cash_applied"
"dispute_opened"
"dispute_resolved"
"arc_agent_registered"

# auth events
"identity_registered"
"role_assigned"
"session_started"
"session_revoked"
"access_evaluated"
"security_agent_registered"
```

### 7.5 Guardrails

Guardrails are rule names declared in `streaming.guardrails` that correspond to rules in the `rule_engine.rules` list. Their purpose is to enforce that streaming operations are always routed through Bytewax.

Standard guardrails present in most capabilities:

```python
"guardrails": [
    "<cap>_batch_requires_bytewax",        # batch mutations must use Bytewax
    "<cap>_event_requires_bytewax",        # lifecycle events must use Bytewax
    "privileged_<cap>_action_requires_human_approval",
]
```

The corresponding rule in `rule_engine.rules`:

```python
{
    "name": "alert_batch_requires_bytewax",
    "condition": {
        "operation": "alert_batch",
        "event_stream_ne": "bytewax",      # fires when event_stream != "bytewax"
    },
    "effect": {
        "decision": "deny",
        "reason": "bytewax_event_stream_required",
        "required_action": "route_alert_batch_to_bytewax",
    },
}
```

The `_ne` suffix on `event_stream_ne` means the condition fires when `context["event_stream"] != "bytewax"`. If the caller does not supply `event_stream` in context, it defaults to `None`, which is `!= "bytewax"`, so the rule fires. The caller must explicitly pass `event_stream: "bytewax"` to pass this guardrail.

### 7.6 Consuming events from another capability

To consume events from capability B in capability A, read the `streaming.stream` field from capability B's contract and configure a Bytewax consumer on that stream name:

```python
from capabilities.capability_contract_registry import get_contract

arc_contract = get_contract("arc_accounts_receivable", tenant_id="acme")
arc_stream = arc_contract["streaming"]["stream"]
# arc_stream == "apg.fin.arc.lifecycle"

# Configure Bytewax consumer:
# input_stream = KafkaInputConfig(topic=arc_stream, partition_key="acme")
```

Capabilities must never access each other's databases directly — all cross-capability data exchange is via the declared event streams.

### 7.7 Optional `states` field

Some contracts declare `streaming.states` — a list of entity lifecycle states that the stream carries:

```python
"states": [
    "draft", "active", "assessed", "approved", "issued",
    "paid", "applied", "overdue", "disputed", "resolved", "blocked"
]
```

This is informational for consumers building state machines on top of the event stream.

---

## 8. Configuration Schema

### 8.1 `configuration` — runtime config dict

The `configuration` dict contains all runtime-configurable values for the capability, organized by concern:

```
configuration
├── tenant_id         string — always the active tenant
├── <domain_config>   dict per capability domain (alerts, invoices, customers, …)
├── governance        dict — cross-cutting governance flags
├── observability     dict — event stream and processor settings
├── adapters          dict — declared external dependencies by role
├── ui                dict — feature flags for screens
└── theme             dict — default theme name and override policy
```

The `adapters` sub-dict is particularly important — it maps semantic roles to capability IDs or well-known service names:

```python
"adapters": {
    "auth":          "auth",    # capability ID
    "audit":         "audl",
    "notifications": "ntfy",
    "event_stream":  "bytewax", # not a capability ID — the processor name
    "nlp":           "nlpc",
    "graph":         "grph",
}
```

### 8.2 `configuration_schema.required` — mandatory keys

The registry enforces:

```python
REQUIRED_SCHEMA_KEYS = {"tenant_id", "ui", "theme"}
```

Every capability's `configuration_schema.required` list must include at minimum `["tenant_id", "ui", "theme"]`. In practice, capabilities declare all top-level configuration keys in `required`:

```python
"required": [
    "tenant_id", "ui", "theme",
    "alerts", "governance", "observability", "adapters",
    "workspaces", "rules", "signals",
]
```

### 8.3 `configuration_schema.properties` — property definitions

Each key in `required` should have a corresponding entry in `properties`:

```python
"properties": {
    "tenant_id": {"type": "string", "minLength": 1},
    "ui":        {"type": "object"},
    "theme":     {"type": "object"},
    "alerts":    {"type": "object"},
}
```

Domain-specific properties may use richer schemas with `enum`, `minimum`, `maximum`, etc. The registry does not validate property values against their schemas — schemas are informational for tooling and documentation generators.

### 8.4 Tenant isolation

Every call to `get_capability_contract(tenant_id)` returns a **deep copy** of the default configuration with `configuration["tenant_id"]` set to the requested tenant. The capability module must not share mutable state between tenants:

```python
def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
    configuration = deepcopy(DEFAULT_CONFIGURATION)   # deep copy, never mutate the module global
    configuration["tenant_id"] = tenant_id
    return {
        "capability": CAPABILITY_ID,
        "configuration": configuration,
        ...
    }
```

Contracts supporting tenant-specific overrides accept an optional `overrides` parameter:

```python
def get_capability_contract(
    tenant_id: str = "default",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configuration = deepcopy(DEFAULT_CONFIGURATION)
    configuration["tenant_id"] = tenant_id
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(configuration.get(key), dict):
                configuration[key].update(value)   # merge dicts
            else:
                configuration[key] = value          # replace scalars
    ...
```

### 8.5 Registry configuration validation

The registry validates `configuration` at load time via `_validate_configuration`:

1. `configuration["tenant_id"]` must be a non-empty string
2. `configuration_schema["required"]` must contain `{"tenant_id", "ui", "theme"}`

These are the only runtime-enforced configuration constraints. All other schema declarations are advisory.

---

## 9. Composability Patterns

### 9.1 Hub-and-Spoke

One orchestrating application capability routes requests to multiple domain capabilities. The hub capability declares `requires` for each spoke. Spokes are independent and do not require each other.

```
Application Hub (e.g., fintech_neobanking)
    ├── requires: fintech_payments
    ├── requires: fintech_wallets
    ├── requires: fintech_kyc
    ├── requires: fintech_aml
    └── requires: fintech_cards
```

The hub's `provides` list aggregates services from all spokes. Routes from all spokes appear in the shared navigation, grouped by `nav_group`. The hub is responsible for composing the unified UI shell.

### 9.2 Pipeline

Capability A emits events to its stream. Capability B consumes that stream and reacts.

```
intel_threats (emits threat_assessed)
    → apg.intel.threats.lifecycle
        → intel_alerts (consumes, creates alerts)
            → apg.intel.alerts.lifecycle
                → intel_reporting (consumes, generates reports)
```

Each capability in the pipeline is independent. B does not `require` A in the hard sense — it subscribes to A's stream. However, B should document the stream dependency in its `configuration.adapters` or `configuration.observability`:

```python
"observability": {
    "upstream_streams": ["apg.intel.threats.lifecycle"],
    "own_stream": "apg.intel.alerts.lifecycle",
}
```

No circular stream dependencies are permitted. All stream flows are directed.

### 9.3 Layered

Platform capabilities underpin domain capabilities. Every domain capability sits on the foundation tier:

```
Domain Capability (e.g., intel_alerts)
    └── requires: auth, audl, ntfy, nlpc, grph, ragn, geos
                   ↑           ↑
            platform tier    shared services tier
```

The layered pattern is the default for all APG capabilities. Domain capabilities should never depend on each other's internals — only on the foundation tier and on well-defined service names via `provides`/`requires`.

### 9.4 Sidecar

An analytics or observability capability runs alongside a primary capability, consuming its event stream without being in its `requires` chain.

```
arc_accounts_receivable (primary)
    emits → apg.fin.arc.lifecycle

bia_anl (sidecar)
    consumes ← apg.fin.arc.lifecycle
    provides → receivables_analytics_dashboard
```

`bia_anl` does not appear in `arc_accounts_receivable.requires`. The dependency is one-way (stream consumption) and is not a hard runtime dependency. `bia_anl` can be deployed or undeployed without affecting `arc_accounts_receivable`.

### 9.5 Federation

Multiple tenant contexts coexist in a single deployment. Each tenant receives an isolated configuration view from the contract:

```python
acme_contract = get_capability_contract("acme")
beta_contract = get_capability_contract("beta")
# acme_contract["configuration"]["tenant_id"] == "acme"
# beta_contract["configuration"]["tenant_id"] == "beta"
```

Cross-tenant operations are blocked by rule engine rules:

```python
{
    "name": "cross_tenant_access_requires_membership",
    "condition": {"tenant_mismatch": True, "tenant_membership_confirmed": False},
    "effect": {"decision": "deny", ...},
}
```

Federation across tenants (federated identity, shared data) requires explicit `tenant_membership_confirmed: True` in context.

### 9.6 Standalone

A single capability deployed without the full APG platform. The capability's `requires` list becomes a dependency manifest that the deployer must satisfy. If foundation capabilities (`auth`, `audl`) are not available, the capability should fail fast at startup rather than silently producing incorrect audit trails.

For standalone deployments, mock adapters or minimal implementations of `auth` and `audl` must be provided. The contract's `configuration.adapters` dict documents what is needed:

```python
"adapters": {
    "auth":          "auth",   # required — must provide identity and session
    "audit":         "audl",   # required — must provide audit log sink
    "event_stream":  "bytewax", # required — must provide stream processor
}
```

---

## 10. The Composability Graph

### 10.1 Graph source

`capabilities/COMPOSABILITY.md` is the authoritative composability map. It is generated by a composability audit tool that:

1. Loads all 259 capability contracts
2. Builds a directed graph of `requires` edges
3. Topologically sorts the graph into deployment tiers
4. Reports broken requires (references with no provider) and circular dependencies

### 10.2 Graph metrics (current state)

| Metric | Value |
|---|---|
| Total capabilities | 259 |
| Total require edges | 1,900 |
| Total provide entries | 2,050 |
| Average requires per capability | 7.34 |
| Average provides per capability | 7.92 |
| Broken requires | 0 |
| Circular dependencies | 0 |
| Orphaned provides | 1,997 |

### 10.3 Reading the dependency graph

A deployment tier listing means: all capabilities in that tier may be deployed in parallel, and all capabilities in all prior tiers must be healthy before deployment begins.

```
Tier 1: audl, conf, mten, ntfy, ...    (no external APG dependencies)
Tier 2: keym                            (depends only on Tier 1)
Tier 3: auth                            (depends on Tier 2 + Tier 1)
Tier 4: cach, mdm, secu, ...           (depends on Tier 3 and below)
```

A domain dependency chain shows left-to-right deploy order within a domain:

```
fin:  glr_general_ledger → cbm_cash_management → apy_accounts_payable
       → arc_accounts_receivable → bfc_budgeting_forecasting → fin_rpt
```

This means `arc_accounts_receivable` cannot deploy until `glr_general_ledger` and `cbm_cash_management` are live.

### 10.4 Detecting circular dependencies

A circular dependency exists when capability A requires B and B (directly or transitively) requires A. The audit tool uses topological sort (Kahn's algorithm) — any node that cannot be assigned a tier index is part of a cycle.

Eight cycles were found and resolved in the current graph. The resolution principle: **remove the edge from the more capable capability back to the more foundational one**. Examples:

| Cycle | Resolution |
|---|---|
| `auth ↔ keym` | Remove `auth` from `keym.requires`. `keym` (crypto primitives) is foundational; `auth` builds on it. |
| `auth ↔ secu` | Remove `secu` from `auth.requires`. `secu` (security posture) builds on `auth`. |
| `composition_registry ↔ composition_access/config/events` | Remove `composition_registry` from the three lower capabilities. The registry depends on them; they do not need the registry to function. |

When adding a `requires` edge, verify it does not introduce a cycle by checking that the required capability's tier is lower than the declaring capability's tier. If it is not, either the dependency is wrong or the declaring capability's tier must be recalculated.

### 10.5 Orphaned provides

Provides not referenced in any `requires` list are **not bugs**. `provides` is a service advertisement. Consumers bind to service names at application integration time — they do not necessarily declare `requires` edges for every service they consume. Hard `requires` is reserved for capabilities that must be deployed before the declaring capability starts.

### 10.6 Using the manifest API

`capabilities/manifest.py` provides programmatic navigation of the capability inventory:

```python
from capabilities.manifest import (
    get_capability,
    find_capabilities,
    get_domain,
    get_by_path,
    id_to_path,
)

# Look up by capability ID
cap = get_capability("intel_alerts")
# cap["display_name"], cap["description"], cap["path"], cap["package"]

# Search by keyword
results = find_capabilities("alerts")  # scored by relevance, up to 20 results

# All capabilities in a domain
intel_caps = get_domain("intel")

# Path ↔ ID conversions
path = id_to_path("intel_alerts")       # "capabilities/intel/alerts"
cap  = get_by_path("capabilities/intel/alerts")
```

The manifest is a pre-built `MANIFEST.json` loaded from `capabilities/MANIFEST.json`. It is updated by the manifest generation tool when capabilities are added or modified. The API is `lru_cache`-backed — the JSON file is read once per process.

---

## Appendix A: Minimal Compliant Contract

The smallest contract that passes `validate_contract_shape`:

```python
from __future__ import annotations
from copy import deepcopy
from typing import Any

def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
    return {
        "capability": "my_cap",
        "display_name": "My Capability",
        "version": "1.0.0",
        "provides": ["my_service"],
        "requires": ["auth", "audl"],
        "configuration": {
            "tenant_id": tenant_id,
            "ui": {"enable_dashboard": True},
            "theme": {"default_theme": "my_cap_control", "allow_tenant_overrides": True},
        },
        "configuration_schema": {
            "type": "object",
            "required": ["tenant_id", "ui", "theme"],
            "properties": {
                "tenant_id": {"type": "string", "minLength": 1},
                "ui":        {"type": "object"},
                "theme":     {"type": "object"},
            },
        },
        "rule_engine": {
            "type": "deterministic",
            "default_decision": "allow",
            "rules": [
                {
                    "name": "tenant_context_required",
                    "condition": {"tenant_context_present": False},
                    "effect": {
                        "decision": "deny",
                        "reason": "tenant_context_required",
                        "required_action": "attach_tenant_context",
                    },
                }
            ],
        },
        "ui": {
            "shell": "apg_python",
            "requires_theme": True,
            "template_roots": ["templates/"],
            "routes": [
                {
                    "name":       "dashboard",
                    "path":       "/my-cap/dashboard",
                    "component":  "MyCapDashboard",
                    "permission": "my_cap:view",
                    "nav_group":  "Overview",
                }
            ],
        },
        "theme": {
            "name": "my_cap_control",
            "tokens": {
                "color.primary":   "#1E3A5F",
                "border.radius":   "8px",
            },
            "components": {
                "dashboard": {"icon": "layout-dashboard", "status_indicator": "status-chip"},
            },
        },
        "streaming": {
            "processor":  "bytewax",
            "stream":     "apg.my_domain.my_cap.lifecycle",
            "key":        "tenant_id",
            "events":     ["my_cap_record_created"],
            "guardrails": ["my_cap_batch_requires_bytewax"],
        },
    }


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
    from capabilities.capability_contract_registry import _evaluate_default
    contract = get_capability_contract(context.get("tenant_id", "default"))
    return _evaluate_default(contract["rule_engine"]["rules"], context)
```

---

## Appendix B: Validation Quick Reference

```python
from capabilities.capability_contract_registry import (
    validate_contract_registry,   # validate every contract in the tree
    validate_contract_shape,      # validate one contract dict
    load_contract_registry,       # load all contracts into CapabilityContractRecord
    get_contract,                 # get one contract by ID
    evaluate_rules,               # evaluate rules for one capability + context
    discover_contract_paths,      # list all capability_contract.py paths
)

# Validate entire tree
report = validate_contract_registry(root="capabilities/", tenant_id="acme")
assert report["valid"], report["errors"]

# Validate one contract
validate_contract_shape(my_contract_dict, source="my_contract.py")

# Evaluate rules
result = evaluate_rules("intel_alerts", context={"tenant_context_present": True, "operation": "record_alert", ...})
if result["decision"] == "deny":
    raise PermissionError(result["actions"][0]["reason"])
```

---

## Appendix C: Registry Validation Error Reference

| Error message | Cause | Fix |
|---|---|---|
| `missing contract keys: configuration, rule_engine, ...` | Top-level required keys absent | Add the missing keys |
| `missing capability id` | `contract["capability"]` is empty or absent | Set a non-empty string capability ID |
| `configuration must be a dict` | `configuration` is not a dict | Wrap configuration values in a dict |
| `configuration.tenant_id must be a non-empty string` | `tenant_id` is missing, empty, or wrong type | Set `configuration["tenant_id"] = tenant_id` in `get_capability_contract` |
| `configuration_schema.required missing: theme, ui` | schema `required` list missing mandatory keys | Add `"tenant_id"`, `"ui"`, `"theme"` to `required` |
| `rule_engine.type must be deterministic` | `type` is wrong or absent | Set `rule_engine["type"] = "deterministic"` |
| `rule_engine.rules must be a non-empty list` | `rules` is missing or empty | Add at least one rule |
| `rule_engine.rules[N].effect.decision is required` | Rule effect has no `decision` | Add `"decision"` to the effect dict |
| `rule_engine.rules[N].condition must be a dict` | Condition is not a dict | Wrap condition in a dict |
| `ui.requires_theme must be true` | `requires_theme` is `False` or absent | Set `"requires_theme": True` |
| `ui.shell must be a non-empty string` | `shell` is missing or empty | Set `"shell": "apg_python"` |
| `ui.template_roots must be a non-empty list` | `template_roots` missing or empty | Add at least one template root path |
| `ui.routes must be a non-empty list` | `routes` missing or empty | Add at least one route |
| `ui.routes[N].path must start with /` | Path does not begin with `/` | Prefix path with `/` |
| `theme.tokens missing: border.radius` | `border.radius` not in tokens | Add `"border.radius": "8px"` to tokens |
| `theme.components must be a non-empty dict` | `components` missing or empty | Add at least one entity component descriptor |
