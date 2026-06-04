# APG Capability Development Guide

**Datacraft — © 2025**

A developer reading this document should be able to build a production-quality APG capability from scratch. Every pattern shown is drawn from the `capabilities/intel/alerts` reference implementation; deviations are noted where they are intentional.

---

## Table of Contents

1. [Capability Anatomy](#1-capability-anatomy)
2. [capability_contract.py Template](#2-capability_contractpy-template)
3. [models.py Patterns](#3-modelspy-patterns)
4. [service.py Architecture](#4-servicepy-architecture)
5. [domain/adapters.py Pattern](#5-domainadapterspy-pattern)
6. [database/store.py Pattern](#6-databasestorepy-pattern)
7. [api.py Pattern](#7-apipy-pattern)
8. [views.py Pattern](#8-viewspy-pattern)
9. [app.py Pattern](#9-apppy-pattern)
10. [Testing Guide](#10-testing-guide)
11. [Standalone Development Workflow](#11-standalone-development-workflow)
12. [Integration Patterns](#12-integration-patterns)
13. [Quality Checklist](#13-quality-checklist)

---

## 1. Capability Anatomy

Every capability is a self-contained Python package that lives under `capabilities/<domain>/<code>/`. The tree below is canonical — every file has a defined role that the registry and tooling rely on.

```
capabilities/<domain>/<code>/
├── capability_contract.py   # THE contract: capability_id, rules, ui, theme, streaming
├── models.py                # In-memory data classes (dataclass + to_dict)
├── service.py               # Business logic — 40+ methods, all rule-enforced
├── api.py                   # Process-local helpers; Blueprint REST API
├── views.py                 # View model functions for each screen
├── app.py                   # Standalone Flask server + self_test + semantic_model
├── alerts_runtime.py        # (Optional) small pure-function helpers (normalize, validate)
├── __init__.py              # __version__, __capability_id__, public re-exports
├── __main__.py              # python -m <module> → calls app.main()
├── pyproject.toml           # Package metadata, entry points
├── README.md                # Capability documentation
├── CHANGELOG.md             # Release history
├── py.typed                 # PEP 561 marker (empty file)
├── alembic.ini              # Database migration config
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── 0001_initial.py
├── domain/
│   ├── __init__.py          # Re-exports adapters
│   ├── adapters.py          # Protocol interfaces + Null* fallbacks + factory fns
│   ├── rules.py             # Pure deterministic business rule functions
│   └── events.py            # Domain event dataclasses
├── database/
│   ├── __init__.py
│   ├── store.py             # InMemoryStore + PostgreSQLStore + get_store()
│   └── schema.sql           # Canonical PostgreSQL schema (JSONB apg_records table)
└── tests/
    ├── __init__.py
    ├── test_contract.py     # Contract shape + rule engine tests
    ├── test_service.py      # Service unit tests (InMemoryStore, no mocks)
    └── test_api_views_app.py
```

### File roles in brief

| File | Role |
|---|---|
| `capability_contract.py` | Single source of truth for what this capability is and what it enforces. The registry validates this file. |
| `models.py` | Plain dataclasses. Each model has `to_dict()`. No ORM coupling. |
| `service.py` | All business logic. Every mutating operation calls `_enforce(context)` before touching state. |
| `api.py` | A thin shim. Holds a module-level `_SERVICE` singleton and wrapper functions that unpack `dict` payloads. Optionally exposes a Flask `Blueprint`. |
| `views.py` | Pure functions `→ dict`. Called by templates or API handlers to build screen data. |
| `app.py` | Standalone HTTP server. Provides `/health`, `/contract`, `/evaluate`, `/semantic-model.json`. Always importable even without Flask. |
| `domain/adapters.py` | Protocol definitions and Null implementations for every `REQUIRES` capability. |
| `database/store.py` | `InMemoryStore` (tests/CLI) and `PostgreSQLStore` (production). Selected via `get_store()`. |

---

## 2. capability_contract.py Template

This is the most important file. The registry loads it, validates it, and refuses to register a capability that fails `validate_contract_shape`. Study the validation rules in `capabilities/capability_contract_registry.py` before writing yours.

**Hard requirements enforced by the registry:**
- Top-level keys: `capability`, `configuration`, `configuration_schema`, `rule_engine`, `ui`, `theme`
- `configuration.tenant_id` must be a non-empty string
- `configuration_schema.required` must contain `["tenant_id", "ui", "theme"]`
- `rule_engine.type` must be `"deterministic"`
- `rule_engine.rules` must be a non-empty list; every rule needs `name`, `condition`, `effect`, `effect.decision`
- `ui.requires_theme` must be `True`
- `ui.shell` must be a non-empty string
- `ui.template_roots` must be a non-empty list
- `ui.routes` must be a non-empty list; every route needs `name`, `path` (starts with `/`), `component`, `permission`
- `theme.name`, `theme.tokens` (must include `"border.radius"`), `theme.components` all required

```python
"""Executable capability contract for APG <CapabilityName>."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Identity
# ──────────────────────────────────────────────────────────────────────────────

CAPABILITY_ID      = "domain_code"          # snake_case, globally unique
CAPABILITY_NAME    = "Human Readable Name"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_EVENT_STREAM = f"apg.{CAPABILITY_ID}.lifecycle"


# ──────────────────────────────────────────────────────────────────────────────
# Supported value sets  (minimum 10 — used in rules and service validation)
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_RECORD_TYPES    = ["type_a", "type_b", "type_c"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_STATUSES        = ["draft", "active", "suspended", "archived"]
SUPPORTED_SEVERITIES      = ["low", "medium", "high", "critical"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_RESOLUTION_TYPES = ["confirmed", "false_positive", "duplicate", "mitigated"]
SUPPORTED_ASSIGNMENT_TYPES = ["owner", "reviewer", "observer"]
SUPPORTED_NOTIFICATION_TYPES = ["in_app", "email", "sms", "webhook"]
SUPPORTED_AGENT_RUNTIMES  = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES     = ["steward", "reviewer", "triage", "escalation_reviewer"]
SUPPORTED_AUTHORITY_TYPES = ["mission_order", "legal_mandate", "consent", "partner_authority"]


# ──────────────────────────────────────────────────────────────────────────────
# Default configuration  (tenant_id is injected at call time)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIGURATION: dict[str, Any] = {
    # Required by registry validator
    "tenant_id": "default",

    # Domain-specific config blocks — one dict per major feature area
    "records": {
        "supported_record_types": SUPPORTED_RECORD_TYPES,
        "supported_classifications": SUPPORTED_CLASSIFICATIONS,
        "evidence_required": True,
        "approver_required": True,
    },
    "lifecycle": {
        "supported_statuses": SUPPORTED_STATUSES,
        "supported_severities": SUPPORTED_SEVERITIES,
        "audit_required": True,
    },
    "reviews": {
        "supported_statuses": SUPPORTED_REVIEW_STATUSES,
        "reviewer_required": True,
        "evidence_required": True,
    },
    "agents": {
        "enabled": True,
        "supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
        "supported_roles": SUPPORTED_AGENT_ROLES,
        "scope_required": True,
        "human_approval_required_for_privileged_actions": True,
    },
    "governance": {
        "require_tenant_context": True,
        "policy_attached_for_writes": True,
        "audit_events": True,
        "cross_tenant_access_denied": True,
    },
    "observability": {
        "event_stream": CAPABILITY_EVENT_STREAM,
        "stream_processor": "bytewax",
    },
    "adapters": {
        "auth": "auth",
        "audit": "audl",
        "notifications": "ntfy",
        "event_stream": "bytewax",
    },
    # Required by registry schema validator
    "ui": {
        "enable_dashboard": True,
        "enable_records": True,
        "enable_agents": True,
    },
    # Required by registry schema validator
    "theme": {
        "default_theme": "domain_code_theme",
        "allow_tenant_overrides": True,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Provides / Requires
# ──────────────────────────────────────────────────────────────────────────────

PROVIDES = [
    "record_workflow",
    "lifecycle_workflow",
    "review_workflow",
    "agent_workflow",
]

# Each entry is a CAPABILITY_ID of a capability this one depends on.
# Common capabilities live under capabilities/common/.
REQUIRES = ["auth", "audl", "ntfy"]


# ──────────────────────────────────────────────────────────────────────────────
# UI routes  (minimum 8; every route needs name, path, component, permission)
# ──────────────────────────────────────────────────────────────────────────────

UI_ROUTES = [
    {"name": "dashboard",    "path": "/domain-code/dashboard",   "component": "Dashboard",       "permission": "domain_code:view",    "nav_group": "Overview"},
    {"name": "records",      "path": "/domain-code/records",     "component": "RecordConsole",   "permission": "domain_code:records", "nav_group": "Operations"},
    {"name": "create",       "path": "/domain-code/records/new", "component": "RecordCreateForm","permission": "domain_code:write",   "nav_group": "Operations"},
    {"name": "detail",       "path": "/domain-code/records/:id", "component": "RecordDetail",    "permission": "domain_code:view",    "nav_group": "Operations"},
    {"name": "reviews",      "path": "/domain-code/reviews",     "component": "ReviewQueue",     "permission": "domain_code:reviews", "nav_group": "Governance"},
    {"name": "assignments",  "path": "/domain-code/assignments", "component": "AssignmentConsole","permission": "domain_code:assign", "nav_group": "Operations"},
    {"name": "agents",       "path": "/domain-code/agents",      "component": "AgentWorkbench",  "permission": "domain_code:admin",   "nav_group": "Automation"},
    {"name": "analytics",    "path": "/domain-code/analytics",   "component": "Analytics",       "permission": "domain_code:view",    "nav_group": "Reporting"},
    {"name": "settings",     "path": "/domain-code/settings",    "component": "Settings",        "permission": "domain_code:admin",   "nav_group": "Administration"},
]


# ──────────────────────────────────────────────────────────────────────────────
# Theme
# ──────────────────────────────────────────────────────────────────────────────

THEME = {
    "name": "domain_code_theme",
    "tokens": {
        # border.radius is the only token the registry validates explicitly,
        # but all standard tokens should be present.
        "color.primary":    "#1D4ED8",
        "color.accent":     "#0F766E",
        "color.success":    "#166534",
        "color.warning":    "#A16207",
        "color.danger":     "#991B1B",
        "surface.canvas":   "#F8FAFC",
        "surface.panel":    "#FFFFFF",
        "text.primary":     "#111827",
        "text.secondary":   "#4B5563",
        "border.radius":    "8px",      # required by registry
        "density":          "comfortable",
    },
    "components": {
        # One entry per major UI widget. Values are display hints for the shell.
        "record_card":   {"icon": "file-text",    "status_indicator": "status-chip"},
        "review_queue":  {"icon": "clipboard",    "status_indicator": "review-chip"},
        "agent_panel":   {"icon": "bot",          "status_indicator": "runtime-chip"},
        "analytics":     {"icon": "bar-chart-2",  "status_indicator": "metric-chip"},
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Streaming
# ──────────────────────────────────────────────────────────────────────────────

STREAMING = {
    "processor": "bytewax",              # must be "bytewax"
    "stream":    CAPABILITY_EVENT_STREAM,
    "key":       "tenant_id",
    "events": [
        "record_created",
        "record_updated",
        "record_status_changed",
        "review_recorded",
        "assignment_recorded",
        "agent_registered",
        "batch_mutation",
    ],
    "guardrails": [
        "batch_mutation_requires_bytewax",
        "privileged_agent_action_requires_human_approval",
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# Rules  (minimum 20 — cover all operations that can be denied)
#
# Rule structure:
#   name      — unique snake_case identifier; becomes the PermissionError message
#   condition — flat dict; all keys must match context for rule to fire
#               suffix _ne means "not equal": {"event_stream_ne": "bytewax"}
#   effect    — must contain "decision" ("deny" | "require_review" | "allow")
#               and "reason" (the PermissionError message key)
# ──────────────────────────────────────────────────────────────────────────────

RULES: list[dict[str, Any]] = [
    # ── Baseline governance ──────────────────────────────────────────────────
    {
        "name": "tenant_context_required",
        "condition": {"tenant_context_present": False},
        "effect": {"decision": "deny", "reason": "tenant_context_required",
                   "required_action": "attach_tenant_context"},
    },
    {
        "name": "write_requires_policy",
        "condition": {"operation_type": "write", "policy_attached": False},
        "effect": {"decision": "deny", "reason": "policy_required",
                   "required_action": "attach_write_policy"},
    },

    # ── Record creation ──────────────────────────────────────────────────────
    {
        "name": "record_type_supported",
        "condition": {"operation": "create_record", "record_type_supported": False},
        "effect": {"decision": "deny", "reason": "record_type_not_supported",
                   "required_action": "select_supported_record_type"},
    },
    {
        "name": "record_classification_supported",
        "condition": {"operation": "create_record", "classification_supported": False},
        "effect": {"decision": "deny", "reason": "classification_not_supported",
                   "required_action": "select_supported_classification"},
    },
    {
        "name": "record_approver_required",
        "condition": {"operation": "create_record", "approver_present": False},
        "effect": {"decision": "deny", "reason": "approver_required",
                   "required_action": "attach_approver"},
    },
    {
        "name": "record_evidence_required",
        "condition": {"operation": "create_record", "evidence_present": False},
        "effect": {"decision": "deny", "reason": "evidence_required",
                   "required_action": "attach_evidence"},
    },

    # ── Lifecycle transitions ────────────────────────────────────────────────
    {
        "name": "status_transition_supported",
        "condition": {"operation": "transition_status", "status_supported": False},
        "effect": {"decision": "deny", "reason": "status_not_supported",
                   "required_action": "select_supported_status"},
    },
    {
        "name": "severity_supported",
        "condition": {"operation": "set_severity", "severity_supported": False},
        "effect": {"decision": "deny", "reason": "severity_not_supported",
                   "required_action": "select_supported_severity"},
    },
    {
        "name": "state_change_requires_audit",
        "condition": {"state_change_requested": True, "audit_event_recorded": False},
        "effect": {"decision": "deny", "reason": "audit_event_required",
                   "required_action": "record_audit_event"},
    },

    # ── Review ───────────────────────────────────────────────────────────────
    {
        "name": "review_status_supported",
        "condition": {"operation": "record_review", "status_supported": False},
        "effect": {"decision": "deny", "reason": "review_status_not_supported",
                   "required_action": "select_supported_review_status"},
    },
    {
        "name": "reviewer_required",
        "condition": {"operation": "record_review", "reviewer_present": False},
        "effect": {"decision": "deny", "reason": "reviewer_required",
                   "required_action": "assign_reviewer"},
    },
    {
        "name": "review_evidence_required",
        "condition": {"operation": "record_review", "evidence_present": False},
        "effect": {"decision": "deny", "reason": "review_evidence_required",
                   "required_action": "attach_review_evidence"},
    },
    {
        "name": "review_requires_independent_reviewer",
        "condition": {"operation": "record_review", "reviewer_same_as_requester": True},
        "effect": {"decision": "deny", "reason": "independent_reviewer_required",
                   "required_action": "route_to_independent_reviewer"},
    },

    # ── Assignment ───────────────────────────────────────────────────────────
    {
        "name": "assignment_type_supported",
        "condition": {"operation": "record_assignment", "assignment_type_supported": False},
        "effect": {"decision": "deny", "reason": "assignment_type_not_supported",
                   "required_action": "select_supported_assignment_type"},
    },
    {
        "name": "assignee_required",
        "condition": {"operation": "record_assignment", "assignee_present": False},
        "effect": {"decision": "deny", "reason": "assignee_required",
                   "required_action": "assign_owner"},
    },

    # ── Batch mutations ──────────────────────────────────────────────────────
    {
        "name": "batch_mutation_requires_bytewax",
        # _ne suffix: matches when event_stream != "bytewax"
        "condition": {"operation": "batch_mutation", "event_stream_ne": "bytewax"},
        "effect": {"decision": "deny", "reason": "bytewax_event_stream_required",
                   "required_action": "route_batch_to_bytewax"},
    },

    # ── Agent guardrails ─────────────────────────────────────────────────────
    {
        "name": "agent_runtime_supported",
        "condition": {"operation": "register_agent", "agent_runtime_supported": False},
        "effect": {"decision": "deny", "reason": "agent_runtime_not_supported",
                   "required_action": "select_supported_runtime"},
    },
    {
        "name": "agent_role_supported",
        "condition": {"operation": "register_agent", "agent_role_supported": False},
        "effect": {"decision": "deny", "reason": "agent_role_not_supported",
                   "required_action": "select_supported_role"},
    },
    {
        "name": "agent_scope_required",
        "condition": {"operation": "register_agent", "agent_scope_present": False},
        "effect": {"decision": "deny", "reason": "agent_scope_required",
                   "required_action": "bound_agent_scope"},
    },
    {
        "name": "privileged_agent_action_requires_human_approval",
        "condition": {"operation": "agent_action", "privileged_scope": True,
                      "human_approval_recorded": False},
        "effect": {"decision": "deny", "reason": "human_approval_required",
                   "required_action": "record_human_approval"},
    },
    {
        "name": "autonomous_closure_denied",
        "condition": {"operation": "agent_action", "autonomous_closure_scope": True},
        "effect": {"decision": "deny", "reason": "autonomous_closure_scope_denied",
                   "required_action": "remove_autonomous_closure_scope"},
    },
    {
        "name": "evidence_fabrication_denied",
        "condition": {"operation": "agent_action", "evidence_fabrication_scope": True},
        "effect": {"decision": "deny", "reason": "evidence_fabrication_scope_denied",
                   "required_action": "remove_evidence_fabrication_scope"},
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Public functions
# ──────────────────────────────────────────────────────────────────────────────

def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
    """Return the complete executable capability contract for tenant_id."""
    configuration = deepcopy(DEFAULT_CONFIGURATION)
    configuration["tenant_id"] = tenant_id
    return {
        "capability":    CAPABILITY_ID,
        "name":          CAPABILITY_NAME,
        "display_name":  CAPABILITY_NAME,
        "version":       CAPABILITY_VERSION,
        "provides":      list(PROVIDES),
        "requires":      list(REQUIRES),
        "configuration": configuration,
        "configuration_schema": {
            "type": "object",
            # registry requires tenant_id, ui, theme in `required`
            "required": list(configuration.keys()),
            "properties": {
                key: {"type": "object"}
                for key in configuration
                if key != "tenant_id"
            } | {"tenant_id": {"type": "string", "minLength": 1}},
        },
        "rule_engine": {
            "type":             "deterministic",
            "default_decision": "allow",
            "rules":            deepcopy(RULES),
        },
        "ui": {
            "shell":          "apg_python",
            "api_prefix":     f"/{CAPABILITY_ID.replace('_', '-')}/api/v1",
            "requires_theme": True,         # registry requires this to be True
            "view_module":    "views.py",
            "template_roots": ["templates/", "static/"],
            "routes":         deepcopy(UI_ROUTES),
        },
        "theme":     deepcopy(THEME),
        "streaming": deepcopy(STREAMING),
    }


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate all rules against context. Returns decision + matched actions."""
    actions: list[dict[str, Any]] = []
    for rule in RULES:
        if _matches(rule["condition"], context):
            actions.append(rule["effect"] | {"rule": rule["name"]})
    if not actions:
        return {"decision": "allow", "actions": [], "context": dict(context)}
    return {"decision": "deny", "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
    """Match a rule condition against an evaluation context.

    Supports the _ne suffix for inequality checks:
        {"event_stream_ne": "bytewax"}  fires when context["event_stream"] != "bytewax"
    """
    for key, expected in condition.items():
        if key.endswith("_ne"):
            if context.get(key[:-3]) == expected:
                return False
            continue
        if context.get(key) != expected:
            return False
    return True
```

### Registry validation checklist

Before running the registry validator, verify manually:

1. `get_capability_contract()` returns without error.
2. `configuration["tenant_id"]` is `"default"` (not empty).
3. `configuration_schema["required"]` contains `"tenant_id"`, `"ui"`, `"theme"`.
4. Every rule has `name` (non-empty string), `condition` (dict), `effect` (dict with `decision`).
5. Every route `path` starts with `/`.
6. `theme["tokens"]["border.radius"]` is present.
7. `ui["requires_theme"]` is `True`.

---

## 3. models.py Patterns

APG models are plain Python dataclasses. Pydantic is available for complex validation but the reference capabilities use `dataclass` + `to_dict()` for simplicity. Use Pydantic where validation logic justifies the extra dependency.

### Dataclass pattern (standard)

```python
"""In-memory models for APG <CapabilityName>."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DomainRecord:
    id: str                   # caller-supplied or uuid7str()
    tenant_id: str
    record_type: str
    classification: str
    reference: str
    severity: str
    owner_id: str
    evidence_reference: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DomainReview:
    id: str
    tenant_id: str
    reference_id: str         # ID of the entity being reviewed
    reviewer_id: str
    status: str               # one of SUPPORTED_REVIEW_STATUSES
    evidence_reference: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DomainAgent:
    id: str
    tenant_id: str
    name: str
    runtime: str              # one of SUPPORTED_AGENT_RUNTIMES
    role: str                 # one of SUPPORTED_AGENT_ROLES
    scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

### Pydantic v2 pattern (use when field validation is complex)

```python
from __future__ import annotations

from typing import Annotated, Any
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from <your_pkg> import uuid7str


def _non_empty(v: str) -> str:
    assert v.strip(), "must be non-empty"
    return v

NonEmpty = Annotated[str, AfterValidator(_non_empty)]


class DomainRecordCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
    )

    tenant_id:            NonEmpty
    record_type:          NonEmpty
    classification:       NonEmpty
    reference:            NonEmpty
    severity:             NonEmpty
    owner_id:             NonEmpty
    evidence_reference:   NonEmpty


class DomainRecordResponse(DomainRecordCreate):
    id: str = Field(default_factory=uuid7str)
    created_at: str
    updated_at: str
```

### ID convention

IDs are caller-supplied strings in the reference implementation (enabling deterministic test scenarios). When generating IDs in production code use `uuid7str()`:

```python
# libs/situ-cloudevents/src/situ_cloudevents/_uuid7.py  (project shim)
from uuid6 import uuid7

def uuid7str() -> str:
    return str(uuid7())
```

---

## 4. service.py Architecture

The service class is the capability runtime. Every mutating operation must:

1. Call `_enforce(context)` — which calls `evaluate_capability_rules(context)` and raises `PermissionError` on deny.
2. Write to the in-memory store.
3. Call `_audit(tenant_id, event_type, reference_id)`.

### Constructor

```python
class MyCapabilityService:
    def __init__(self) -> None:
        # Entity stores — keyed by (tenant_id, id)
        self.records:      dict[tuple[str, str], DomainRecord]   = {}
        self.reviews:      dict[tuple[str, str], DomainReview]   = {}
        self.assignments:  dict[tuple[str, str], DomainAssignment] = {}
        self.agents:       dict[tuple[str, str], DomainAgent]    = {}
        self.audit_events: list[dict[str, Any]] = []

        # Mutable metadata (status, timestamps, timeline) separate from
        # the immutable record — keeps to_dict() clean
        self._record_meta: dict[tuple[str, str], dict[str, Any]] = {}

        # Extended state (dedup caches, correlation groups, etc.)
        self._correlation_groups: dict[tuple[str, str], dict[str, Any]] = {}
        self._agent_actions:      dict[tuple[str, str], dict[str, Any]] = {}
```

### Standard CRUD operation

```python
def create_record(
    self,
    record_id: str, tenant_id: str,
    record_type: str, classification: str,
    reference: str, severity: str,
    owner_id: str, evidence_reference: str,
) -> dict[str, Any]:
    record_type    = normalize_code(record_type)
    classification = normalize_code(classification)
    severity       = normalize_code(severity)

    # Build evaluation context — every boolean reflects a pre-check
    self._enforce({
        "tenant_id":                 tenant_id,
        "tenant_context_present":    bool(tenant_id),
        "operation_type":            "write",
        "policy_attached":           True,
        "operation":                 "create_record",
        "record_type_supported":     record_type in SUPPORTED_RECORD_TYPES,
        "classification_supported":  classification in SUPPORTED_CLASSIFICATIONS,
        "severity_supported":        severity in SUPPORTED_SEVERITIES,
        "approver_present":          present(owner_id),
        "evidence_present":          present(evidence_reference),
    })

    item = DomainRecord(record_id, tenant_id, record_type, classification,
                        reference, severity, owner_id, evidence_reference)
    self.records[self._tenant_key(tenant_id, record_id)] = item
    self._record_meta[self._tenant_key(tenant_id, record_id)] = {
        "status":     "active",
        "created_at": _now(),
        "timeline":   [{"ts": _now(), "event": "created", "actor": "system", "notes": ""}],
    }
    self._audit(tenant_id, "record_created", record_id)
    return item.to_dict()
```

### State machine pattern

```python
def transition_record(
    self, record_id: str, tenant_id: str,
    new_status: str, transitioned_by: str, reason: str,
) -> dict[str, Any]:
    assert present(transitioned_by), "transitioned_by is required"
    new_status = normalize_code(new_status)

    record = self._tenant_record_or_none(record_id, tenant_id)
    if record is None:
        raise KeyError(f"record {record_id!r} not found for tenant {tenant_id!r}")

    meta = self._record_meta[self._tenant_key(tenant_id, record_id)]
    current_status = meta["status"]

    # Deterministic rule check
    self._enforce({
        "tenant_id":              tenant_id,
        "tenant_context_present": bool(tenant_id),
        "operation":              "transition_status",
        "status_supported":       new_status in SUPPORTED_STATUSES,
        "state_change_requested": True,
        "audit_event_recorded":   False,   # forces audit path
    })

    _VALID_TRANSITIONS = {
        "active":    {"suspended", "archived"},
        "suspended": {"active", "archived"},
        "draft":     {"active"},
    }
    allowed = _VALID_TRANSITIONS.get(current_status, set())
    if new_status not in allowed:
        raise ValueError(
            f"invalid transition {current_status!r} → {new_status!r}; "
            f"allowed: {sorted(allowed)}"
        )

    meta["status"] = new_status
    self._append_timeline(record_id, tenant_id, "status_changed",
                          transitioned_by, f"{current_status}→{new_status}: {reason}")
    self._audit(tenant_id, "record_status_changed", record_id)
    return {"record_id": record_id, "tenant_id": tenant_id,
            "previous_status": current_status, "status": new_status,
            "transitioned_by": transitioned_by}
```

### Analytics methods

```python
def severity_distribution(self, tenant_id: str) -> dict[str, Any]:
    """Count of active records by severity level."""
    dist: dict[str, int] = {s: 0 for s in SUPPORTED_SEVERITIES}
    for (tid, rid), record in self.records.items():
        if tid != tenant_id:
            continue
        meta = self._record_meta.get((tid, rid), {})
        if meta.get("status") in ("archived",):
            continue
        dist[record.severity] = dist.get(record.severity, 0) + 1
    return {"tenant_id": tenant_id, "distribution": dist, "as_of": _now()}

def mean_time_to_resolve(self, tenant_id: str) -> float:
    """Average resolution time in minutes. Returns 0.0 when no data."""
    durations: list[float] = []
    for (tid, rid), meta in self._record_meta.items():
        if tid != tenant_id or meta.get("status") != "archived":
            continue
        created_dt  = _parse_iso(meta.get("created_at"))
        resolved_dt = _parse_iso(meta.get("resolved_at"))
        if created_dt and resolved_dt:
            durations.append((resolved_dt - created_dt).total_seconds() / 60.0)
    return sum(durations) / len(durations) if durations else 0.0
```

### Private helpers (required in every service)

```python
def _tenant_key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
    return (tenant_id, item_id)

def _tenant_record_or_none(self, item_id: str, tenant_id: str) -> DomainRecord | None:
    return self.records.get(self._tenant_key(tenant_id, item_id))

def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
    self.audit_events.append({
        "tenant_id":    tenant_id,
        "event_type":   event_type,
        "reference_id": reference_id,
        "processor":    "bytewax",
    })

def _enforce(self, context: dict[str, Any]) -> None:
    result = evaluate_capability_rules(context)
    if result["decision"] == "allow":
        return
    reasons = ", ".join(
        action.get("reason", action.get("rule", "policy_denied"))
        for action in result["actions"]
    )
    raise PermissionError(reasons or "policy_denied")

def _append_timeline(
    self, record_id: str, tenant_id: str,
    event: str, actor: str, notes: str,
) -> None:
    meta = self._record_meta.get(self._tenant_key(tenant_id, record_id))
    if meta is None:
        return
    meta.setdefault("timeline", []).append(
        {"ts": _now(), "event": event, "actor": actor, "notes": notes}
    )

def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
    return sum(1 for item in items.values() if item.tenant_id == tenant_id)

def describe(self, tenant_id: str = "default") -> dict[str, Any]:
    return get_capability_contract(tenant_id)

def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
    return evaluate_capability_rules(context)
```

---

## 5. domain/adapters.py Pattern

The adapter pattern decouples the service from platform capabilities. When running standalone, Null implementations are used automatically — no external services needed.

```python
"""Adapter protocols for <CapabilityName>.

Each entry in REQUIRES maps to one Protocol here.
Null* classes are the standalone fallbacks.
_Installed* classes wrap the real platform package when installed.
Factory functions (get_*_adapter) wire the correct implementation.
"""
from __future__ import annotations

import json
import os
from typing import Any, Protocol, runtime_checkable


# ── Auth ──────────────────────────────────────────────────────────────────────
@runtime_checkable
class AuthAdapter(Protocol):
    async def verify_token(self, token: str) -> dict[str, Any]: ...
    async def check_permission(
        self, user_id: str, permission: str, resource: str | None = None
    ) -> bool: ...
    async def get_current_user(self, token: str) -> dict[str, Any]: ...


class NullAuthAdapter:
    """Standalone fallback — all tokens accepted, all permissions granted."""
    async def verify_token(self, token: str) -> dict[str, Any]:
        return {"user_id": token or "anonymous", "tenant_id": "default", "roles": ["admin"]}

    async def check_permission(
        self, user_id: str, permission: str, resource: str | None = None
    ) -> bool:
        return True

    async def get_current_user(self, token: str) -> dict[str, Any]:
        return {"id": token or "anonymous", "name": "Standalone User", "roles": ["admin"]}


class _InstalledAuthAdapter:
    def __init__(self, svc: Any) -> None:
        self._svc = svc

    async def verify_token(self, token: str) -> dict[str, Any]:
        return await self._svc.verify_token(token)

    async def check_permission(
        self, user_id: str, permission: str, resource: str | None = None
    ) -> bool:
        return await self._svc.check_permission(user_id, permission, resource)

    async def get_current_user(self, token: str) -> dict[str, Any]:
        return await self._svc.get_current_user(token)


# ── Audit ─────────────────────────────────────────────────────────────────────
@runtime_checkable
class AuditAdapter(Protocol):
    async def log_event(
        self, event_type: str, actor_id: str, tenant_id: str,
        resource_id: str, details: dict[str, Any],
    ) -> None: ...


class NullAuditAdapter:
    """Standalone fallback — logs to stdout as JSON."""
    async def log_event(
        self, event_type: str, actor_id: str, tenant_id: str,
        resource_id: str, details: dict[str, Any],
    ) -> None:
        print(json.dumps({
            "event_type": event_type, "actor_id": actor_id,
            "tenant_id": tenant_id, "resource_id": resource_id,
            "details": details,
        }, default=str))


# ── Notify ────────────────────────────────────────────────────────────────────
@runtime_checkable
class NotifyAdapter(Protocol):
    async def send(
        self, recipient: str, channel: str, subject: str, body: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...


class NullNotifyAdapter:
    async def send(
        self, recipient: str, channel: str, subject: str, body: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        print(f"[NOTIFY] {channel}→{recipient}: {subject}")


# ── Workflow ──────────────────────────────────────────────────────────────────
@runtime_checkable
class WorkflowAdapter(Protocol):
    async def start_workflow(
        self, definition_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...
    async def complete_task(
        self, task_id: str, outcome: str, variables: dict[str, Any]
    ) -> None: ...


class NullWorkflowAdapter:
    async def start_workflow(
        self, definition_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return {"instance_id": f"local-{definition_id}", "status": "running"}

    async def complete_task(
        self, task_id: str, outcome: str, variables: dict[str, Any]
    ) -> None:
        pass


# ── Factories ─────────────────────────────────────────────────────────────────
def get_auth_adapter(auth_service: Any | None = None) -> AuthAdapter:
    if auth_service is not None:
        return _InstalledAuthAdapter(auth_service)
    try:
        from apg_common_auth import AuthService  # type: ignore[import]
        return _InstalledAuthAdapter(AuthService.from_env())
    except ImportError:
        return NullAuthAdapter()


def get_audit_adapter(audit_service: Any | None = None) -> AuditAdapter:
    if audit_service is not None:
        return audit_service
    try:
        from apg_common_audl import AuditService  # type: ignore[import]
        return AuditService.from_env()
    except ImportError:
        return NullAuditAdapter()


def get_notify_adapter(notify_service: Any | None = None) -> NotifyAdapter:
    if notify_service is not None:
        return notify_service
    try:
        from apg_common_ntfy import NotifyService  # type: ignore[import]
        return NotifyService.from_env()
    except ImportError:
        return NullNotifyAdapter()


def get_workflow_adapter(workflow_service: Any | None = None) -> WorkflowAdapter:
    if workflow_service is not None:
        return workflow_service
    try:
        from apg_common_wflo import WorkflowService  # type: ignore[import]
        return WorkflowService.from_env()
    except ImportError:
        return NullWorkflowAdapter()
```

**Key rule:** Null adapters must never fail. They exist so the full service can be exercised in tests and standalone mode with zero configuration.

---

## 6. database/store.py Pattern

The store abstracts persistence. The same interface works in-memory (tests) and against PostgreSQL (production).

```python
"""Persistence store for <CapabilityName>.

- APG_DATABASE_URL or DATABASE_URL env var → PostgreSQLStore
- Otherwise → InMemoryStore
"""
from __future__ import annotations

import json
import os
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Store(Protocol):
    async def get(self, collection: str, id: str) -> dict[str, Any] | None: ...
    async def put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]: ...
    async def query(
        self, collection: str, filters: dict[str, Any], limit: int = 100
    ) -> list[dict[str, Any]]: ...
    async def delete(self, collection: str, id: str) -> bool: ...
    async def count(self, collection: str, filters: dict[str, Any]) -> int: ...


class InMemoryStore:
    """Zero-config single-process store. Not thread-safe."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, dict[str, Any]]] = {}

    async def get(self, collection: str, id: str) -> dict[str, Any] | None:
        return self._data.get(collection, {}).get(id)

    async def put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        self._data.setdefault(collection, {})[record["id"]] = dict(record)
        return record

    async def query(
        self, collection: str, filters: dict[str, Any], limit: int = 100
    ) -> list[dict[str, Any]]:
        records = list(self._data.get(collection, {}).values())
        for key, value in filters.items():
            records = [r for r in records if r.get(key) == value]
        return records[:limit]

    async def delete(self, collection: str, id: str) -> bool:
        col = self._data.get(collection, {})
        if id in col:
            del col[id]
            return True
        return False

    async def count(self, collection: str, filters: dict[str, Any]) -> int:
        return len(await self.query(collection, filters, limit=100_000))


class PostgreSQLStore:
    """JSONB-backed async store. Requires sqlalchemy[asyncio] + asyncpg.

    Uses the shared apg_records table (see SCHEMA_SQL below).
    Run schema.sql once per database before first use.
    """

    def __init__(self, db_url: str) -> None:
        try:
            from sqlalchemy.ext.asyncio import (
                create_async_engine, async_sessionmaker, AsyncSession,
            )
        except ImportError as exc:
            raise RuntimeError(
                "pip install 'sqlalchemy[asyncio]' asyncpg"
            ) from exc
        engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)
        self._session = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False,
        )

    async def get(self, collection: str, id: str) -> dict[str, Any] | None:
        async with self._session() as s:
            row = (await s.execute(
                "SELECT data FROM apg_records WHERE collection = :c AND id = :id",
                {"c": collection, "id": id},
            )).fetchone()
            return json.loads(row[0]) if row else None

    async def put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        async with self._session() as s:
            await s.execute(
                "INSERT INTO apg_records (id, collection, tenant_id, data) "
                "VALUES (:id, :c, :t, :data) "
                "ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data, updated_at = now()",
                {
                    "id": record["id"], "c": collection,
                    "t": record.get("tenant_id", "default"),
                    "data": json.dumps(record, default=str),
                },
            )
            await s.commit()
        return record

    async def query(
        self, collection: str, filters: dict[str, Any], limit: int = 100,
    ) -> list[dict[str, Any]]:
        conds = " AND ".join(f"data->>'{k}' = :{k}" for k in filters)
        where = f"WHERE collection = :_c" + (f" AND {conds}" if conds else "")
        async with self._session() as s:
            rows = (await s.execute(
                f"SELECT data FROM apg_records {where} LIMIT :lim",
                {"_c": collection, "lim": limit, **filters},
            )).fetchall()
            return [json.loads(r[0]) for r in rows]

    async def delete(self, collection: str, id: str) -> bool:
        async with self._session() as s:
            result = await s.execute(
                "DELETE FROM apg_records WHERE collection = :c AND id = :id",
                {"c": collection, "id": id},
            )
            await s.commit()
            return result.rowcount > 0

    async def count(self, collection: str, filters: dict[str, Any]) -> int:
        conds = " AND ".join(f"data->>'{k}' = :{k}" for k in filters)
        where = f"WHERE collection = :_c" + (f" AND {conds}" if conds else "")
        async with self._session() as s:
            row = (await s.execute(
                f"SELECT COUNT(*) FROM apg_records {where}",
                {"_c": collection, **filters},
            )).fetchone()
            return int(row[0]) if row else 0


# ── Canonical schema (run once per database) ──────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS apg_records (
    id          TEXT        NOT NULL,
    collection  TEXT        NOT NULL,
    tenant_id   TEXT        NOT NULL DEFAULT 'default',
    data        JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection, id)
);

CREATE INDEX IF NOT EXISTS idx_apg_records_tenant
    ON apg_records (collection, tenant_id);

CREATE INDEX IF NOT EXISTS idx_apg_records_data_gin
    ON apg_records USING gin (data);
"""


def get_store(db_url: str | None = None) -> Store:
    resolved = db_url or os.environ.get("APG_DATABASE_URL")
    if resolved:
        try:
            return PostgreSQLStore(resolved)
        except (ImportError, RuntimeError):
            pass
    return InMemoryStore()
```

---

## 7. api.py Pattern

`api.py` serves two purposes: a process-local function API (used in tests and views) and an optional Flask Blueprint for HTTP exposure.

```python
"""Process-local API helpers for <CapabilityName>."""

from __future__ import annotations

try:
    from .service import MyCapabilityService
except ImportError:  # pragma: no cover
    from service import MyCapabilityService  # type: ignore


# Module-level singleton — created once per process
_SERVICE = MyCapabilityService()


def service() -> MyCapabilityService:
    """Return the module-level service instance."""
    return _SERVICE


# ── Process-local helpers (unpack dict payloads) ─────────────────────────────
def create_record(payload: dict) -> dict:
    return _SERVICE.create_record(
        payload["record_id"],
        payload.get("tenant_id", "default"),
        payload["record_type"],
        payload["classification"],
        payload["reference"],
        payload["severity"],
        payload["owner_id"],
        payload["evidence_reference"],
    )


def record_review(payload: dict) -> dict:
    return _SERVICE.record_review(
        payload["review_id"],
        payload.get("tenant_id", "default"),
        payload["reference_id"],
        payload["reviewer_id"],
        payload["status"],
        payload["evidence_reference"],
    )


def register_agent(payload: dict) -> dict:
    return _SERVICE.register_agent(
        payload["agent_id"],
        payload.get("tenant_id", "default"),
        payload["name"],
        payload["runtime"],
        payload["role"],
        payload.get("scope", ""),
    )


def dashboard(payload: dict) -> dict:
    return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))


# ── Optional Flask Blueprint ──────────────────────────────────────────────────
try:
    import asyncio
    from flask import Blueprint, jsonify, request

    blueprint = Blueprint("domain_code_api", __name__)

    def _run(coro):
        """Run an async coroutine from a sync Flask handler."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @blueprint.get("/records")
    def list_records():
        tenant_id = request.args.get("tenant_id", "default")
        filters   = {"tenant_id": tenant_id}
        results   = _run(_SERVICE._store.query("records", filters))
        return jsonify({"records": results, "tenant_id": tenant_id})

    @blueprint.post("/records")
    def create_record_route():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            result = create_record(payload)
            return jsonify(result), 201
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 403
        except (KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @blueprint.get("/records/<record_id>")
    def get_record(record_id: str):
        tenant_id = request.args.get("tenant_id", "default")
        record    = _SERVICE._tenant_record_or_none(record_id, tenant_id)
        if record is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(record.to_dict())

    @blueprint.post("/reviews")
    def record_review_route():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            result = record_review(payload)
            return jsonify(result), 201
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 403

    @blueprint.post("/evaluate")
    def evaluate_route():
        ctx = request.get_json(force=True, silent=True) or {}
        return jsonify(_SERVICE.evaluate(ctx))

except ImportError:
    blueprint = None  # type: ignore[assignment]
```

**Key rule:** The `_run()` helper is the standard way to call async methods from synchronous Flask handlers without needing an ASGI server. Use `asyncio.new_event_loop()` — do not use `asyncio.get_event_loop()` in Flask; it is deprecated for this use case.

---

## 8. views.py Pattern

View model functions are pure: they take a service instance and a tenant_id, return a dict. No rendering — the dict goes to a template or is returned as JSON.

```python
"""View models for <CapabilityName> screens."""

from __future__ import annotations

from typing import Any

try:
    from .capability_contract import get_capability_contract
    from .service import MyCapabilityService
except ImportError:  # pragma: no cover
    from capability_contract import get_capability_contract  # type: ignore
    from service import MyCapabilityService  # type: ignore


def dashboard_model(
    service: MyCapabilityService, tenant_id: str = "default",
) -> dict[str, Any]:
    contract = get_capability_contract(tenant_id)
    return {
        "title":     "My Capability",
        "tenant_id": tenant_id,
        "summary":   service.dashboard_summary(tenant_id),
        "theme":     contract["theme"],
        "routes":    contract["ui"]["routes"],
    }


def records_console_model(
    service: MyCapabilityService, tenant_id: str = "default",
) -> dict[str, Any]:
    return {
        "tenant_id": tenant_id,
        "records":   _tenant_items(service.records, tenant_id),
        "reviews":   _tenant_items(service.reviews, tenant_id),
        "agents":    _tenant_items(service.agents, tenant_id),
    }


def agent_workbench_model(
    service: MyCapabilityService, tenant_id: str = "default",
) -> dict[str, Any]:
    contract = get_capability_contract(tenant_id)
    return {
        "tenant_id":          tenant_id,
        "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
        "supported_roles":    contract["configuration"]["agents"]["supported_roles"],
        "agents":             [
            item.to_dict()
            for item in service.agents.values()
            if item.tenant_id == tenant_id
        ],
    }


def _tenant_items(
    items: dict[Any, Any], tenant_id: str,
) -> list[dict[str, Any]]:
    return [
        item.to_dict()
        for item in sorted(items.values(), key=lambda v: v.id)
        if item.tenant_id == tenant_id
    ]
```

---

## 9. app.py Pattern

`app.py` is the standalone HTTP server. It must work with or without Flask installed (the `create_app` function raises `ImportError` when Flask is absent, but the module still imports).

```python
"""Standalone <CapabilityName> server."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent

try:
    from .capability_contract import (
        CAPABILITY_ID, CAPABILITY_NAME, CAPABILITY_VERSION,
        get_capability_contract, evaluate_capability_rules,
    )
except ImportError:  # direct script execution
    spec = importlib.util.spec_from_file_location(
        "cap_contract", PACKAGE_DIR / "capability_contract.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    CAPABILITY_ID          = module.CAPABILITY_ID
    CAPABILITY_NAME        = module.CAPABILITY_NAME
    CAPABILITY_VERSION     = module.CAPABILITY_VERSION
    get_capability_contract = module.get_capability_contract
    evaluate_capability_rules = module.evaluate_capability_rules


# ── Introspection endpoints ────────────────────────────────────────────────────

def semantic_model() -> dict[str, Any]:
    contract = get_capability_contract()
    return {
        "format":       "apg.semantic-model.v1",
        "ok":           True,
        "app":          {"name": CAPABILITY_ID, "version": CAPABILITY_VERSION},
        "capabilities": {
            CAPABILITY_ID: {
                "name":          contract["name"],
                "version":       contract["version"],
                "provides":      contract["provides"],
                "requires":      contract["requires"],
                "configuration": contract["configuration"],
                "rules":         contract["rule_engine"]["rules"],
                "ui":            contract["ui"],
                "screens": {
                    route["name"]: {
                        "route":      route["path"],
                        "component":  route["component"],
                        "permission": route["permission"],
                    }
                    for route in contract["ui"]["routes"]
                },
                "theme":     contract["theme"],
                "streaming": contract["streaming"],
                "runtime": {
                    "entrypoint": "app.py",
                    "service":    "service.py",
                    "api":        "api.py",
                    "views":      "views.py",
                },
            }
        },
        "deployment": {"source": "capability_contract.py", "target": "python"},
    }


def component_manifest() -> dict[str, Any]:
    return {
        "format":       "apg.component-manifest.v1",
        "kind":         "apg.generated_application",
        "name":         CAPABILITY_ID,
        "display_name": CAPABILITY_NAME,
        "target":       "python",
        "interfaces": {
            "health":         "/health",
            "self_test":      "/self-test",
            "semantic_model": "/semantic-model.json",
        },
        "capabilities": [CAPABILITY_ID],
    }


def self_test() -> dict[str, Any]:
    """Basic sanity checks — run before claiming a build is healthy."""
    model    = semantic_model()
    manifest = component_manifest()
    cap      = model.get("capabilities", {}).get(CAPABILITY_ID, {})
    errors:  list[str] = []

    if model.get("format") != "apg.semantic-model.v1":
        errors.append("semantic model format mismatch")
    if not cap:
        errors.append("capability missing from semantic model")
    if cap.get("streaming", {}).get("processor") != "bytewax":
        errors.append("streaming processor must be bytewax")
    if "agent_workflow" not in cap.get("provides", []):
        errors.append("agent_workflow provide missing")
    if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
        errors.append("component manifest semantic model interface mismatch")

    return {
        "passed":     not errors,
        "status":     "ok" if not errors else "failed",
        "errors":     errors,
        "capability": CAPABILITY_ID,
    }


# ── Flask application ─────────────────────────────────────────────────────────

try:
    from flask import Flask, jsonify, request

    def create_app(config: dict | None = None) -> Flask:
        app = Flask(__name__)
        if config:
            app.config.update(config)

        # Wire adapters and store
        from .domain.adapters import (
            get_auth_adapter, get_audit_adapter,
            get_notify_adapter, get_workflow_adapter,
        )
        from .database.store import get_store

        db_url = (config or {}).get("DB_URL") or os.environ.get("APG_DATABASE_URL")
        store  = get_store(db_url)

        try:
            from .service import MyCapabilityService
            svc = MyCapabilityService()
            app.config["SERVICE"] = svc
        except Exception:
            pass

        try:
            from .api import blueprint as api_bp
            app.register_blueprint(api_bp, url_prefix="/api/v1")
        except (ImportError, AttributeError):
            pass

        try:
            from .views import blueprint as views_bp
            app.register_blueprint(views_bp)
        except (ImportError, AttributeError):
            pass

        @app.get("/health")
        def health():
            return jsonify({
                "status":     "ok",
                "capability": CAPABILITY_ID,
                "version":    CAPABILITY_VERSION,
                "standalone": True,
            })

        @app.get("/contract")
        def contract():
            return jsonify(get_capability_contract())

        @app.post("/evaluate")
        def evaluate():
            ctx = request.get_json(force=True, silent=True) or {}
            return jsonify(evaluate_capability_rules(ctx))

        @app.get("/semantic-model.json")
        def semantic_model_route():
            return jsonify(semantic_model())

        @app.get("/self-test")
        def self_test_route():
            result = self_test()
            return jsonify(result), (200 if result["passed"] else 500)

        @app.get("/component.json")
        def component_manifest_route():
            return jsonify(component_manifest())

        return app

except ImportError:
    def create_app(config=None):  # type: ignore[misc]
        raise ImportError("flask is required: pip install flask")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=f"APG {CAPABILITY_NAME} standalone server"
    )
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   type=int, default=8080)
    parser.add_argument("--debug",  action="store_true")
    parser.add_argument("--db-url", default=None,
                        help="PostgreSQL URL (optional; default: in-memory)")
    parser.add_argument("--tenant", default="default")
    args = parser.parse_args(argv)

    app = create_app({"DB_URL": args.db_url, "DEFAULT_TENANT": args.tenant})
    print(f"APG {CAPABILITY_NAME} v{CAPABILITY_VERSION}")
    print(f"  Store:    {'PostgreSQL' if args.db_url else 'InMemory'}")
    print(f"  Listening: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
```

---

## 10. Testing Guide

### Test file organisation

```
tests/
├── __init__.py
├── test_contract.py          # contract shape + rule engine
├── test_service.py           # full service lifecycle + guardrails
└── test_api_views_app.py     # api.py, views.py, app.py integration
```

All tests go in `tests/`. Passing CI tests go in `tests/ci/` for autodiscovery.

### Loading the module under test without installing it

```python
import importlib.util
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
```

### test_contract.py

```python
from pathlib import Path
import pytest
from capabilities.capability_contract_registry import validate_contract_shape

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def test_contract_shape_passes_registry_validation():
    module   = _load_module("cap_contract", PACKAGE_DIR / "capability_contract.py")
    contract = module.get_capability_contract("test-tenant")

    # Registry validates these — any failure here means the contract is broken
    validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")

    assert contract["capability"] == "domain_code"
    assert contract["streaming"]["processor"] == "bytewax"
    assert "agent_workflow" in contract["provides"]
    assert contract["theme"]["tokens"]["border.radius"] == "8px"
    assert contract["configuration"]["agents"]["supported_runtimes"] == \
        ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_tenant_context():
    module = _load_module("cap_rules", PACKAGE_DIR / "capability_contract.py")
    result = module.evaluate_capability_rules({"tenant_context_present": False})
    assert result["decision"] == "deny"
    deny_reasons = [a["reason"] for a in result["actions"]]
    assert "tenant_context_required" in deny_reasons


def test_rule_engine_blocks_non_bytewax_batch():
    module = _load_module("cap_rules2", PACKAGE_DIR / "capability_contract.py")
    result = module.evaluate_capability_rules({
        "tenant_id": "t", "tenant_context_present": True,
        "operation": "batch_mutation", "event_stream": "sqs",
    })
    assert result["decision"] == "deny"


def test_rule_engine_allows_valid_read():
    module = _load_module("cap_rules3", PACKAGE_DIR / "capability_contract.py")
    result = module.evaluate_capability_rules({
        "tenant_id": "t", "tenant_context_present": True,
        "operation_type": "read",
    })
    assert result["decision"] == "allow"
```

### test_service.py

```python
from pathlib import Path
import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def test_full_lifecycle():
    svc_mod = _load_module("svc", PACKAGE_DIR / "service.py")
    svc     = svc_mod.MyCapabilityService()

    record = svc.create_record(
        "rec-1", "tenant-test", "type_a", "confidential",
        "ref-1", "high", "owner-1", "evidence-1",
    )
    assert record["record_type"] == "type_a"

    review = svc.record_review(
        "rev-1", "tenant-test", record["id"],
        "reviewer-1", "approved", "review-evidence",
    )
    assert review["status"] == "approved"

    summary = svc.dashboard_summary("tenant-test")
    assert summary["record_count"] == 1
    assert summary["audit_event_count"] == 2


def test_tenant_isolation():
    svc_mod = _load_module("svc2", PACKAGE_DIR / "service.py")
    svc     = svc_mod.MyCapabilityService()

    svc.create_record("rec-a", "tenant-a", "type_a", "confidential", "r", "low", "o", "e")
    svc.create_record("rec-b", "tenant-b", "type_a", "confidential", "r", "low", "o", "e")

    assert svc.dashboard_summary("tenant-a")["record_count"] == 1
    assert svc.dashboard_summary("tenant-b")["record_count"] == 1


def test_guardrails_reject_missing_tenant():
    svc_mod = _load_module("svc3", PACKAGE_DIR / "service.py")
    svc     = svc_mod.MyCapabilityService()

    with pytest.raises(PermissionError, match="tenant_context_required"):
        svc.create_record("x", "", "type_a", "confidential", "r", "high", "o", "e")


def test_guardrails_reject_unsupported_type():
    svc_mod = _load_module("svc4", PACKAGE_DIR / "service.py")
    svc     = svc_mod.MyCapabilityService()

    with pytest.raises(PermissionError, match="record_type_not_supported"):
        svc.create_record("x", "t", "unknown_type", "confidential", "r", "high", "o", "e")


def test_guardrails_reject_non_bytewax_batch():
    svc_mod = _load_module("svc5", PACKAGE_DIR / "service.py")
    svc     = svc_mod.MyCapabilityService()

    with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
        svc.validate_batch("t", 10, event_stream="rabbitmq")


def test_guardrails_agent_privileged_requires_approval():
    svc_mod = _load_module("svc6", PACKAGE_DIR / "service.py")
    svc     = svc_mod.MyCapabilityService()

    with pytest.raises(PermissionError, match="human_approval_required"):
        svc.validate_agent_action("t", privileged_scope=True, human_approval_recorded=False)
```

### test_api_views_app.py

```python
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def test_api_views_app_are_executable():
    api   = _load_module("api",   PACKAGE_DIR / "api.py")
    views = _load_module("views", PACKAGE_DIR / "views.py")
    app   = _load_module("app",   PACKAGE_DIR / "app.py")

    record = api.create_record({
        "record_id": "api-rec", "tenant_id": "api-tenant",
        "record_type": "type_a", "classification": "unclassified",
        "reference": "ref", "severity": "low",
        "owner_id": "owner", "evidence_reference": "evidence",
    })
    agent  = api.register_agent({
        "agent_id": "ag-1", "tenant_id": "api-tenant",
        "name": "Test Agent", "runtime": "codex",
        "role": "reviewer", "scope": "read-only review",
    })
    dash   = views.dashboard_model(api.service(), "api-tenant")
    result = app.self_test()
    model  = app.semantic_model()

    assert record["record_type"] == "type_a"
    assert agent["runtime"] == "codex"
    assert dash["summary"]["record_count"] == 1
    assert result["passed"] is True
    assert model["format"] == "apg.semantic-model.v1"
```

### Running tests

```bash
# All tests in the capability directory
pytest tests/ -q

# CI subset only
pytest tests/ci/ -q

# Single file
pytest tests/test_contract.py -vxs

# With coverage
pytest tests/ --cov=. --cov-report=term-missing -q
```

**No mocks.** Use the `InMemoryStore` directly. The Null adapters handle all platform dependencies. The only acceptable mock is for LLM API calls.

---

## 11. Standalone Development Workflow

### Step 1: Create the directory structure

```bash
DOMAIN=intel
CODE=myfeature
mkdir -p capabilities/$DOMAIN/$CODE/{domain,database,tests,templates,static,alembic/versions}
touch capabilities/$DOMAIN/$CODE/{__init__,__main__,capability_contract,models,service,api,views,app,alerts_runtime}.py
touch capabilities/$DOMAIN/$CODE/domain/{__init__,adapters,rules,events}.py
touch capabilities/$DOMAIN/$CODE/database/{__init__,store}.py
touch capabilities/$DOMAIN/$CODE/tests/__init__.py
touch capabilities/$DOMAIN/$CODE/py.typed
```

### Step 2: Write capability_contract.py

Start from the template in Section 2. The minimum viable contract requires:
- `CAPABILITY_ID` that is unique across the repo
- At least 1 rule
- At least 1 route with path starting `/`
- `theme.tokens["border.radius"]`
- `ui.requires_theme = True`
- `configuration_schema.required` containing `tenant_id`, `ui`, `theme`

### Step 3: Validate the contract standalone

```bash
python -c "
import sys; sys.path.insert(0, '.')
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('c', 'capabilities/intel/myfeature/capability_contract.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
c = m.get_capability_contract('test')
print('capability:', c['capability'])
print('rules:', len(c['rule_engine']['rules']))
print('routes:', len(c['ui']['routes']))
print('OK')
"
```

### Step 4: Run the registry validator

```bash
python -c "
from capabilities.capability_contract_registry import validate_contract_registry
result = validate_contract_registry()
print('valid:', result['valid'])
print('capabilities:', result['capabilities'])
if result['errors']:
    for e in result['errors']: print('ERROR:', e)
"
```

### Step 5: Write pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "apg-intel-myfeature"
version = "1.0.0"
description = "APG My Feature capability"
requires-python = ">=3.11"
dependencies = ["flask>=3.0"]

[project.scripts]
apg-intel-myfeature = "apg_intel_myfeature.app:main"

[project.entry-points."apg.capabilities"]
intel_myfeature = "apg_intel_myfeature.capability_contract:get_capability_contract"

[tool.setuptools.packages.find]
where = ["."]
include = ["apg_intel_myfeature*"]
```

### Step 6: Build the wheel

```bash
python -m build --wheel .
```

### Step 7: Install and run standalone

```bash
pip install dist/apg_intel_myfeature-1.0.0-py3-none-any.whl
apg-intel-myfeature --port 8080
# or without installing:
python -m apg_intel_myfeature --port 8080
```

### Step 8: Verify endpoints

```bash
# Liveness
curl -s http://localhost:8080/health | python -m json.tool

# Contract
curl -s http://localhost:8080/contract | python -m json.tool

# Rule evaluation
curl -s -X POST http://localhost:8080/evaluate \
  -H "Content-Type: application/json" \
  -d '{"tenant_context_present": false}' | python -m json.tool
# expect: {"decision": "deny", ...}

curl -s -X POST http://localhost:8080/evaluate \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "t", "tenant_context_present": true}' | python -m json.tool
# expect: {"decision": "allow", ...}

# Semantic model
curl -s http://localhost:8080/semantic-model.json | python -m json.tool

# Self-test
curl -s http://localhost:8080/self-test | python -m json.tool
# expect: {"passed": true, ...}
```

### Step 9: Run tests

```bash
pytest tests/ -q
```

---

## 12. Integration Patterns

### Pattern 1: Require platform capabilities via adapters

The service constructor accepts adapter overrides. When not provided, `get_*_adapter()` auto-discovers the installed platform capability or falls back to the Null implementation.

```python
# service.py
from .domain.adapters import (
    get_auth_adapter, get_audit_adapter,
    get_notify_adapter, get_workflow_adapter,
    AuthAdapter, AuditAdapter, NotifyAdapter, WorkflowAdapter,
)

class MyCapabilityService:
    def __init__(
        self,
        auth:     AuthAdapter     | None = None,
        audit:    AuditAdapter    | None = None,
        notify:   NotifyAdapter   | None = None,
        workflow: WorkflowAdapter | None = None,
    ) -> None:
        self._auth     = auth     or get_auth_adapter()
        self._audit    = audit    or get_audit_adapter()
        self._notify   = notify   or get_notify_adapter()
        self._workflow = workflow or get_workflow_adapter()
        # ... rest of __init__

    async def protected_operation(self, actor_token: str, record_id: str) -> dict:
        user = await self._auth.verify_token(actor_token)
        allowed = await self._auth.check_permission(
            user["user_id"], "domain_code:write", record_id,
        )
        if not allowed:
            raise PermissionError("insufficient_permission")
        # proceed with operation
        result = self._do_operation(record_id, user)
        await self._audit.log_event(
            "operation_completed", user["user_id"], user["tenant_id"],
            record_id, {"operation": "protected_operation"},
        )
        return result
```

### Pattern 2: Consume Bytewax streaming events from another capability

When your capability depends on lifecycle events emitted by another capability's `STREAMING.stream`, subscribe to the Bytewax topic declared in that capability's contract.

```python
# In your domain/rules.py or a dedicated stream_consumer.py
from capabilities.capability_contract_registry import get_contract

def get_upstream_stream(capability_id: str = "intel_alerts") -> str:
    contract = get_contract(capability_id)
    return contract["streaming"]["stream"]
    # returns e.g. "apg.intel.alerts.lifecycle"


# Bytewax dataflow example (requires bytewax package)
import bytewax.operators as op
from bytewax.dataflow import Dataflow

def build_consumer_dataflow(upstream_stream: str) -> Dataflow:
    flow = Dataflow("my_consumer")
    # Subscribe to the upstream capability's stream
    inp = op.input("kafka_in", flow, KafkaSourceConfig(
        brokers=["localhost:9092"],
        topics=[upstream_stream],
    ))
    # Process events
    filtered = op.filter("alert_events", inp,
        lambda msg: msg.get("event_type") in ("alert_recorded", "alert_escalation_recorded")
    )
    op.output("downstream_sink", filtered, MySinkConfig())
    return flow
```

**Guardrail:** Batch mutations into any capability's stream must use Bytewax. The `batch_mutation_requires_bytewax` rule (or equivalent) in the target capability will deny any batch request using a different `event_stream` value.

### Pattern 3: Declare a capability dependency in an APG application

When composing multiple capabilities into an APG application, declare dependencies by listing them in `REQUIRES`. The registry and composition tooling will:
1. Resolve the dependency graph from all loaded contracts.
2. Verify that every item in `REQUIRES` has a loaded contract with a matching `CAPABILITY_ID`.
3. Wire adapter factory calls in `create_app`.

```python
# In your capability_contract.py
REQUIRES = [
    "auth",          # capabilities/common/auth
    "audl",          # capabilities/common/audit_log
    "intel_alerts",  # capabilities/intel/alerts — upstream capability
]

# In your app.py create_app()
def create_app(config=None):
    from .domain.adapters import get_auth_adapter, get_audit_adapter
    # For intel_alerts, import its service directly if co-located
    try:
        from capabilities.intel.alerts.service import AlertManagementService
        alert_svc = AlertManagementService()
    except ImportError:
        alert_svc = None

    svc = MyCapabilityService(
        auth=get_auth_adapter(),
        upstream_alerts=alert_svc,
    )
    # ...
```

---

## 13. Quality Checklist

Use this before merging a new capability or shipping a release.

### Contract completeness

- [ ] `CAPABILITY_ID` is unique across the repository (`grep -r 'CAPABILITY_ID' capabilities/`)
- [ ] `get_capability_contract()` returns without error for any non-empty `tenant_id`
- [ ] Registry validator passes: `validate_contract_registry()` returns `{"valid": true}`
- [ ] 20+ governance rules covering all operations that can be denied
- [ ] All rules have non-empty `name`, `condition`, `effect.decision`, `effect.reason`
- [ ] `_ne` suffix used correctly for inequality conditions (e.g. `event_stream_ne`)
- [ ] 8+ UI routes, all paths start with `/`
- [ ] `theme.tokens["border.radius"]` present
- [ ] `ui.requires_theme` is `True`
- [ ] `configuration_schema.required` contains `tenant_id`, `ui`, `theme`
- [ ] `streaming.processor` is `"bytewax"`

### Service completeness

- [ ] 40+ methods implemented (no `pass`, no `raise NotImplementedError`)
- [ ] Every mutating method calls `_enforce(context)` before writing state
- [ ] Every mutating method calls `_audit(tenant_id, event_type, reference_id)` after writing
- [ ] Tenant isolation: all stores keyed by `(tenant_id, id)`
- [ ] `describe(tenant_id)` and `evaluate(context)` methods present
- [ ] `dashboard_summary(tenant_id)` returns counts for all entity types
- [ ] Analytics methods: severity distribution, throughput, MTTR, SLA compliance

### Models

- [ ] Every model has `to_dict()` returning a plain `dict`
- [ ] No ORM objects leak out of service methods — only dicts

### Adapters

- [ ] One Protocol + one Null* + one factory function per `REQUIRES` entry
- [ ] Null adapters never raise exceptions; they degrade gracefully
- [ ] `_Installed*` wrappers exist for each real platform package

### API / Views / App

- [ ] `api.py` exposes a function for every mutating service method
- [ ] `views.py` has a model function for every UI route
- [ ] `app.py` responds to: `/health`, `/contract`, `/evaluate`, `/semantic-model.json`, `/self-test`, `/component.json`
- [ ] `self_test()` returns `{"passed": true}` with the real contract

### Testing

- [ ] `pytest tests/ -q` exits 0
- [ ] `test_contract.py` runs `validate_contract_shape` from the registry
- [ ] Tests cover: happy path lifecycle, tenant isolation, every guardrail rule
- [ ] No mocks (except LLM calls) — `InMemoryStore` and Null adapters only

### Packaging

- [ ] `pyproject.toml` present with `apg.capabilities` entry point
- [ ] `py.typed` marker present
- [ ] `__init__.py` exports `__version__`, `__capability_id__`, `get_capability_contract`, `evaluate_capability_rules`
- [ ] `__main__.py` calls `app.main()`
- [ ] `python -m build --wheel .` succeeds
- [ ] Installed wheel: `<package-name> --port 8080` starts without errors
- [ ] `curl http://localhost:8080/health` returns `{"status": "ok", ...}`
- [ ] `curl http://localhost:8080/contract` returns the full contract
- [ ] `curl http://localhost:8080/self-test` returns `{"passed": true}`
