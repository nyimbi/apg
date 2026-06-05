# Contributing to APG

APG (Agentic Platform for Governance) is a capability-driven platform built by Datacraft. This document covers everything you need to contribute a new capability, fix a bug, or extend existing functionality.

---

## 1. Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- PostgreSQL 15+
- Git

### Bootstrap

```bash
git clone <repo-url> apg && cd apg
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Verify

```bash
uv run pytest tests/ci -q          # all CI-passing tests
uv run pyright                      # type-check
```

---

## 2. Building New Capabilities

### Scaffold

Every capability lives under `capabilities/<domain>/<name>/`. Use the dev command to scaffold:

```bash
/dev <domain>/<name>
```

Or manually create the required files (see below).

### Required Files

| File | Purpose |
|---|---|
| `capability_contract.py` | Executable contract — **the single source of truth** |
| `service.py` | Async service layer, main business logic |
| `models.py` | Pydantic v2 domain models |
| `views.py` | Flask-AppBuilder view models / API schemas |
| `api.py` | Route handlers and API helpers |
| `app.py` | Blueprint registration |
| `tests/` | Unit, integration, composition, and APG tests |
| `docs/` | User guides and reference documentation |

### Capability Contract

`capability_contract.py` must expose:

```python
def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]: ...
def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]: ...
```

The contract dict must include all of: `capability`, `configuration`, `configuration_schema`, `rule_engine`, `ui`, `theme`, `streaming`.

Model prefixes: use a 2–4 character prefix specific to the capability (e.g. `auth_`, `pay_`, `ia_`).

---

## 3. Quality Standards

### Method Count

Each service class must implement **40+ methods**. Structure them as:

- Lifecycle CRUD (create, read, update, delete, list)
- Workflow methods (approve, reject, escalate, assign, resolve)
- Validation helpers (prefixed `_validate_`)
- Logging helpers (prefixed `_log_`)
- Streaming/event emission

### Rule Count

Each capability contract must define **20+ deterministic governance rules** in `rule_engine.rules`. Every rule must have:

- `name` — non-empty string, snake_case
- `condition` — dict of field matchers
- `effect.decision` — one of: `allow`, `deny`, `require_review`, `warn`, `audit`, `quarantine`, `challenge`

### Streaming Events

Every capability must declare a `streaming` section with:

```python
{
    "processor": "bytewax",
    "stream": "apg.<domain>.<name>.lifecycle",
    "key": "tenant_id",
    "events": [...],       # lifecycle event names
    "guardrails": [...],   # rule names that gate batch operations
}
```

### Theme Token

Every capability theme must include `border.radius` in `tokens`. This is enforced by the contract validator.

---

## 4. Testing

### Running Tests

```bash
uv run pytest tests/ci -q              # CI suite
uv run pytest tests/ -q                # full suite including integration
uv run pytest tests/test_security_integration.py -v   # governance tests
```

### Test Categories

| Directory / file pattern | Purpose |
|---|---|
| `tests/ci/` | Passing tests, run in CI on every push |
| `tests/test_*_integration.py` | Integration tests, real objects, no mocks |
| `capability_contract.py` tests | APG contract shape and rule correctness |
| `tests/fixtures/` | Shared pytest fixtures |

### Rules

- No mocks for domain logic. Use `pytest` fixtures + real objects.
- Use `pytest-httpserver` for HTTP boundary tests.
- Async tests use plain `async def` — no `@pytest.mark.asyncio` decorator needed.
- All passing tests must live in `tests/ci/` for CI autodiscovery.
- Tests go in `tests/`, docs in `docs/`, todo/planning files in `works/`.

### Capability Test Checklist

Every new capability must ship tests covering:

1. Contract shape validation (`validate_contract_shape`)
2. Rule evaluation — deny paths, allow paths, require_review paths
3. Tenant isolation — `cross_tenant_access=True` → deny
4. Missing tenant context → deny
5. Write without policy → deny or require_review
6. Valid context → allow
7. Service CRUD happy paths
8. At least one composition test (capability used by another)

---

## 5. Pull Request Process

### Before Opening a PR

```bash
uv run pytest tests/ci -q           # must be green
uv run pyright                       # zero errors
```

### PR Requirements

- Title: imperative mood, ≤ 72 characters (`Add fintech/payments capability contract`)
- Description: what changed, why, and a test plan checklist
- All CI checks must pass before review
- At least one approving review from a Datacraft maintainer
- Squash-merge into `main`

### CI Checks

- `pytest tests/ci` — unit and integration suite
- `pyright` — static type checking
- Contract validator — runs `validate_contract_registry()` against all discovered capabilities

---

## 6. APG Language Guide

APG programs are declarative compositions of capabilities. Syntax basics:

```apg
compose MyApp {
  capabilities: [auth, fintech_payments, intel_alerts]
  tenant: "acme"

  wire auth -> fintech_payments via "identity_registry"
  wire intel_alerts -> fintech_payments via "alert_signal_workflow"

  rules: inherit_all
}
```

Key concepts:

- **compose** — declares an application composed from capabilities
- **wire** — routes a `provides` surface of one capability to a `requires` of another
- **rules: inherit_all** — all governance rules from every wired capability are active
- Capability IDs match the `capability` field in each contract
- `provides` and `requires` lists in each contract define legal wire endpoints

See `docs/apg_language.md` for the full grammar.

---

## 7. Capability Contract Reference

### Top-Level Keys

| Key | Required | Description |
|---|---|---|
| `capability` | yes | Unique snake_case ID, matches directory name |
| `display_name` | yes | Human-readable name |
| `provides` | yes | List of surface IDs this capability exports |
| `requires` | yes | List of capability IDs this capability depends on |
| `configuration` | yes | Tenant-scoped default config dict (must contain `tenant_id`) |
| `configuration_schema` | yes | JSON Schema; `required` must include `tenant_id`, `ui`, `theme` |
| `rule_engine` | yes | `{type: "deterministic", rules: [...]}` |
| `ui` | yes | UI manifest (see below) |
| `theme` | yes | Theme token dict (see below) |
| `streaming` | yes | Bytewax stream manifest |

### UI Manifest

```python
{
    "shell": "apg_python",
    "requires_theme": True,           # always True
    "template_roots": ["templates/"],
    "routes": [
        {
            "name": "dashboard",
            "path": "/capability/dashboard",   # must start with /
            "component": "CapabilityDashboard",
            "permission": "domain:action",     # namespaced; use "public" for open routes only
        },
        ...
    ],
}
```

All route permissions must follow `domain:action` or `domain.action` namespacing. The only valid unnested value is `"public"` for genuinely unauthenticated routes.

### Theme Tokens

Minimum required token: `border.radius`. Full set:

```python
{
    "color.primary": "#...",
    "color.accent": "#...",
    "color.success": "#...",
    "color.warning": "#...",
    "color.danger": "#...",
    "surface.canvas": "#...",
    "surface.panel": "#...",
    "text.primary": "#...",
    "text.secondary": "#...",
    "border.radius": "8px",    # REQUIRED
    "density": "comfortable",
}
```

---

## 8. Versioning

Capabilities use [Semantic Versioning](https://semver.org/):

- **MAJOR** — breaking change to `provides`, `requires`, or contract shape
- **MINOR** — new rules, new routes, new streaming events (backward compatible)
- **PATCH** — bug fixes, clarifications, non-breaking rule condition adjustments

Version is declared in `CAPABILITY_VERSION = "x.y.z"` at the top of `capability_contract.py`.

The platform version in `MANIFEST.json` follows the same scheme and is bumped when any capability version changes.

---

## 9. Code Style

### Python

- Python **3.12+** only
- **Tabs** for indentation (not spaces)
- `async`/`await` throughout — no synchronous I/O in service layer
- Modern union types: `str | None`, `list[str]`, `dict[str, Any]`
- Pydantic **v2** models:

```python
from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated
from pydantic.functional_validators import AfterValidator

class MyModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
    )
    id: str = Field(default_factory=uuid7str)
```

### IDs

UUID7 strings via the project shim:

```python
from situ_cloudevents._uuid7 import uuid7str

id: str = Field(default_factory=uuid7str)
```

### Logging

Log-formatting helpers must be prefixed `_log_`:

```python
def _log_pretty_path(self, path: Path) -> str: ...
```

### Assertions

Runtime precondition/postcondition checks at function boundaries:

```python
assert isinstance(tenant_id, str) and tenant_id, "tenant_id required"
```

### Database

PostgreSQL exclusively. ORM: SQLAlchemy 2.x with async sessions. Every model file ships a corresponding SQL DDL script in `docs/sql/`.

### UI Framework

Flask-AppBuilder blueprints. Every capability's `app.py` registers a blueprint with prefix `/<capability_id>`.

---

## 10. License and Copyright

```
Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
```

APG is proprietary software. All rights reserved. Contributions submitted via pull request are assigned to Datacraft under the project's contributor license agreement.
